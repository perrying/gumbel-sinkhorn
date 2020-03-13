import os, sys
import logging
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import ConvModel
from puzzle_utils import batch_tch_divide_image
from dataset_builder import build_dataset

sys.path.append(os.pardir)
from utils import gumbel_sinkhorn_ops, metric

def train(cfg):
    logger = logging.getLogger("JigsawPuzzle")
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"

    if cfg.dataset == "MNIST":
        in_c = 1
    else:
        in_c = 3

    train_data = build_dataset(cfg, "train")

    model = ConvModel(in_c, cfg.pieces, cfg.image_size, cfg.hid_c, cfg.stride, cfg.kernel_size).to(device)
    optimizer = optim.Adam(model.parameters(), cfg.lr, eps=1e-8)

    train_loader = DataLoader(train_data, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)

    logger.info("start training")
    for epoch in range(1, cfg.epochs+1):
        sum_loss = 0
        for i, data in enumerate(train_loader):
            inputs, _ = data
            pieces, random_pieces, _ = batch_tch_divide_image(inputs, cfg.pieces)
            pieces, random_pieces = pieces.to(device), random_pieces.to(device)

            log_alpha = model(random_pieces)

            gumbel_sinkhorn_mat = [
                gumbel_sinkhorn_ops.gumbel_sinkhorn(log_alpha, cfg.tau, cfg.n_sink_iter)
                for _ in range(cfg.n_samples)
            ]

            est_ordered_pieces = [
                gumbel_sinkhorn_ops.inverse_permutation_for_image(random_pieces, gs_mat)
                for gs_mat in gumbel_sinkhorn_mat
            ]

            loss = sum([
                torch.nn.functional.mse_loss(X, pieces)
                for X in est_ordered_pieces
            ])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()

            if cfg.display > 0 and ((i+1) % cfg.display) == 0:
                logger.info("epoch %i [%i/%i] loss %f", epoch, i+1, len(train_loader), loss.item())
        logger.info("epoch %i|  mean loss %f", epoch, sum_loss/len(train_loader))

    torch.save(model.state_dict(), os.path.join(cfg.out_dir, "model.pth"))

def evaluation(cfg):
    logger = logging.getLogger("JigsawPuzzle")
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"
    if cfg.dataset == "MNIST":
        in_c = 1
    else:
        in_c = 3
    model = ConvModel(in_c, cfg.pieces, cfg.image_size, cfg.hid_c, cfg.stride, cfg.kernel_size)
    model.load_state_dict(torch.load(os.path.join(cfg.out_dir, "model.pth")))
    model = model.to(device)

    eval_data = build_dataset(cfg, split="test")
    loader = DataLoader(eval_data, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, drop_last=False)
    with torch.no_grad():
        l1_diffs = []
        l2_diffs = []
        prop_wrongs = []
        prop_any_wrongs = []
        kendall_taus = []
        logger.info("start evaluation")
        for data in loader:
            inputs, _ = data
            pieces, random_pieces, perm_index = batch_tch_divide_image(inputs, cfg.pieces)
            pieces, random_pieces = pieces.to(device), random_pieces.to(device)

            log_alpha = model(random_pieces)

            gumbel_matching_mat = gumbel_sinkhorn_ops.gumbel_matching(log_alpha, noise=False)

            hard_sorted_pieces = gumbel_sinkhorn_ops.inverse_permutation_for_image(random_pieces, gumbel_matching_mat)

            est_perm_index = gumbel_matching_mat.max(1)[1].float()

            hard_l1_diff = (pieces - hard_sorted_pieces).abs().mean((2,3,4)) # (batchsize, num_pieces)
            hard_l2_diff = (pieces - hard_sorted_pieces).pow(2).mean((2,3,4))
            sign_l1_diff = hard_l1_diff.sign()
            prop_wrong = sign_l1_diff.mean(1)
            prop_any_wrong = sign_l1_diff.sum(1).sign()

            np_perm_index = perm_index.detach().numpy()
            np_est_perm_index = est_perm_index.to("cpu").numpy()
            kendall_tau = metric.kendall_tau(np_est_perm_index, np_perm_index)

            l1_diffs.append(hard_l1_diff); l2_diffs.append(hard_l2_diff)
            prop_wrongs.append(prop_wrong); prop_any_wrongs.append(prop_any_wrong)
            kendall_taus.append(kendall_tau)

        mean_l1_diff = torch.cat(l1_diffs).mean()
        mean_l2_diff = torch.cat(l2_diffs).mean()
        mean_prop_wrong = torch.cat(prop_wrongs).mean()
        mean_prop_any_wrong = torch.cat(prop_any_wrongs).mean()
        mean_kendall_tau = np.concatenate(kendall_taus).mean()
        logger.info("\nmean l1 diff : %f\n mean l2 diff : %f\n mean prop wrong : %f\n mean prop any wrong : %f\n mean kendall tau : %f",
            mean_l1_diff, mean_l2_diff, mean_prop_wrong, mean_prop_any_wrong, mean_kendall_tau
        )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # directory option
    parser.add_argument("--root", "-r", default="./data", type=str, help="dataset root directory")
    parser.add_argument("--out_dir", "-o", default="./log", type=str, help="output directory")
    # optimizer option
    parser.add_argument("--epochs", "-e", default=10, type=int, help="number of epochs")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--batch_size", default=32, type=int, help="mini-batch size")
    parser.add_argument("--num_workers", default=8, type=int, help="number of threads for CPU parallel")
    # dataset option
    parser.add_argument("--dataset", default="MNIST", type=str, help="dataset name chosen from ['MNIST',]")
    parser.add_argument("--pieces", "-p", default=2, type=int, help="number of pieces each side")
    parser.add_argument("--image_size", default=28, type=int, help="original image size")
    # model parameter option
    parser.add_argument("--hid_c", default=64, type=int, help="number of hidden channels")
    parser.add_argument("--stride", default=2, type=int, help="stride in pooling operator")
    parser.add_argument("--kernel_size", default=5, type=int, help="kernel size in convolution operator")
    # Gumbel sinkhorn option
    parser.add_argument("--tau", default=1.0, type=float, help="temperture parameter")
    parser.add_argument("--n_sink_iter", default=20, type=int, help="number of iterations for sinkhorn normalization")
    parser.add_argument("--n_samples", default=5, type=int, help="number of samples from gumbel-sinkhorn distribution")
    # misc option
    parser.add_argument("--display", default=50, type=int, help="display loss every 'display' iteration. if set to 0, won't display")
    parser.add_argument("--eval_only", action="store_true", help="evaluation without training")

    cfg = parser.parse_args()

    if not os.path.exists(cfg.out_dir):
        os.mkdir(cfg.out_dir)

    # logger setup
    logging.basicConfig(
        filename=os.path.join(cfg.out_dir, "console.log"),
    )
    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )
    logger = logging.getLogger("JigsawPuzzle")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = plain_formatter
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if not cfg.eval_only:
        train(cfg)
    
    evaluation(cfg)
