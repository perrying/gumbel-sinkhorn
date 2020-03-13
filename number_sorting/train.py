import logging
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model import FCModel
from number_utils import NumberGenerator
import os, sys
sys.path.append(os.pardir)
from utils import gumbel_sinkhorn_ops

def train(cfg):
    logger = logging.getLogger("NumberSorting")
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"
    
    model = FCModel(cfg.hid_c, cfg.n_numbers).to(device)
    optimizer = optim.Adam(model.parameters(), cfg.lr)

    dataset = NumberGenerator(cfg.n_numbers, cfg.n_train_lists, cfg.min_value, cfg.max_value, cfg.train_seed)
    train_loader = DataLoader(dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)

    logger.info("training")
    for epoch in range(cfg.epochs):
        for i, data in enumerate(train_loader):
            X, ordered_X, _ = data
            X = X.to(device)
            ordered_X = ordered_X.to(device)

            log_alpha = model(X[:,None])

            gumbel_sinkhorn_mat = [
                gumbel_sinkhorn_ops.gumbel_sinkhorn(log_alpha, cfg.tau, cfg.n_sink_iter)
                for _ in range(cfg.n_samples)
            ]

            est_ordered_X = [
                gumbel_sinkhorn_ops.inverse_permutation(X, gs_mat)
                for gs_mat in gumbel_sinkhorn_mat
            ]

            loss = sum([
                torch.nn.functional.mse_loss(X, ordered_X)
                for X in est_ordered_X
            ])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ((i+1) % cfg.display) == 0:
                logger.info("%i epoch [%i/%i] training loss %f", epoch, i+1, len(train_loader), loss.item())

    torch.save(model.state_dict(), os.path.join(cfg.out_dir, "model_weight.pth"))

def evaluation(cfg):
    logger = logging.getLogger("NumberSorting")
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"
    model = FCModel(cfg.hid_c, cfg.n_numbers).to(device)
    model.load_state_dict(torch.load(os.path.join(cfg.out_dir, "model_weight.pth")))

    dataset = NumberGenerator(cfg.n_numbers, cfg.n_test_lists, cfg.min_value, cfg.max_value, cfg.test_seed)
    test_loader = DataLoader(dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)

    logger.info("evaluation")
    prop_wrongs = []
    prop_any_wrongs = []
    model.eval()
    for data in test_loader:
        X, ordered_X, permutation = data
        X = X.to(device)
        ordered_X = ordered_X.to(device)
        permutation = permutation.to(device)

        log_alpha = model(X[:,None])

        assingment_matrix = gumbel_sinkhorn_ops.gumbel_matching(log_alpha, noise=False)

        est_permutation = assingment_matrix.max(1)[1].float()

        prop_wrong = (permutation - est_permutation).sign().abs().mean(1)
        prop_any_wrong = (permutation - est_permutation).sign().abs().sum(1).sign()

        prop_wrongs.append(prop_wrong)
        prop_any_wrongs.append(prop_any_wrong)

    mean_prop_wrong = torch.cat(prop_wrongs).mean()
    mean_prop_any_wrong = torch.cat(prop_any_wrongs).mean()

    logger.info("Mean Prop Wrongs %f", mean_prop_wrong)
    logger.info("Mean Prop Any Wrongs %f", mean_prop_any_wrong)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # gumbel sinkhorn option
    parser.add_argument("--tau", default=1.0, type=float, help="temperture parameter")
    parser.add_argument("--n_sink_iter", default=20, type=int, help="number of iterations for sinkhorn normalization")
    parser.add_argument("--n_samples", default=5, type=int, help="number of samples from gumbel-sinkhorn distribution")
    # datase option
    parser.add_argument("--n_numbers", default=100, type=int, help="number of sorted numbers")
    parser.add_argument("--n_train_lists", default=10000, type=int, help="number of sorted number lists for training")
    parser.add_argument("--n_test_lists", default=100, type=int, help="number of sorted number lists for evaluation")
    parser.add_argument("--min_value", default=0, type=float, help="minimum value of uniform distribution")
    parser.add_argument("--max_value", default=1, type=float, help="maximum value of uniform distribution")
    parser.add_argument("--train_seed", default=1, type=int, help="random seed for training data generation")
    parser.add_argument("--test_seed", default=2, type=int, help="random seed for evaluation data generation")
    parser.add_argument("--num_workers", default=8, type=int, help="number of threads for CPU parallel")
    # optimizer option
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--batch_size", default=100, type=int, help="mini-batch size")
    parser.add_argument("--epochs", default=100, type=int, help="number of epochs")
    # misc
    parser.add_argument("--hid_c", default=64, type=int, help="number of hidden channels")
    parser.add_argument("--out_dir", default="log", type=str, help="/path/to/output directory")
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
    logger = logging.getLogger("NumberSorting")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = plain_formatter
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info(cfg)

    if not cfg.eval_only:
        train(cfg)
    evaluation(cfg)