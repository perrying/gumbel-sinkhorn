from torchvision import datasets
from torchvision import transforms

def build_dataset(cfg, split):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    if cfg.dataset == "MNIST":
        dataset = datasets.MNIST(cfg.root, train = (split == "train"), download=True, transform=transform)
    elif cfg.dataset == "CIFAR-10":
        dataset = datasets.CIFAR10(cfg.root, train = (split == "train"), download=True, transform=transform)
    elif cfg.dataset == "CIFAR-100":
        dataset = datasets.CIFAR100(cfg.root, train = (split == "train"), download=True, transform=transform)
    elif cfg.dataset == "SVHN":
        dataset = datasets.SVHN(cfg.root, split=split, download=True, transform=transform)
    else:
        raise NotImplementedError
    return dataset