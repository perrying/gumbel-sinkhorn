import numpy as np
from scipy.stats import kendalltau

def kendall_tau(perm1: np.ndarray, perm2: np.ndarray) -> (np.ndarray,):
    assert perm1.ndim == 2 and perm2.ndim == 2
    return np.stack([kendalltau(p1, p2) for p1, p2 in zip(perm1, perm2)])