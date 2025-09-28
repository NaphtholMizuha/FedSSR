import torch
from .btbcn import BinaryClusterTree

def cos_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.matmul(y, x) / (torch.norm(x) * torch.norm(y, dim=1))

def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, min=0)

def normalize(x: torch.Tensor) -> torch.Tensor:
    return x / torch.sum(x)

def z_score_outliers(x: torch.Tensor, thr=3):
    z_scores = (x - x.mean()) / x.std()
    return torch.abs(z_scores) > thr

def krum(x: torch.Tensor):
    diff = x.unsqueeze(1) - x.unsqueeze(0)
    squared_diff = diff ** 2
    sum_squared_diff = torch.sum(squared_diff, dim=2)
    dist_mat = torch.sqrt(sum_squared_diff)
    dist_vec = torch.sum(dist_mat, dim=1)
    target = torch.argmin(dist_vec)
    return x[target]

def trimmed_mean(x: torch.Tensor, k):
    sorted_grads, _ = torch.sort(x, dim=0)
    trimmed_grads = sorted_grads[k : -k, :]
    column_means = torch.mean(trimmed_grads, dim=0)
    return column_means

def feddmc(x: torch.Tensor, k, min_clu_size=3):
    U, S, _ = torch.pca_lowrank(x, q=k)
    x_proj = U.matmul(torch.diag(S)).to('cpu')
    bct = BinaryClusterTree(min_clu_size)
    bct.fit(x_proj)
    benign, _, _ = bct.classify()
    
    return torch.mean(x[benign], dim=0)
    