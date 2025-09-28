import torch
from math import floor
from scipy.stats import norm
from random import random

def attack(grads: torch.Tensor, method: str, m: int, n: int) -> torch.Tensor:
    """Perform gradient attack
    
    Args:
        grads: Gradient tensor
        method: Attack method
        m: Number of malicious clients
        n: Total number of clients
        state: State for fedghost attack
        **kwargs: Other parameters
    
    Returns:
        Modified gradients and updated state (if using fedghost)
    """
    match method:
        case "none":
            return grads
        case "ascent":
            grads[:m] = -grads[:m]
            return grads
        case "lie":
            return little_is_enough(grads, m, n)
        case _ if method.startswith("min"):
            _, crit, pert = method.split('-')
            assert crit in ['max', 'sum']
            assert pert in ['uv', 'std', 'sgn']
            return min_max_sum(grads, m, crit=crit, pert_type=pert)
        case _:
            raise ValueError(f"Attack Method Not Found: {method}")

def little_is_enough(grads: torch.Tensor, m: int, n: int) -> torch.Tensor:
    """Implement 'little is enough' attack strategy
    
    Args:
        grads: Gradient tensor
        m: Number of malicious clients
        n: Total number of clients
    
    Returns:
        Modified gradients
    """
    n_supp = floor(n / 2 + 1) - m
    phi_z = (n - m - n_supp) / (n - m)
    z = norm.ppf(phi_z)
    sigma, mu = torch.std_mean(grads, dim=0)
    grads[:m] = mu - z * sigma
    return grads

def min_max_sum(grads: torch.Tensor, m: int, crit: str = 'max', pert_type: str = 'uv') -> torch.Tensor:
    """Implement min-max-sum attack strategy
    
    Args:
        grads: Gradient tensor
        m: Number of malicious clients
        crit: Optimization criterion, `max` or `sum`
        pert_type: Perturbation type, `uv`, `std` or `sgn`
    
    Returns:
        Modified gradients
    """
    max_iter, eps = 50, 1e-6
    avg = grads.mean(dim=0)
    
    match pert_type:
        case "uv":
            pert = - avg / avg.norm()
        case "std":
            pert = -grads.std(dim=0) 
        case "sgn":
            pert = -avg.sign()
            
    op = {"max": torch.max, "sum": torch.sum}[crit]
    max_grad_diff = op(torch.cdist(grads, grads)).item()

    left, right = 0.0, 1.0
    
    while True:
        mal = avg + right * pert
        dist = op(torch.norm(grads - mal, dim=1)).item()
        if dist > max_grad_diff:
            break
        right *= 2

    for _ in range(max_iter):
        mid = (left + right) / 2
        mal = avg + mid * pert
        dist = op(torch.norm(grads - mal, dim=1)).item()
        if dist <= max_grad_diff:
            left = mid
        else:
            right = mid

        if right - left < eps:
            break
            
    print(f"γ = {left}")
    grads[:m] = avg + left * pert
    return grads
