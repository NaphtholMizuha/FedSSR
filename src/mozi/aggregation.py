import torch
from torch import Tensor
from loguru import logger
def aggregate(grads: Tensor, method: str, **kwargs) -> Tensor:
    """Aggregate multiple gradients
    
    Args:
        grads: Gradient tensor with shape (n_clients, *grad_shape)
        method: Name of the aggregation method
        **kwargs: Additional parameters depending on the specific aggregation method
    
    Returns:
        Aggregated gradient
    """
    match method:
        case 'fedavg':
            return fedavg(grads)
        case 'trimmed_mean':
            return trimmed_mean(grads, **kwargs)
        case 'krum':
            return krum(grads)
        case 'median':
            return median(grads)
        case 'collude':
            return collude(grads, **kwargs)
        case 'random':
            return random(grads)
        case _:
            raise ValueError(f"Aggregation Method Not Found: {method}")

def fedavg(grads: Tensor) -> Tensor:
    """Implement Federated Averaging (FedAvg) aggregation
    
    Args:
        grads: Gradient tensor with shape (n_clients, *grad_shape)
    
    Returns:
        Aggregated gradient
    """
    return grads.mean(dim=0)

def trimmed_mean(grads: Tensor, prop: float = 0.8) -> Tensor:
    """Implement trimmed mean aggregation
    
    Args:
        grads: Gradient tensor with shape (n_clients, *grad_shape)
        prop: Proportion of gradients to keep, default is 0.8
    
    Returns:
        Aggregated gradient
    """
    k = int(grads.shape[0] * (1 - prop) / 2)  # Trim k values from both ends
    sorted_grads, _ = torch.sort(grads, dim=0)
    trimmed_grads = sorted_grads[k:-k] if k > 0 else sorted_grads
    return trimmed_grads.mean(dim=0)
    
def krum(grads: Tensor) -> Tensor:
    """Implement Krum aggregation
    
    Args:
        grads: Gradient tensor with shape (n_clients, *grad_shape)
    
    Returns:
        Aggregated gradient, selecting the one closest to others
    """
    dist_mat = torch.cdist(grads, grads)  # Calculate Euclidean distances between gradients
    dist_vec = torch.sum(dist_mat, dim=1)  # Sum of distances from each gradient to all others
    target = torch.argmin(dist_vec)  # Select the gradient with minimum distance sum
    return grads[target].squeeze()

def median(grads: Tensor) -> Tensor:
    """Implement median aggregation
    
    Args:
        grads: Gradient tensor with shape (n_clients, *grad_shape)
    
    Returns:
        Aggregated gradient, taking median in each dimension
    """
    return grads.median(dim=0).values

def collude(grads: Tensor, m: int) -> Tensor:
    """Implement collusion aggregation (randomly select from malicious clients)
    
    Args:
        grads: Gradient tensor with shape (n_clients, *grad_shape)
        m: Number of malicious clients
    
    Returns:
        Randomly selected gradient from malicious clients
    """
    idx = torch.randint(0, m, (1,))
    logger.warning(f'Malicious server apply the update from client {idx}')
    return grads[idx].squeeze()

def random(grads: Tensor) -> Tensor:
    """Generate random gradient
    
    Args:
        grads: Gradient tensor with shape (n_clients, *grad_shape), used for shape information
    
    Returns:
        Random gradient with the same shape as input
    """
    return torch.rand_like(grads[0])
