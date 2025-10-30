import torch
from torch import Tensor
from sklearn.mixture import GaussianMixture
import numpy as np
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
        case 'weighted':
            return weighted_fedavg(grads, **kwargs)
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
        case "gmm":
            return gmm_md(grads)
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

def weighted_fedavg(grads: Tensor, weights: Tensor) -> Tensor:
    if torch.all(weights == 0):
        weights = torch.ones_like(weights)
    weights = weights / (weights.sum() + 1e-8)
    return (grads * weights.unsqueeze(1)).sum(dim=0)

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

def gmm_md(
    grads: torch.Tensor,
    n_components: int = 2,
    threshold_factor: float = 2.0,
    verbose: bool = True,
):
    num_users, grad_dim = grads.shape
    
    if num_users < n_components:
        if verbose:
            print(f"Warning: Number of users ({num_users}) is less than GMM components ({n_components}). Aggregating all gradients.")
        return torch.mean(grads, dim=0)

    grads_np = grads.cpu().numpy()
    
    gmm = GaussianMixture(n_components=n_components, covariance_type='diag', random_state=42)
    gmm.fit(grads_np)
    
    benign_idx = np.argmax(gmm.weights_)
    if verbose:
        print(f"Benign cluster identified as index: {benign_idx} (Weight: {gmm.weights_[benign_idx]:.2f})")
        
    labels = gmm.predict(grads_np)
    mahalanobis_dists = np.zeros(num_users)
    
    for k in range(n_components):
        indices = np.where(labels == k)[0]
        if len(indices) == 0:
            continue
        
        diff = grads_np[indices,:] - gmm.means_[k]
        sq_dists = np.sum(diff**2 * gmm.precisions_[k], axis=1)
        mahalanobis_dists[indices] = np.sqrt(sq_dists)
        
    dists_in_benign_cluster = mahalanobis_dists[labels == benign_idx]
    
    if len(dists_in_benign_cluster) == 0:
        if verbose:
            print("Warning: Benign cluster is empty. Cannot determine threshold. Returning zero vector.")
        return torch.zeros(grad_dim, device=grads.device, dtype=grads.dtype)
    
    mean_md = np.mean(dists_in_benign_cluster)
    std_md = np.std(dists_in_benign_cluster)
    threshold = mean_md + threshold_factor * std_md
    if verbose:
        print(f"Calculated dynamic threshold T = {threshold:.4f} (Mean MD: {mean_md:.4f}, Std MD: {std_md:.4f})")

    # --- 5. 确定良性梯度的索引 ---
    benign_mask = (labels == benign_idx) & (mahalanobis_dists < threshold)
    benign_indices = np.where(benign_mask)[0]
    
    num_benign = len(benign_indices)
    
    if verbose:
        print(f"Audit complete. Found {num_benign} benign gradients, filtered out {num_users - num_benign}.")

    # --- 6. 聚合良性梯度 ---
    if num_benign > 0:
        benign_gradients = grads[benign_indices]
        aggregated_gradient = torch.mean(benign_gradients, dim=0)
    else:
        if verbose:
            print("Warning: All gradients were filtered out. Returning a zero vector.")
        aggregated_gradient = torch.zeros(grad_dim, device=grads.device, dtype=grads.dtype)

    return aggregated_gradient