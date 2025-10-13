import torch
from math import floor
from scipy.stats import norm
from collections.abc import Callable


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
            _, crit, pert = method.split("-")
            assert crit in ["max", "sum"]
            assert pert in ["uv", "std", "sgn"]
            return min_max_sum(grads, m, crit=crit, pert_type=pert)
        case _ if method.startswith("ss"):
            _, crit = method.split("-")
            assert crit in ["max", "sum"]
            return scale_sign_attack(grads=grads, m=m, crit=crit)
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

def min_max_sum(
    grads: torch.Tensor, m: int, crit: str = "max", pert_type: str = "uv"
) -> torch.Tensor:
    """Implement min-max-sum attack strategy (refactored)."""
    honest_grads = grads[m:]
    avg = honest_grads.mean(dim=0)

    # Define the perturbation vector
    match pert_type:
        case "uv":
            pert = -avg / (avg.norm() + 1e-9) # Add epsilon for stability
        case "std":
            pert = -honest_grads.std(dim=0)
        case "sgn":
            pert = -avg.sign()

    # Define how to generate a malicious gradient from a gamma factor
    def mal_generator(gamma: float) -> torch.Tensor:
        return avg + gamma * pert

    # Find the optimal factor using the reusable helper function
    best_gamma = _find_optimal_factor(honest_grads, crit, mal_generator)
    print(f"Optimal γ for Min-Max/Sum = {best_gamma:.4f}")
    
    final_mal_grad = mal_generator(best_gamma)
    
    grads[:m] = final_mal_grad
    return grads

def scale_sign_attack(
    grads: torch.Tensor, m: int, crit: str = "max"
) -> torch.Tensor:
    """Implement ScaleSign attack strategy (refactored)."""
    honest_grads = grads[m:]
    avg = honest_grads.mean(dim=0)
    
    # Define how to generate a malicious gradient from a gamma factor
    def mal_generator(gamma: float) -> torch.Tensor:
        return gamma * avg
    # Find the optimal factor using the reusable helper function
    best_gamma = _find_optimal_factor(honest_grads, crit, mal_generator)
    print(f"Optimal γ for Scaling Attack = {best_gamma:.4f}")
    
    scaled_mal_grad = mal_generator(best_gamma)
    
    # Sign Modification step
    if len(honest_grads) > 0:
        ref_grad = honest_grads[0] # Using the first honest as reference
        final_mal_grad = _modify_signs(scaled_mal_grad, ref_grad)
    else:
        final_mal_grad = scaled_mal_grad
        
    grads[:m] = final_mal_grad
    return grads


def _find_optimal_factor(
    grads: torch.Tensor,
    crit: str,
    mal_generator: Callable[[float], torch.Tensor],
    max_iter: int = 50,
    eps: float = 1e-6,
) -> float:
    op = {"max": torch.max, "sum": torch.sum}[crit]
    
    max_grad_diff = op(torch.cdist(grads, grads)).item()
    
    left, right = 0.0, 1.0
    if op(torch.norm(grads - mal_generator(right), dim=1)).item() <= max_grad_diff:
        while True:
          mal = mal_generator(right)
          dist = op(torch.norm(grads - mal, dim=1)).item()
          if dist > max_grad_diff:
              break
          right *= 2
          if right > 1e6: # Safety break
              break
          
     # Binary search for the best factor
    for _ in range(max_iter):
        mid = (left + right) / 2
        mal = mal_generator(mid)
        dist = op(torch.norm(grads - mal, dim=1)).item()
        
        if dist <= max_grad_diff:
            left = mid
        else:
            right = mid

        if right - left < eps:
            break
    return left


def _modify_signs(mal_grad: torch.Tensor, ref_grad: torch.Tensor) -> torch.Tensor:
    """
    Adjusts the sign statistics of a malicious gradient to mimic a reference gradient.

    This function is the core component for bypassing sign-based defenses like SignGuard.
    It takes a potentially malicious gradient and modifies it by changing the signs of
    the smallest magnitude values, ensuring that the final count of positive, negative,
    and zero elements matches those of a benign reference gradient. This makes the
    malicious gradient statistically indistinguishable from an honest one based on sign counts.

    The modification strategy is designed to be minimal to preserve the attack's potency:
    - If there are too many positive/negative values, the ones with the smallest
      magnitudes are changed to zero.
    - If there are too few, non-positive/non-negative values with the smallest
      magnitudes are flipped to a small positive/negative epsilon.

    Args:
        mal_grad (torch.Tensor): The 1-D malicious gradient tensor to be modified.
        ref_grad (torch.Tensor): A 1-D benign gradient tensor used as a template
                                 for the target sign statistics.

    Returns:
        torch.Tensor: A new gradient tensor with the same shape as `mal_grad` but with
                      sign statistics matching `ref_grad`.
    """
    # Create a copy to avoid modifying the original malicious gradient in place.
    grad = mal_grad.clone()
    
    # --- Step 1: Calculate sign statistics for both gradients ---
    # p_mal/n_mal: Count of positive/negative values in the malicious gradient.
    # p_ref/n_ref: Target count of positive/negative values from the reference gradient.
    p_mal, n_mal = (grad > 0).sum(), (grad < 0).sum()
    p_ref, n_ref = (ref_grad > 0).sum(), (ref_grad < 0).sum()

    # --- Step 2: Adjust the number of positive values ---
    if p_mal > p_ref:
        # Case: Too many positive values.
        # We need to reduce the count by 'diff'.
        diff = p_mal - p_ref
        # Find all current positive values and their indices.
        pos_indices = torch.where(grad > 0)[0]
        # Sort these positive values to find the smallest ones.
        _, sorted_indices = torch.sort(grad[pos_indices])
        # Identify the indices of the 'diff' smallest positive values to be changed.
        indices_to_zero = pos_indices[sorted_indices[:diff]]
        # Change them to zero.
        grad[indices_to_zero] = 0.0
    elif p_mal < p_ref:
        # Case: Too few positive values.
        # We need to increase the count by 'diff'.
        diff = p_ref - p_mal
        # Find all non-positive values (candidates to be flipped).
        non_pos_indices = torch.where(grad <= 0)[0]
        # Sort them by absolute value to find those closest to zero.
        _, sorted_indices = torch.sort(grad[non_pos_indices].abs())
        # Identify the indices of the 'diff' non-positive values closest to zero.
        indices_to_pos = non_pos_indices[sorted_indices[:diff]]
        # Change them to a small positive value (epsilon).
        grad[indices_to_pos] = 1e-9

    # --- Step 3: Adjust the number of negative values ---
    # This logic mirrors the process for positive values.
    if n_mal > n_ref:
        # Case: Too many negative values.
        diff = n_mal - n_ref
        # Find all current negative values.
        neg_indices = torch.where(grad < 0)[0]
        # Sort them by absolute value to find those closest to zero.
        _, sorted_indices = torch.sort(grad[neg_indices].abs())
        # Identify the indices of the 'diff' "least negative" values.
        indices_to_zero = neg_indices[sorted_indices[:diff]]
        # Change them to zero.
        grad[indices_to_zero] = 0.0
    elif n_mal < n_ref:
        # Case: Too few negative values.
        diff = n_ref - n_mal
        # Find all non-negative values.
        non_neg_indices = torch.where(grad >= 0)[0]
        # Sort them by absolute value to find those closest to zero.
        _, sorted_indices = torch.sort(grad[non_neg_indices].abs())
        # Identify the indices of the 'diff' non-negative values closest to zero.
        indices_to_neg = non_neg_indices[sorted_indices[:diff]]
        # Change them to a small negative value (-epsilon).
        grad[indices_to_neg] = -1e-9
        
    return grad