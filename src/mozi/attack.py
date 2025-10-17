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
    grads: torch.Tensor, 
    m: int, 
    crit: str = "max",
    k: int | None = 1000
) -> torch.Tensor:
    """
    实现 ScaleSign 攻击策略。
    现在支持均匀缩放和坐标级缩放。

    Args:
        grads (torch.Tensor): 所有参与方的梯度张量。
        m (int): 恶意参与方的数量。
        crit (str): 判定标准，'max' 或 'sum'。
        k (Optional[int]): 要进行坐标级缩放的坐标数量。
                           如果为 None 或 0，则执行均匀缩放 (旧行为)。
                           如果为正数，则选择绝对值最大的k个坐标进行缩放。

    Returns:
        torch.Tensor: 包含恶意梯度的新梯度张量。
    """
    honest_grads = grads[m:]
    avg = honest_grads.mean(dim=0)
    d = avg.numel() # 获取梯度维度
    
    # --- 根据 k 的值决定缩放策略 ---
    if k is not None and k > 0:
        # --- 坐标级缩放 (Coordinate-wise Scaling) ---
        print(f"执行坐标级缩放，k = {k}")
        
        # 1. 确定坐标集 Θ₁ 和 Θ₂
        # 策略：选择平均梯度中绝对值最大的 k 个坐标
        abs_avg = avg.abs()
        _, top_k_indices = torch.topk(abs_avg, min(k, d)) # 防止 k > d
        
        # 将前一半作为 Θ₁ (放大)，后一半作为 Θ₂ (缩小)
        split_idx = len(top_k_indices) // 2
        theta1 = top_k_indices[:split_idx]
        theta2 = top_k_indices[split_idx:]
        
        # 2. 定义坐标级缩放的恶意梯度生成器
        def mal_generator(gamma: float) -> torch.Tensor:
            scaling_vector = _generate_scaling_vector(d, theta1, theta2, gamma, avg.device, avg.dtype)
            return scaling_vector * avg # 元素级乘法
    else:
        # --- 均匀缩放 (Uniform Scaling)，保持旧行为 ---
        print("执行均匀缩放")
        def mal_generator(gamma: float) -> torch.Tensor:
            return gamma * avg

    # 找到最优的缩放因子 gamma
    best_gamma = _find_optimal_factor(honest_grads, crit, mal_generator)
    print(f"Optimal γ for Scaling Attack = {best_gamma:.4f}")
    
    # 生成最优的恶意梯度
    scaled_mal_grad = mal_generator(best_gamma)
    
    # 符号修改步骤
    if len(honest_grads) > 0:
        ref_grad = honest_grads[0] # 使用第一个良性梯度作为参考
        final_mal_grad = _modify_signs(scaled_mal_grad, ref_grad)
    else:
        final_mal_grad = scaled_mal_grad
        
    grads[:m] = final_mal_grad
    return grads

def _generate_scaling_vector(
    d: int, 
    theta1: torch.Tensor, 
    theta2: torch.Tensor, 
    gamma: float,
    device: torch.device,
    dtype: torch.dtype
) -> torch.Tensor:
    """
    根据算法1生成一个缩放向量 (Scaling Vector)。
    这个向量等效于对角矩阵 Φ 的对角线。

    Args:
        d (int): 梯度向量的维度。
        theta1 (torch.Tensor): 坐标集 Θ₁, 将被乘以 gamma。
        theta2 (torch.Tensor): 坐标集 Θ₂, 将被乘以 1/gamma。
        gamma (float): 缩放系数。
        device: a torch.device object.
        dtype: a torch.dtype object.
        
    Returns:
        torch.Tensor: 一个1-D的缩放向量。
    """
    # 1. 初始化一个全为1的向量，相当于单位矩阵的对角线。
    scaling_vector = torch.ones(d, device=device, dtype=dtype)
    
    # 避免 gamma 为 0 导致除零错误
    if gamma == 0.0:
        gamma = 1e-9 # 用一个极小值代替

    # 2. 对 Θ₁ 中的坐标应用缩放因子 γ
    scaling_vector[theta1] = gamma
    
    # 3. 对 Θ₂ 中的坐标应用缩放因子 1/γ
    scaling_vector[theta2] = 1.0 / gamma
    
    return scaling_vector

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