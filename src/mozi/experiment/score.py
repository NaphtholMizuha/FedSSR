import torch
import torch.nn.functional as F
from loguru import logger
from typing import List

def _rescale_scores(scores: torch.Tensor) -> torch.Tensor:
    """
    Standardizes a tensor of scores using Z-score normalization.

    Args:
        scores (torch.Tensor): The input scores.

    Returns:
        torch.Tensor: The standardized scores.
    """
    # Ensure scores are float for mean and std calculation
    float_scores = scores.to(torch.float32)
    mean = torch.mean(float_scores)
    std = torch.std(float_scores)

    # Use a small epsilon to avoid division by zero
    if std < 1e-9:
        # If std is zero or negligible, all scores are the same.
        # The Z-score is 0 for all elements.
        return torch.zeros_like(scores, dtype=torch.float32)

    return (scores - mean) / std


def _cos_sim_mat(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Calculates the cosine similarity matrix between two sets of vectors.

    Args:
        X (torch.Tensor): The first set of vectors (m x d).
        Y (torch.Tensor): The second set of vectors (n x d).

    Returns:
        torch.Tensor: The cosine similarity matrix (m x n).
    """
    X_norm = F.normalize(X, dim=1)
    Y_norm = F.normalize(Y, dim=1)
    return X_norm.matmul(Y_norm.T)


def _dist_sim_mat(X: torch.Tensor, Y: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """
    Calculates similarity based on the ratio of Euclidean distance to the norm of the client_update.
    The ratio is then converted to a similarity score using an exponential function.

    Args:
        X (torch.Tensor): The server updates (m x d).
        Y (torch.Tensor): The client updates (n x d).
        gamma (float): The gamma parameter for the exponential conversion.

    Returns:
        torch.Tensor: The similarity matrix (m x n).
    """
    dist = torch.cdist(X, Y, p=2)
    norm_Y = torch.norm(Y, dim=1)

    # Use a small epsilon to avoid division by zero for zero vectors
    norm_Y = norm_Y + 1e-9

    ratio = dist / norm_Y

    return torch.exp(-gamma * ratio)


def _rand_sim_mat(X: torch.Tensor, Y: torch.Tensor, d_proj: int = 1000) -> torch.Tensor:
    """
    Calculates cosine similarity after random projection.

    Args:
        X (torch.Tensor): The first set of vectors (m x d).
        Y (torch.Tensor): The second set of vectors (n x d).
        d_proj (int): The dimension of the random projection space.

    Returns:
        torch.Tensor: The cosine similarity matrix (m x n).
    """
    d = X.shape[1]
    rand_proj = torch.randn(d, d_proj, device=X.device)
    X_proj = X @ rand_proj
    Y_proj = Y @ rand_proj
    return _cos_sim_mat(X_proj, Y_proj)




def _calc_chunk_scores(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Calculates cosine similarity on stds of chunks.

    Args:
        x (torch.Tensor): The first set of vectors (m x d).
        y (torch.Tensor): The second set of vectors (n x d).

    Returns:
        torch.Tensor: The chunk-based similarity matrix (m x n).
    """

    def _get_chunk_stats(t: torch.Tensor, n_chunks: int = 1000):
        """Computes std for chunks of a tensor."""
        chunks = torch.chunk(t, chunks=n_chunks, dim=1)
        stds_list = [torch.std(c, dim=1, unbiased=True) for c in chunks]
        # Stack them to get a (n, n_chunks) tensor for the stat
        stds = torch.stack(stds_list, dim=1)
        return stds

    local_stats = _get_chunk_stats(x)
    probe_stats = _get_chunk_stats(y)
    return _cos_sim_mat(local_stats, probe_stats)


def calculate_scores(
    client_updates: torch.Tensor,
    server_updates: torch.Tensor,
    score_types: list[str] | None = None,
) -> torch.Tensor:
    """
    Calculates a multi-dimensional score for each server.

    Args:
        client_updates (torch.Tensor): Updates from all clients.
        server_updates (torch.Tensor): Aggregated updates from all servers.
        score_types (list[str], optional): A list of score types to calculate. Defaults to all.

    Returns:
        torch.Tensor: A 2D tensor of final scores (m x k) for each server.
    """
    if score_types is None:
        score_types = ["cos", "sgn", "dist"]

    raw_scores_list = []
    calculated_score_names = []

    if "cos" in score_types:
        cos_scores = _cos_sim_mat(server_updates, client_updates)
        raw_scores_list.append(cos_scores)
        calculated_score_names.append("cos")

    if "sgn" in score_types:
        s_pos_ratio = (server_updates > 0).float().mean(dim=1)
        s_neg_ratio = (server_updates < 0).float().mean(dim=1)
        s_ratios = torch.stack([s_pos_ratio, s_neg_ratio], dim=1)

        c_pos_ratio = (client_updates > 0).float().mean(dim=1)
        c_neg_ratio = (client_updates < 0).float().mean(dim=1)
        c_ratios = torch.stack([c_pos_ratio, c_neg_ratio], dim=1)

        sgn_scores = _cos_sim_mat(s_ratios, c_ratios)
        raw_scores_list.append(sgn_scores)
        calculated_score_names.append("sgn")

    if "chunk" in score_types:
        chunk_scores = _calc_chunk_scores(server_updates, client_updates)
        raw_scores_list.append(chunk_scores)
        calculated_score_names.append("chunk")

    if "dist" in score_types:
        dist_scores = _dist_sim_mat(server_updates, client_updates)
        raw_scores_list.append(dist_scores)
        calculated_score_names.append("dist")

    if "rand" in score_types:
        rand_scores = _rand_sim_mat(server_updates, client_updates)
        raw_scores_list.append(rand_scores)
        calculated_score_names.append("rand")
        

    if not raw_scores_list:
        logger.warning(
            "No valid score types provided, returning zero scores for all servers."
        )
        return torch.zeros(server_updates.shape[0], len(score_types) or 1)

    # 1. First aggregation: Take the median per client to get server scores.
    server_scores_per_type = [s.median(dim=1).values for s in raw_scores_list]
    log_message = f"raw scores matrix (m x k) with types: {calculated_score_names}\n{server_scores_per_type}"
    logger.info(log_message)
    # 2. Standardization: Apply Z-score normalization to each score type's vector.
    standardized_server_scores = [_rescale_scores(s) for s in server_scores_per_type]

    # 3. Second aggregation: Stack and average across score types.
    final_scores_matrix = torch.stack(standardized_server_scores, dim=1)

    log_message = f"Final scores matrix (m x k) with types: {calculated_score_names}\n{final_scores_matrix}"
    logger.info(log_message)

    return final_scores_matrix.cpu()