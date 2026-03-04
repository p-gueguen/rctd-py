"""Simplex projection using Duchi et al. (2008) algorithm.

Projects a vector onto the probability simplex {x | x >= 0, sum(x) = 1}.
O(K log K) where K is the vector dimension.
"""

import torch


def project_simplex(v: torch.Tensor) -> torch.Tensor:
    """Project vector v onto the probability simplex.

    Implements the algorithm from:
    Duchi, Shalev-Shwartz, Singer, Chandra (2008)
    "Efficient Projections onto the l1-Ball for Learning in High Dimensions"

    Args:
        v: Input vector of shape (K,)

    Returns:
        Projected vector of shape (K,) on the simplex.
    """
    n = v.shape[0]
    u = torch.sort(v, descending=True).values
    cssv = torch.cumsum(u, dim=0)
    ind = torch.arange(1, n + 1, dtype=v.dtype, device=v.device)
    cond = u * ind > (cssv - 1.0)
    rho = torch.max(torch.where(cond, ind, torch.zeros_like(ind)))
    theta = (cssv[int(rho.item()) - 1] - 1.0) / rho
    return torch.clamp(v - theta, min=0.0)


def project_simplex_batch(v: torch.Tensor) -> torch.Tensor:
    """Project each row of v onto the probability simplex.

    Batched version of project_simplex for (N, K) input.

    Args:
        v: Input matrix of shape (N, K)

    Returns:
        Projected matrix of shape (N, K) with each row on the simplex.
    """
    u = torch.sort(v, descending=True, dim=1).values
    cssv = torch.cumsum(u, dim=1)
    K = v.shape[1]
    ind = torch.arange(1, K + 1, dtype=v.dtype, device=v.device).unsqueeze(0)  # (1, K)
    cond = u * ind > (cssv - 1.0)
    # rho = last index where condition holds (per row)
    rho = (cond * ind).max(dim=1).values  # (N,)
    rho_idx = rho.long() - 1  # 0-indexed
    theta = (cssv[torch.arange(v.shape[0], device=v.device), rho_idx] - 1.0) / rho  # (N,)
    return torch.clamp(v - theta.unsqueeze(1), min=0.0)
