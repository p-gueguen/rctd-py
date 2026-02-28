"""Simplex projection using Duchi et al. (2008) algorithm.

Projects a vector onto the probability simplex {x | x >= 0, sum(x) = 1}.
O(K log K) where K is the vector dimension.
"""

import jax.numpy as jnp


def project_simplex(v: jnp.ndarray) -> jnp.ndarray:
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
    u = jnp.sort(v)[::-1]
    cssv = jnp.cumsum(u)
    ind = jnp.arange(1, n + 1, dtype=v.dtype)
    cond = u * ind > (cssv - 1.0)
    rho = jnp.max(jnp.where(cond, ind, 0))
    theta = (cssv[jnp.int32(rho) - 1] - 1.0) / rho
    return jnp.maximum(v - theta, 0.0)
