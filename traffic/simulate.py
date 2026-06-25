"""Forward-sample synthetic data from the generative model (recovery tests)."""
from __future__ import annotations

import jax
import jax.numpy as jnp


def sample_M(key, a, b):
    """Draw M_{zz'} ~ Gamma(shape=a, rate=b), shape [L, L]."""
    return jax.random.gamma(key, a) / b


def sample_counts(key, M, Xtilde, D, mask):
    """Draw y_j(z') ~ Poisson(d_{j,z'} * (Xtilde M)(z')), zeroed at missing states."""
    rate = D * (Xtilde @ M)
    Y = jax.random.poisson(key, rate).astype(Xtilde.dtype)
    return Y * mask


def make_synthetic(key, L=33, J=2000, S=3, depth_lo=50, depth_hi=4000,
                   p_missing=0.4, sparsity=0.3, a_true=0.6, b_true=0.6):
    """Build a synthetic dataset that mimics the real-data geometry.

    Returns (Xtilde, Y, D, mask, M_true). xtilde rows are sparse depth-normalized
    source compositions; depths and a per-observation tissue missingness mask
    mirror the 52/108 sample coverage of the real set.
    """
    K = L // S
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    # True growth matrix: mostly within-tissue/phenotype-local mass + some traffic.
    M_true = sample_M(k1, jnp.full((L, L), a_true), jnp.full((L, L), b_true))

    # Sparse source compositions (each clone occupies a few states), depth-normalized.
    raw = jax.random.exponential(k2, (J, L))
    keep = (jax.random.uniform(k3, (J, L)) < sparsity).astype(raw.dtype)
    src = raw * keep
    src = src / (src.sum(1, keepdims=True) + 1e-9)          # source composition (sums ~1)
    # Source depth per tissue -> depth-rescaled source xtilde = src.
    Xtilde = src

    # Destination depth per (obs, tissue), broadcast to states; plus tissue missingness.
    tissue_of = jnp.repeat(jnp.arange(S), K)
    d_tissue = jax.random.randint(k4, (J, S), depth_lo, depth_hi).astype(raw.dtype)
    m_tissue = (jax.random.uniform(k5, (J, S)) > p_missing).astype(raw.dtype)
    D = d_tissue[:, tissue_of] * m_tissue[:, tissue_of]      # [J, L], 0 where missing
    mask = (D > 0).astype(raw.dtype)

    Y = sample_counts(key, M_true, Xtilde, D, mask)
    return Xtilde, Y, D, mask, M_true
