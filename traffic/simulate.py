"""Forward-sample synthetic data from the generative model (recovery tests)."""
from __future__ import annotations

import jax
import jax.numpy as jnp


def sample_M(key, a, b):
    """Draw M_{zz'} ~ Gamma(shape=a, rate=b), shape [L, L]."""
    return jax.random.gamma(key, a) / b


def sample_counts(key, M, Xtilde, D, mask, r=None):
    """Draw destination counts, zeroed at missing states.

    r=None  -> Poisson(d * (Xtilde M)).
    r given -> NegBinomial2(mean = d * (Xtilde M), concentration r), via the
    Gamma-Poisson mixture y ~ Poisson(mean * lambda), lambda ~ Gamma(r, r).
    r may be a scalar or a length-L array (e.g. per-tissue, broadcast over rows).
    """
    rate = D * (Xtilde @ M)
    if r is None:
        Y = jax.random.poisson(key, rate)
    else:
        k1, k2 = jax.random.split(key)
        rr = jnp.broadcast_to(jnp.asarray(r, rate.dtype), rate.shape)
        lam = jax.random.gamma(k1, rr) / rr                 # mean 1, var 1/r
        Y = jax.random.poisson(k2, rate * lam)
    return Y.astype(Xtilde.dtype) * mask


def make_synthetic(key, L=33, J=2000, S=3, depth_lo=50, depth_hi=4000,
                   p_missing=0.4, sparsity=0.3, a_true=0.6, b_true=0.6, r=None):
    """Build a synthetic dataset that mimics the real-data geometry.

    Returns (Xtilde, Y, D, mask, M_true). xtilde rows are sparse depth-normalized
    source compositions; depths and a per-observation tissue missingness mask
    mirror the 52/108 sample coverage of the real set. Pass r (scalar or per-tissue
    [S]) to draw over-dispersed Negative-Binomial counts instead of Poisson.
    """
    K = L // S
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    M_true = sample_M(k1, jnp.full((L, L), a_true), jnp.full((L, L), b_true))

    raw = jax.random.exponential(k2, (J, L))
    keep = (jax.random.uniform(k3, (J, L)) < sparsity).astype(raw.dtype)
    src = raw * keep
    src = src / (src.sum(1, keepdims=True) + 1e-9)
    Xtilde = src

    tissue_of = jnp.repeat(jnp.arange(S), K)
    d_tissue = jax.random.randint(k4, (J, S), depth_lo, depth_hi).astype(raw.dtype)
    m_tissue = (jax.random.uniform(k5, (J, S)) > p_missing).astype(raw.dtype)
    D = d_tissue[:, tissue_of] * m_tissue[:, tissue_of]
    mask = (D > 0).astype(raw.dtype)

    if r is not None and jnp.ndim(r) > 0:        # per-tissue [S] -> per-column [L]
        r = jnp.asarray(r)[tissue_of]
    Y = sample_counts(key, M_true, Xtilde, D, mask, r=r)
    return Xtilde, Y, D, mask, M_true
