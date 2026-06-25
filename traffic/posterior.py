"""Posterior uncertainty + identifiability from the variational Gamma factors."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def sample_M(shape, rate, key, n=2000):
    """Draw n samples of M ~ Gamma(shape, rate), returns np array [n, L, L]."""
    shape = jnp.asarray(shape)
    draws = jax.random.gamma(key, jnp.broadcast_to(shape, (n,) + shape.shape)) / jnp.asarray(rate)
    return np.asarray(draws)


def credible_interval(samples_of_scalar, level=0.95):
    """mean + (lo, hi) percentile interval for a 1-D array of posterior draws."""
    x = np.asarray(samples_of_scalar)
    lo = (1 - level) / 2 * 100
    return float(x.mean()), float(np.percentile(x, lo)), float(np.percentile(x, 100 - lo))


def identifiability(shape, rate, a0, b0):
    """Per-entry data information vs prior.

    effective_count = shape - a0  (the accumulated sum_j y r; 0 => prior-only).
    prior_fraction  = a0 / shape  (->1 means prior-dominated / data-thin).
    Returns dict of [L,L] arrays.
    """
    shape = np.asarray(shape)
    eff = shape - a0
    return {
        "effective_count": eff,
        "prior_fraction": a0 / np.maximum(shape, 1e-12),
        "cv": 1.0 / np.sqrt(np.maximum(shape, 1e-12)),   # posterior coefficient of variation
    }
