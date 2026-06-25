"""Gamma-Poisson population-dynamics model -- the spec (pure JAX functions).

Generative model (see gbm_trafficking/docs/model_methods.tex):

    M_{zz'}   ~ Gamma(a_z, b_z)                  non-negative growth matrix, free row sums
    mu_j      = xtilde_j @ M                      destination intensity vector
    y_j(z')   ~ Poisson(d_{j,z'} * mu_j(z'))      over non-missing states

Missingness is folded into the arrays: at a missing destination state the
data builder sets both Y[j,z'] = 0 and D[j,z'] = 0, so the term drops out of
the likelihood (rate 0, count 0).
"""
from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.special import gammaln


def intensity(xtilde, M):
    """mu_j = xtilde_j @ M, shape [J, L]."""
    return xtilde @ M


def expected_counts(xtilde, M, D):
    """E[y_j(z')] = d_{j,z'} * mu_j(z'), shape [J, L]."""
    return D * (xtilde @ M)


def poisson_loglik(Y, D, xtilde, M, eps=1e-12):
    """Masked Poisson log-likelihood (scalar).

    Y, D are zeroed at missing states, so masked terms contribute 0.
    """
    rate = D * (xtilde @ M)
    ll = Y * jnp.log(rate + eps) - rate - gammaln(Y + 1.0)
    return jnp.sum(jnp.where(D > 0, ll, 0.0))
