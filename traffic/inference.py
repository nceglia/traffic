"""Conjugate Gamma-Poisson CAVI -- the inference core.

A direct transcription of the Inference section of model_methods.tex, with one
efficiency: because the model is conjugate, the variational *rate* is constant
(computed once) and only the *shape* iterates.  Derivation -- the allocation /
entry updates collapse to matmuls:

    rate_{zz'}  = b + sum_j d_{j,z'} xtilde_j(z)            = b + Xt^T D       (constant)
    rho_{zz'}   = exp( psi(shape_{zz'}) - log rate_{zz'} )  = exp E_q[log M]
    Z_{j z'}    = sum_z xtilde_j(z) rho_{zz'}               = Xt @ rho         [J,L]
    shape_{zz'} = a + rho_{zz'} * sum_j xtilde_j(z) y_j(z')/Z_{j z'}
                = a + rho * ( Xt^T (Y / (Xt @ rho)) )

No [J,L,L] allocation tensor is ever formed.  M-hat = shape / rate.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.scipy.special import digamma

from . import model
from .config import CAVIConfig, PriorConfig


@dataclass
class Posterior:
    shape: jnp.ndarray   # [L, L] variational Gamma shape
    rate: jnp.ndarray    # [L, L] variational Gamma rate

    @property
    def mean(self):
        """M-hat = E_q[M] = shape / rate."""
        return self.shape / self.rate

    @property
    def var(self):
        return self.shape / self.rate ** 2


@dataclass
class FitResult:
    posterior: Posterior
    M_hat: jnp.ndarray
    delta_trace: jnp.ndarray   # max relative change in M-hat per iteration
    loglik_trace: jnp.ndarray  # Poisson log-likelihood at M-hat per iteration
    n_iter: int
    converged: bool


def fit(Xtilde, Y, D, prior: PriorConfig = PriorConfig(), cfg: CAVIConfig = CAVIConfig()) -> FitResult:
    """Fit M by conjugate CAVI.

    Xtilde, Y, D : [J, L] jax arrays (Y, D zeroed at missing states).
    Returns a FitResult with M_hat and convergence traces.
    """
    Xtilde = jnp.asarray(Xtilde, jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32)
    Y = jnp.asarray(Y, Xtilde.dtype)
    D = jnp.asarray(D, Xtilde.dtype)
    L = Xtilde.shape[1]
    a = jnp.full((L, L), prior.a0, Xtilde.dtype)
    b = jnp.full((L, L), prior.b0, Xtilde.dtype)
    eps = cfg.eps

    rate = b + Xtilde.T @ D                       # constant [L, L]

    def body(carry, _):
        shape, M_prev = carry
        rho = jnp.exp(digamma(shape) - jnp.log(rate + eps))   # [L, L]
        Z = Xtilde @ rho + eps                                # [J, L]
        W = Y / Z                                             # [J, L]
        new_shape = a + rho * (Xtilde.T @ W)                  # [L, L]
        M_new = new_shape / rate
        delta = jnp.max(jnp.abs(M_new - M_prev) / (M_new + eps))
        loglik = model.poisson_loglik(Y, D, Xtilde, M_new, eps)
        return (new_shape, M_new), (delta, loglik)

    shape0 = a
    M0 = shape0 / rate
    (shape_f, M_f), (deltas, logliks) = jax.lax.scan(
        body, (shape0, M0), None, length=cfg.max_iter
    )

    deltas = jax.device_get(deltas)
    # first iteration index where delta < tol (1-based count of iters used)
    import numpy as np
    below = np.where(deltas < cfg.tol)[0]
    n_iter = int(below[0] + 1) if below.size else int(cfg.max_iter)
    converged = bool(below.size)

    post = Posterior(shape=shape_f, rate=rate)
    return FitResult(
        posterior=post,
        M_hat=post.mean,
        delta_trace=deltas,
        loglik_trace=jax.device_get(logliks),
        n_iter=n_iter,
        converged=converged,
    )
