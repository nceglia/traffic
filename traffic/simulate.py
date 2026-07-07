"""Forward-sample synthetic data from the FACTORED generative model (recovery tests)."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import FactoredPriorConfig
from .mcmc import alpha_pi_matrix, assemble   # one shared reshape/assembly convention


def sample_counts(key, M, Xtilde, D, mask, phi=None):
    """Draw destination counts, zeroed at missing states.

    phi=None -> Poisson(d * (Xtilde M)).
    phi given -> NegBinomial2(mean = d * (Xtilde M), concentration phi), via the Gamma-Poisson
    mixture y ~ Poisson(mean * lambda), lambda ~ Gamma(phi, phi)  (mean 1, var 1/phi), which gives
    variance mean + mean^2/phi -- the model's NB2. phi may be a scalar or a length-L array.
    """
    rate = D * (Xtilde @ M)
    if phi is None:
        Y = jax.random.poisson(key, rate)
    else:
        k1, k2 = jax.random.split(key)
        rr = jnp.broadcast_to(jnp.asarray(phi, rate.dtype), rate.shape)
        lam = jax.random.gamma(k1, rr) / rr                 # mean 1, var 1/phi
        Y = jax.random.poisson(k2, rate * lam)
    return Y.astype(Xtilde.dtype) * mask


def sample_M_factored(key, mu_g, sigma_g, alpha_pi, beta, L, S=3):
    """Draw the operator factors from their priors and assemble M (SAME convention as the model).

    Returns (M [L,L], {"g": [L], "pi": [L,S], "Phi": [L,S,K]}).
    """
    K = L // S
    k1, k2, k3 = jax.random.split(key, 3)
    g = jnp.exp(mu_g + sigma_g * jax.random.normal(k1, (L,)))             # [L]  LogNormal
    pi = jax.random.dirichlet(k2, jnp.asarray(alpha_pi))                  # [L,S]
    Phi = jax.random.dirichlet(k3, beta * jnp.ones(K), shape=(L, S))      # [L,S,K]
    M = assemble(g, pi, Phi)
    return M, {"g": g, "pi": pi, "Phi": Phi}


def make_synthetic_factored(key, prior: FactoredPriorConfig = FactoredPriorConfig(),
                            L=39, J=2000, S=3, depth_lo=50, depth_hi=4000,
                            p_missing=0.4, sparsity=0.3, phi=None):
    """Synthetic dataset from the factored model, mirroring the real-data geometry.

    Returns (Xtilde, Y, D, mask, M_true, factors_true). xtilde rows are sparse depth-normalized
    source compositions; per-observation tissue depths and a missingness mask mirror the real
    coverage. Pass a scalar `phi` for over-dispersed NB2 counts, else Poisson.
    """
    K = L // S
    k0, k2, k3, k4, k5 = jax.random.split(key, 5)
    alpha_pi = alpha_pi_matrix(prior, L, S)
    M_true, factors = sample_M_factored(k0, prior.mu_g, prior.sigma_g, alpha_pi, prior.beta, L, S)

    raw = jax.random.exponential(k2, (J, L))
    keep = (jax.random.uniform(k3, (J, L)) < sparsity).astype(raw.dtype)
    src = raw * keep
    Xtilde = src / (src.sum(1, keepdims=True) + 1e-9)

    tissue_of = jnp.repeat(jnp.arange(S), K)
    d_tissue = jax.random.randint(k4, (J, S), depth_lo, depth_hi).astype(raw.dtype)
    m_tissue = (jax.random.uniform(k5, (J, S)) > p_missing).astype(raw.dtype)
    D = d_tissue[:, tissue_of] * m_tissue[:, tissue_of]
    mask = (D > 0).astype(raw.dtype)

    Y = sample_counts(key, M_true, Xtilde, D, mask, phi=phi)
    return Xtilde, Y, D, mask, M_true, factors
