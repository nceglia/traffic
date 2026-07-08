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


def sample_M_factored(key, mu_g, sigma_g, alpha_pi, beta, L, S=3, factors=None):
    """Draw the operator factors from their priors and assemble M (SAME convention as the model).

    If `factors` is given (a dict with any of "g","pi","Phi"), those entries are used VERBATIM in
    place of a prior draw and only the remaining factors are sampled -- the hook the recovery /
    boundary experiments use to plant a known truth (e.g. exact pi zeros via
    traffic.recovery.plant_pi_zeros). Defaults (factors=None) reproduce the pure-prior draw exactly.

    Returns (M [L,L], {"g": [L], "pi": [L,S], "Phi": [L,S,K]}).
    """
    K = L // S
    k1, k2, k3 = jax.random.split(key, 3)
    f = factors or {}
    g = jnp.asarray(f["g"]) if "g" in f else jnp.exp(mu_g + sigma_g * jax.random.normal(k1, (L,)))    # [L]  LogNormal
    pi = jnp.asarray(f["pi"]) if "pi" in f else jax.random.dirichlet(k2, jnp.asarray(alpha_pi))       # [L,S]
    Phi = jnp.asarray(f["Phi"]) if "Phi" in f else jax.random.dirichlet(k3, beta * jnp.ones(K), shape=(L, S))  # [L,S,K]
    M = assemble(g, pi, Phi)
    return M, {"g": g, "pi": pi, "Phi": Phi}


def prior_sample(key, prior: FactoredPriorConfig, L, S=3):
    """Draw a FULL parameter vector theta = {g, pi, Phi, log_r} from the model prior -- exactly the
    sites in mcmc.numpyro_model_factored, INCLUDING the global NB2 log-concentration
    log_r ~ Normal(0, sigma_phi) that sample_M_factored omits.

    This is the shared truth-generator for both recovery (Goal 1) and SBC (Goal 2). A valid SBC
    round MUST draw log_r here rather than fixing phi -- ranks are only uniform under draws from the
    FULL prior. Returns {"g":[L], "pi":[L,S], "Phi":[L,S,K], "log_r": scalar, "M":[L,L]}.
    """
    kf, kr = jax.random.split(key)
    alpha_pi = alpha_pi_matrix(prior, L, S)
    M, factors = sample_M_factored(kf, prior.mu_g, prior.sigma_g, alpha_pi, prior.beta, L, S)
    log_r = prior.sigma_phi * jax.random.normal(kr)                       # Normal(0, sigma_phi)
    return {"g": factors["g"], "pi": factors["pi"], "Phi": factors["Phi"], "log_r": log_r, "M": M}


def make_synthetic_factored(key, prior: FactoredPriorConfig = FactoredPriorConfig(),
                            L=39, J=2000, S=3, depth_lo=50, depth_hi=4000,
                            p_missing=0.4, sparsity=0.3, phi=None, factors=None):
    """Synthetic dataset from the factored model, mirroring the real-data geometry.

    Returns (Xtilde, Y, D, mask, M_true, factors_true). xtilde rows are sparse depth-normalized
    source compositions; per-observation tissue depths and a missingness mask mirror the real
    coverage. Pass a scalar `phi` for over-dispersed NB2 counts, else Poisson. Pass `factors` (a
    dict with any of "g","pi","Phi") to inject a KNOWN truth for recovery/boundary tests instead of
    drawing it from the prior (see sample_M_factored).

    The count noise uses its OWN split key (`kc`), independent of the keys that draw the operator
    and the design geometry -- required so observation noise is not a deterministic function of the
    same seed that produced theta (matters for independence across SBC rounds).
    """
    K = L // S
    k0, k2, k3, k4, k5, kc = jax.random.split(key, 6)
    alpha_pi = alpha_pi_matrix(prior, L, S)
    M_true, factors_used = sample_M_factored(k0, prior.mu_g, prior.sigma_g, alpha_pi, prior.beta, L, S,
                                             factors=factors)

    raw = jax.random.exponential(k2, (J, L))
    keep = jax.random.uniform(k3, (J, L)) < sparsity
    # Guarantee >=1 source state per clone: an all-zero source row gives xtilde@M == 0 and an
    # invalid (zero) NB2 mean at its observed states. Only EMPTY rows are touched (their max-raw
    # column is forced on), so denser state spaces -- where empty rows are vanishingly rare -- are
    # bit-identical to before.
    empty = keep.sum(1, keepdims=True) == 0
    keep = (keep | (empty & (raw >= raw.max(1, keepdims=True)))).astype(raw.dtype)
    src = raw * keep
    Xtilde = src / (src.sum(1, keepdims=True) + 1e-9)

    tissue_of = jnp.repeat(jnp.arange(S), K)
    d_tissue = jax.random.randint(k4, (J, S), depth_lo, depth_hi).astype(raw.dtype)
    m_tissue = (jax.random.uniform(k5, (J, S)) > p_missing).astype(raw.dtype)
    D = d_tissue[:, tissue_of] * m_tissue[:, tissue_of]
    mask = (D > 0).astype(raw.dtype)

    Y = sample_counts(kc, M_true, Xtilde, D, mask, phi=phi)
    return Xtilde, Y, D, mask, M_true, factors_used
