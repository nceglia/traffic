"""Gradient-based MCMC backend -- sample M directly from the marginal posterior.

This is the executable form of the "Posterior sampling" subsection of
model_methods.tex.  By Poisson superposition the per-source allocations of each
destination count marginalize exactly, leaving M as the only unknown: a smooth
target over the L x L entries with no auxiliary allocation variables.  We sample
it with NUTS (Hamiltonian Monte Carlo) on the same marginal Poisson likelihood
the CAVI core uses -- positivity is handled by NumPyro's automatic log-transform
of the Gamma support, so the chain explores the unconstrained reparameterization.

Why this exists alongside fit() (CAVI): CAVI's mean-field q(M) factorizes across
entries and systematically *under*-estimates posterior variance (credible
intervals too narrow).  NUTS draws from the full joint posterior, so its sample
spread is calibrated.  Point estimates (M_hat) agree closely between the two;
the uncertainty does not.  See scripts/mcmc_check.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .config import MCMCConfig, PriorConfig


def numpyro_model(Xtilde, Y, D, a0, b0):
    """Marginal Gamma-Poisson model: M ~ Gamma(a0,b0); y ~ Poisson(D * Xtilde M).

    The likelihood is masked at unprofiled states (D == 0 contributes nothing),
    matching model.poisson_loglik.  No latent allocations -- M is sampled directly.
    """
    import numpyro
    import numpyro.distributions as dist

    L = Xtilde.shape[1]
    M = numpyro.sample(
        "M", dist.Gamma(jnp.full((L, L), a0), jnp.full((L, L), b0)).to_event(2)
    )
    rate = D * (Xtilde @ M)                       # [J, L]
    observed = D > 0
    safe_rate = jnp.where(observed, rate, 1.0)    # avoid 0*log(0) at masked states
    numpyro.sample("Y", dist.Poisson(safe_rate).mask(observed), obs=Y)


@dataclass
class MCMCResult:
    samples: np.ndarray      # [N, L, L] posterior draws of M (chains flattened)
    M_hat: np.ndarray        # posterior mean   [L, L]
    M_median: np.ndarray     # posterior median [L, L]
    sd: np.ndarray           # posterior sd     [L, L]  (calibrated, full-posterior)
    num_divergences: int
    r_hat_max: float         # max split-Rhat over entries (->1 = converged)
    ess_min: float           # min effective sample size over entries
    n_draws: int

    @property
    def mean(self):
        return self.M_hat

    def draws(self, n=None, key=None):
        """Return n posterior draws [n, L, L] (a random subset if n < n_draws).

        Mirrors posterior.sample_M's output shape so read-out code that maps over
        draws works unchanged with either backend.
        """
        if n is None or n >= self.n_draws:
            return self.samples
        if key is None:
            key = jax.random.PRNGKey(0)
        idx = np.asarray(jax.random.choice(key, self.n_draws, (n,), replace=False))
        return self.samples[idx]


def fit_nuts(Xtilde, Y, D, prior: PriorConfig = PriorConfig(),
             cfg: MCMCConfig = MCMCConfig()) -> MCMCResult:
    """Sample M from p(M | data) by NUTS.  Drop-in uncertainty backend for fit()."""
    import numpyro
    from numpyro.diagnostics import summary
    from numpyro.infer import MCMC, NUTS

    dtype = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32
    Xt = jnp.asarray(Xtilde, dtype)
    Yj = jnp.asarray(Y, dtype)
    Dj = jnp.asarray(D, dtype)

    kernel = NUTS(numpyro_model, target_accept_prob=cfg.target_accept)
    mcmc = MCMC(
        kernel,
        num_warmup=cfg.num_warmup,
        num_samples=cfg.num_samples,
        num_chains=cfg.num_chains,
        chain_method=cfg.chain_method,
        progress_bar=cfg.progress_bar,
    )
    mcmc.run(jax.random.PRNGKey(cfg.seed), Xt, Yj, Dj, prior.a0, prior.b0,
             extra_fields=("diverging",))

    div = int(np.sum(jax.device_get(mcmc.get_extra_fields()["diverging"])))
    grouped = np.asarray(mcmc.get_samples(group_by_chain=True)["M"])   # [C, N, L, L]
    diag = summary({"M": grouped}, group_by_chain=True)["M"]
    r_hat_max = float(np.nanmax(diag["r_hat"]))
    ess_min = float(np.nanmin(diag["n_eff"]))

    flat = np.asarray(mcmc.get_samples()["M"])                         # [C*N, L, L]
    return MCMCResult(
        samples=flat,
        M_hat=flat.mean(0),
        M_median=np.median(flat, 0),
        sd=flat.std(0),
        num_divergences=div,
        r_hat_max=r_hat_max,
        ess_min=ess_min,
        n_draws=flat.shape[0],
    )
