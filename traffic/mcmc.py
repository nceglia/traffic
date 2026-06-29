"""Gradient-based MCMC backend -- sample M (and dispersion) from the posterior.

By Poisson superposition the per-source allocations of each destination count
marginalize exactly, leaving M (plus, under a Negative-Binomial likelihood, a few
dispersion parameters) as the unknowns -- a smooth target with no auxiliary
allocation variables. NUTS samples it directly; positivity is handled by NumPyro's
automatic log-transform of the constrained supports.

Likelihood (see config.LikelihoodConfig):
    family = "poisson":  y ~ Poisson(d * x~ M)
    family = "nb":       y ~ NegBinomial2(mean = d * x~ M,  concentration r = exp(log_r))
                         dispersion = none | global | tissue | patient(=tissue x patient)

Dispersion uses a log-concentration parameterization (log_r ~ Normal): smooth for
HMC, a soft Poisson limit at log_r -> +inf, and no 1/alpha reciprocal or alpha->0
funnel. NB requires float64 (the gammaln/digamma terms with large counts are
f32-unstable).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from .config import LikelihoodConfig, MCMCConfig, PriorConfig

_DISP_SITES = {"global": ["log_r"], "tissue": ["log_r_s"], "patient": ["mu_s", "sigma", "eps"]}


def numpyro_model(Xtilde, Y, D, a0, b0, family, dispersion, alpha_scale, sigma_scale,
                  col_tissue, patient_idx, n_patient):
    """One generative model; Poisson or NB with structured dispersion.

    Static args (family, dispersion, scales, n_patient) drive control flow at trace
    time; col_tissue [L] and patient_idx [J] are integer index arrays.
    """
    import numpyro
    import numpyro.distributions as dist

    L = Xtilde.shape[1]
    M = numpyro.sample("M", dist.Gamma(jnp.full((L, L), a0), jnp.full((L, L), b0)).to_event(2))
    mean = D * (Xtilde @ M)                        # [J, L]
    observed = D > 0
    safe_mean = jnp.where(observed, mean, 1.0)     # avoid 0*log(0) at masked states

    if family == "poisson":
        numpyro.sample("Y", dist.Poisson(safe_mean).mask(observed), obs=Y)
        return

    # Negative-Binomial: concentration r = exp(log_r), log_r ~ Normal (soft Poisson
    # limit at log_r -> +inf; no reciprocal / alpha->0 funnel).
    if dispersion == "global":
        log_r = numpyro.sample("log_r", dist.Normal(0.0, alpha_scale))
    elif dispersion == "tissue":
        log_r_s = numpyro.sample("log_r_s", dist.Normal(0.0, alpha_scale).expand([3]).to_event(1))
        log_r = log_r_s[col_tissue]                # [L]
    elif dispersion == "patient":                  # tissue x patient, non-centered, pooled
        mu_s = numpyro.sample("mu_s", dist.Normal(0.0, alpha_scale).expand([3]).to_event(1))
        sigma = numpyro.sample("sigma", dist.HalfNormal(sigma_scale))
        eps = numpyro.sample("eps", dist.Normal(0.0, 1.0).expand([n_patient, 3]).to_event(2))
        log_r = (mu_s[None, :] + sigma * eps)[patient_idx][:, col_tissue]   # [J, L]
    else:
        raise ValueError(f"unknown dispersion {dispersion!r}")
    r = jnp.exp(log_r)
    numpyro.sample("Y", dist.NegativeBinomial2(safe_mean, jnp.broadcast_to(r, mean.shape)).mask(observed),
                   obs=Y)


@dataclass
class MCMCResult:
    samples: np.ndarray      # [N, L, L] posterior draws of M (chains flattened)
    M_hat: np.ndarray        # posterior mean   [L, L]
    M_median: np.ndarray     # posterior median [L, L]
    sd: np.ndarray           # posterior sd     [L, L]
    num_divergences: int
    r_hat_max: float         # max split-Rhat over entries (->1 = converged)
    ess_min: float           # min effective sample size over entries
    n_draws: int
    dispersion: dict | None = None   # {"mode": str, "params": {site: [N,...]}} for NB; None for Poisson

    @property
    def mean(self):
        return self.M_hat

    def draws(self, n=None, key=None):
        """Return n posterior draws [n, L, L] (a random subset if n < n_draws)."""
        if n is None or n >= self.n_draws:
            return self.samples
        if key is None:
            key = jax.random.PRNGKey(0)
        idx = np.asarray(jax.random.choice(key, self.n_draws, (n,), replace=False))
        return self.samples[idx]


def fit_nuts(Xtilde, Y, D, prior: PriorConfig = PriorConfig(), cfg: MCMCConfig = MCMCConfig(),
             lik: LikelihoodConfig = LikelihoodConfig(), patient_idx=None) -> MCMCResult:
    """Sample M (and NB dispersion, if any) from p(. | data) by NUTS."""
    import numpyro
    from numpyro.diagnostics import summary
    from numpyro.infer import MCMC, NUTS

    if lik.family == "nb" and not jax.config.read("jax_enable_x64"):
        import warnings
        warnings.warn("NB likelihood is numerically unstable in float32 (the gammaln/"
                      "digamma terms with large counts); enable x64 via "
                      "jax.config.update('jax_enable_x64', True).", RuntimeWarning)
    dtype = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32
    Xt = jnp.asarray(Xtilde, dtype)
    Yj = jnp.asarray(Y, dtype)
    Dj = jnp.asarray(D, dtype)
    L = Xt.shape[1]
    col_tissue = jnp.repeat(jnp.arange(3), L // 3)         # destination tissue of each column

    n_patient = 0
    if lik.dispersion == "patient":
        if patient_idx is None:
            raise ValueError("dispersion='patient' requires patient_idx (int array [J])")
        patient_idx = jnp.asarray(patient_idx, jnp.int32)
        n_patient = int(np.asarray(patient_idx).max()) + 1
    else:
        patient_idx = jnp.zeros(Xt.shape[0], jnp.int32)   # unused placeholder

    kernel = NUTS(numpyro_model, target_accept_prob=cfg.target_accept)
    mcmc = MCMC(kernel, num_warmup=cfg.num_warmup, num_samples=cfg.num_samples,
                num_chains=cfg.num_chains, chain_method=cfg.chain_method,
                progress_bar=cfg.progress_bar)
    mcmc.run(jax.random.PRNGKey(cfg.seed), Xt, Yj, Dj, prior.a0, prior.b0,
             lik.family, lik.dispersion, lik.alpha_scale, lik.sigma_scale,
             col_tissue, patient_idx, n_patient, extra_fields=("diverging",))

    div = int(np.sum(jax.device_get(mcmc.get_extra_fields()["diverging"])))
    samples = mcmc.get_samples()
    grouped = np.asarray(mcmc.get_samples(group_by_chain=True)["M"])   # [C, N, L, L]
    diag = summary({"M": grouped}, group_by_chain=True)["M"]
    flat = np.asarray(samples["M"])                                   # [C*N, L, L]

    dispersion = None
    if lik.family == "nb":
        dispersion = {"mode": lik.dispersion,
                      "params": {s: np.asarray(samples[s]) for s in _DISP_SITES[lik.dispersion]}}

    return MCMCResult(
        samples=flat, M_hat=flat.mean(0), M_median=np.median(flat, 0), sd=flat.std(0),
        num_divergences=div, r_hat_max=float(np.nanmax(diag["r_hat"])),
        ess_min=float(np.nanmin(diag["n_eff"])), n_draws=flat.shape[0], dispersion=dispersion,
    )
