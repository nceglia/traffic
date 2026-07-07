"""Gradient-based MCMC backend -- sample the factored operator from the posterior.

The one-step mean matrix is factored  M = diag(g) . (pi (x) Phi):
    g_{a,u}       ~ LogNormal(mu_g, sigma_g^2)      per-state expansion factor
    pi_{(a,u)}    ~ Dirichlet(alpha_a)              destination-tissue distribution (trafficking)
    Phi_{(a,u),b} ~ Dirichlet(beta * 1)             destination-phenotype distribution (switching)
    M[z, b*K+v]   = g[z] * pi[z,b] * Phi[z,b,v]     (row sum = g[z]; T = M/g row-stochastic)
    log phi       ~ Normal(0, sigma_phi^2)          single GLOBAL NB2 concentration (NumPyro site "log_r")
    y_j(z')       ~ NB2(mean = d_{j,s'} * (x~ M)(z'), phi)   over non-missing tissues

NUTS samples {g, pi, Phi, log phi} in the unconstrained reparameterization (log for g, NumPyro's
simplex bijections for pi, Phi); M is a deterministic site. NB2 needs float64 (the gammaln/digamma
terms with large counts are f32-unstable). There is one model -- no Gamma path, no dispersion modes.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .config import FactoredPriorConfig, MCMCConfig


def alpha_pi_matrix(prior: FactoredPriorConfig, L: int, S: int = 3) -> np.ndarray:
    """(L, S) Dirichlet concentration for pi: off-tissue columns = alpha_off; the stay column
    (b = tissue(z)) = alpha_stay. State index z = tissue*K + phenotype, so rows [a*K:(a+1)*K] are
    tissue a's states and get the boosted concentration on column a. alpha_stay: scalar -> global;
    length-S -> per source tissue.
    """
    K = L // S
    A = np.full((L, S), float(prior.alpha_off))
    stay = prior.alpha_stay
    stay = [float(stay)] * S if np.isscalar(stay) else [float(x) for x in stay]
    if len(stay) != S:
        raise ValueError(f"alpha_stay must be a scalar or length-{S}; got {prior.alpha_stay!r}")
    for a in range(S):
        A[a * K:(a + 1) * K, a] = stay[a]
    return A


def assemble(g, pi, Phi):
    """M[z, b*K+v] = g[z] * pi[z,b] * Phi[z,b,v].  Shared by the model AND simulate (one reshape
    convention). Shapes: g [L], pi [L,S], Phi [L,S,K] -> M [L, L] with columns ordered b*K+v to
    match statespace z = tissue*K + phenotype.
    """
    L, S, K = Phi.shape
    assert S * K == L, f"assemble: S*K ({S}*{K}) must equal L ({L})"
    return (g[:, None, None] * pi[:, :, None] * Phi).reshape(L, L)


def numpyro_model_factored(Xtilde, Y, D, alpha_pi, beta, mu_g, sigma_g, sigma_phi, S=3):
    """Factored generative model with a global NB2 likelihood. `alpha_pi` [L,S] is the per-state
    tissue-Dirichlet concentration; (beta, mu_g, sigma_g, sigma_phi, S) are the prior scalars.
    """
    import numpyro
    import numpyro.distributions as dist

    L = alpha_pi.shape[0]
    K = L // S
    g = numpyro.sample("g", dist.LogNormal(mu_g, sigma_g).expand([L]).to_event(1))               # [L]
    pi = numpyro.sample("pi", dist.Dirichlet(alpha_pi).to_event(1))                              # [L,S]
    Phi = numpyro.sample("Phi", dist.Dirichlet(beta * jnp.ones(K)).expand([L, S]).to_event(2))   # [L,S,K]
    M = numpyro.deterministic("M", assemble(g, pi, Phi))                                         # [L,L]

    observed = D > 0
    safe_mean = jnp.where(observed, D * (Xtilde @ M), 1.0)     # avoid 0*log(0) at masked states
    log_r = numpyro.sample("log_r", dist.Normal(0.0, sigma_phi))   # NB concentration; site name log_r
    numpyro.sample("Y", dist.NegativeBinomial2(safe_mean, jnp.exp(log_r)).mask(observed), obs=Y)


@dataclass
class MCMCResult:
    samples: np.ndarray      # [N, L, L] posterior draws of the assembled M (chains flattened)
    M_hat: np.ndarray        # posterior mean   [L, L]
    M_median: np.ndarray     # posterior median [L, L]
    sd: np.ndarray           # posterior sd     [L, L]
    num_divergences: int
    r_hat_max: float         # max split-Rhat over the free sites {g,pi,Phi,log_r} (->1 = converged)
    ess_min: float           # min ESS over the IDENTIFIED block {g,pi,log_r} (off-route Phi excluded)
    n_draws: int
    dispersion: dict | None = None   # {"mode": "global", "params": {"log_r": [N]}}
    factors: dict | None = None      # {"g": [N,L], "pi": [N,L,S], "Phi": [N,L,S,K]}

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


def fit_nuts(Xtilde, Y, D, prior: FactoredPriorConfig = FactoredPriorConfig(),
             cfg: MCMCConfig = MCMCConfig()) -> MCMCResult:
    """Sample the factored operator {g, pi, Phi, log phi} from p(. | data) by NUTS."""
    import numpyro
    from numpyro.diagnostics import summary
    from numpyro.infer import MCMC, NUTS

    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError("The factored NB2 fit requires float64 (gammaln/digamma with large "
                           "counts are f32-unstable). Enable it before fitting: "
                           "jax.config.update('jax_enable_x64', True).")
    dtype = jnp.float64                                    # guaranteed by the x64 guard above
    Xt = jnp.asarray(Xtilde, dtype)
    Yj = jnp.asarray(Y, dtype)
    Dj = jnp.asarray(D, dtype)
    L = Xt.shape[1]
    alpha_pi = jnp.asarray(alpha_pi_matrix(prior, L), dtype)

    kernel = NUTS(numpyro_model_factored, target_accept_prob=cfg.target_accept)
    mcmc = MCMC(kernel, num_warmup=cfg.num_warmup, num_samples=cfg.num_samples,
                num_chains=cfg.num_chains, chain_method=cfg.chain_method,
                progress_bar=cfg.progress_bar)
    mcmc.run(jax.random.PRNGKey(cfg.seed), Xt, Yj, Dj, alpha_pi,
             prior.beta, prior.mu_g, prior.sigma_g, prior.sigma_phi, extra_fields=("diverging",))

    div = int(np.sum(jax.device_get(mcmc.get_extra_fields()["diverging"])))
    samples = mcmc.get_samples()
    grouped = mcmc.get_samples(group_by_chain=True)

    # Convergence on the FREE sites (not the deterministic M): the funnel lives in the factor
    # geometry. r_hat over all free sites; ESS over the identified block {g,pi,log_r} -- off-route
    # Phi directions are prior-dominated and have intrinsically low ESS (not a mixing failure).
    free = {k: np.asarray(grouped[k]) for k in ("g", "pi", "Phi", "log_r")}
    diag = summary(free, group_by_chain=True)
    # NaN-robust: numpyro's summary returns all-NaN r_hat/n_eff for a (near-)constant site
    # (prior-dominated off-route pi/Phi columns can collapse). Flatten and guard the all-NaN case
    # so the gate fails safe (inf r_hat / 0 ESS) rather than silently reporting NaN.
    rhat_all = np.concatenate([np.asarray(diag[k]["r_hat"]).ravel() for k in free])
    ess_all = np.concatenate([np.asarray(diag[k]["n_eff"]).ravel() for k in ("g", "pi", "log_r")])
    r_hat_max = float(np.nanmax(rhat_all)) if np.isfinite(rhat_all).any() else float("inf")
    ess_min = float(np.nanmin(ess_all)) if np.isfinite(ess_all).any() else 0.0

    flat = np.asarray(samples["M"])                                    # [C*N, L, L]
    factors = {k: np.asarray(samples[k]) for k in ("g", "pi", "Phi")}
    dispersion = {"mode": "global", "params": {"log_r": np.asarray(samples["log_r"])}}

    return MCMCResult(
        samples=flat, M_hat=flat.mean(0), M_median=np.median(flat, 0), sd=flat.std(0),
        num_divergences=div, r_hat_max=r_hat_max, ess_min=ess_min,
        n_draws=flat.shape[0], dispersion=dispersion, factors=factors,
    )
