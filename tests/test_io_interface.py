"""io round-trip + downstream-interface check: a factored fit saves/reloads with .samples,
.dispersion, and .factors, carries parameterization=factored, and decompose(M) is consistent."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from traffic import statespace, io, readouts
from traffic.config import FactoredPriorConfig
from traffic.simulate import sample_M_factored
from traffic.mcmc import alpha_pi_matrix, MCMCResult

ss = statespace.default(); S, K, L = ss.S, ss.K, ss.L
prior = FactoredPriorConfig()
alpha_pi = alpha_pi_matrix(prior, L)
M, fac = sample_M_factored(jax.random.PRNGKey(3), prior.mu_g, prior.sigma_g, alpha_pi, prior.beta, L)
M = np.asarray(M)
g_true, pi_true, Phi_true = (np.asarray(fac[k]) for k in ("g", "pi", "Phi"))

N = 40
rng = np.random.default_rng(0)
samples = np.repeat(M[None], N, 0) + rng.normal(0, 1e-6, (N, L, L))
factors = {k: np.repeat(np.asarray(fac[k])[None], N, 0) for k in ("g", "pi", "Phi")}
disp = {"mode": "global", "params": {"log_r": np.zeros(N)}}
res = MCMCResult(samples=samples, M_hat=samples.mean(0), M_median=np.median(samples, 0),
                 sd=samples.std(0), num_divergences=0, r_hat_max=1.0, ess_min=100.0,
                 n_draws=N, dispersion=disp, factors=factors)

rho = np.full((S, K), 1.0 / K)
with tempfile.TemporaryDirectory() as td:
    p = io.save_fit(os.path.join(td, "f.npz"), res, ss=ss, rho=rho)
    fit = io.load_fit(p)

assert fit.samples.shape == (N, L, L), "M draws must round-trip"
assert fit.dispersion["mode"] == "global" and "log_r" in fit.dispersion["params"], "global NB dispersion"
assert fit.factors is not None and set(fit.factors) == {"g", "pi", "Phi"}, "factors additive"
assert fit.meta.get("parameterization") == "factored", "meta flags factored"
assert fit.meta.get("family") == "nb", "family=nb (global NB2)"

# decompose(M) recovers g = row sum, and T = M/g row-stochastic; g must equal the sampled g
T, g_rec = readouts.decompose(fit.M_hat)
assert np.allclose(g_rec, fit.M_hat.sum(1)), "decompose g == row sum"
assert np.allclose(g_rec, g_true, atol=1e-4), "decompose g == sampled expansion factor"
assert np.allclose(T.sum(1), 1.0, atol=1e-6), "T row-stochastic"
print("PASS io round-trip + factored meta + decompose(M) consistency")
