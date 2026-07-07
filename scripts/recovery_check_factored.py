"""Recovery check for the factored model: simulate from known g,pi,Phi -> fit by NUTS -> verify
M recovery, factor recovery, starved-route Phi -> prior, and convergence.

Run: J=800 W=300 N=300 python scripts/recovery_check_factored.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from traffic import statespace
from traffic.config import FactoredPriorConfig, MCMCConfig
from traffic.simulate import make_synthetic_factored
from traffic.mcmc import fit_nuts

ss = statespace.default(); S, K, L = ss.S, ss.K, ss.L
prior = FactoredPriorConfig(mu_g=0.0, sigma_g=0.4, alpha_off=1.0, alpha_stay=4.0, beta=1.0, sigma_phi=1.0)

key = jax.random.PRNGKey(0)
J = int(os.environ.get("J", "800"))
Xt, Y, D, mask, M_true, ftrue = make_synthetic_factored(key, prior, L=L, J=J, phi=10.0)
Xt, Y, D = np.asarray(Xt), np.asarray(Y), np.asarray(D)
print(f"synthetic: J={Xt.shape[0]} L={L}  Y in [{Y.min():.0f},{Y.max():.0f}]  frac observed={(D>0).mean():.2f}")

cfg = MCMCConfig(num_warmup=int(os.environ.get("W", "300")), num_samples=int(os.environ.get("N", "300")),
                 num_chains=2, seed=0)
res = fit_nuts(Xt, Y, D, prior, cfg)
print(f"fit: divergences={res.num_divergences}  r_hat_max={res.r_hat_max:.3f}  "
      f"ess_min(identified)={res.ess_min:.0f}  n_draws={res.n_draws}")

Mh, Mt = res.M_hat, np.asarray(M_true)
corr = np.corrcoef(Mh.ravel(), Mt.ravel())[0, 1]
mt, mh = Mt.ravel(), Mh.ravel()
hi = mt >= np.quantile(mt, 0.9)
print(f"corr(M_hat, M_true) = {corr:.3f}   |   top-10% route-mass corr = "
      f"{np.corrcoef(mh[hi], mt[hi])[0, 1]:.3f} (n={hi.sum()})")

g_hat, g_true = res.factors["g"].mean(0), np.asarray(ftrue["g"])
pi_hat, pi_true = res.factors["pi"].mean(0), np.asarray(ftrue["pi"])
print(f"corr(g_hat, g_true) = {np.corrcoef(g_hat, g_true)[0, 1]:.3f}   |   "
      f"corr(pi_hat, pi_true) = {np.corrcoef(pi_hat.ravel(), pi_true.ravel())[0, 1]:.3f}")

# starved-route Phi should collapse to the Dir(beta) prior mean 1/K (unidentified switching).
# Guarantee a non-empty starved set so this can never silently no-op.
Phi_hat = res.factors["Phi"].mean(0)
starved = pi_true < 0.02
if not starved.any():
    thresh = np.sort(pi_true.ravel())[min(5, pi_true.size - 1)]
    starved = pi_true <= thresh
dev = float(np.abs(Phi_hat[starved] - 1.0 / K).mean())
print(f"starved-route Phi mean|dev from prior 1/K| = {dev:.3f}  (n={int(starved.sum())} routes) -> ~0 expected")

# Hard gates: a broken assembly/orientation craters corr; bad geometry spikes divergences;
# a Phi-prior-collapse failure blows up dev. r_hat is loose here (short smoke run; production <1.05).
print(f"\ngates: corr>0.90, r_hat<1.15, div<5%, starved-Phi dev<0.10")
assert corr > 0.90, f"M recovery corr {corr:.3f} < 0.90"
assert res.r_hat_max < 1.15, f"r_hat_max {res.r_hat_max:.3f} >= 1.15 (raise W/N)"
assert res.num_divergences < 0.05 * res.n_draws, f"divergences {res.num_divergences} >= 5% of {res.n_draws}"
assert dev < 0.10, f"starved-route Phi deviation {dev:.3f} >= 0.10 (should collapse to prior 1/K)"
print("RECOVERY PASS")
