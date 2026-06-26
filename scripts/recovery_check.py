"""Synthetic recovery check: simulate -> NUTS fit -> compare M_hat to M_true.

Proves the model + inference work independently of the real data.
Run: python scripts/recovery_check.py
"""
import jax

jax.config.update("jax_enable_x64", True)
import numpy as np

from traffic import MCMCConfig, PriorConfig, fit_nuts, simulate


def main():
    Xt, Y, D, mask, M_true = simulate.make_synthetic(jax.random.PRNGKey(0), J=1500)
    res = fit_nuts(Xt, Y, D, PriorConfig(),
                   MCMCConfig(num_warmup=300, num_samples=300, num_chains=2))

    Mh, Mt = np.asarray(res.M_hat), np.asarray(M_true)
    corr = np.corrcoef(Mh.ravel(), Mt.ravel())[0, 1]
    rel_l1 = np.abs(Mh - Mt).sum() / np.abs(Mt).sum()
    growth_corr = np.corrcoef(Mh.sum(1), Mt.sum(1))[0, 1]

    print(f"J={Xt.shape[0]}  L={Xt.shape[1]}  missing frac={(D==0).mean():.2f}")
    print(f"divergences={res.num_divergences}  r_hat_max={res.r_hat_max:.3f}  ess_min={res.ess_min:.0f}")
    print(f"corr(M_hat, M_true)  = {corr:.4f}")
    print(f"rel L1 error         = {rel_l1:.4f}")
    print(f"corr(growth g_z)     = {growth_corr:.4f}")
    ok = corr > 0.9 and growth_corr > 0.9
    print("RECOVERY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
