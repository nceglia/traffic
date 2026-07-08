"""Recovery panel (Goal 1): simulate one synthetic dataset from a KNOWN truth, fit by NUTS, and show
the posterior lands on the truth -- truth-vs-posterior scatter per factor (g, pi, on-route Phi,
starved Phi, M) plus the global-phi posterior. This is the "posterior contracts to the RIGHT value"
complement to the identifiability/contraction panel (which only shows the posterior moved off the
prior). One PNG to figures/validation/recovery/.

Run:  J=800 W=400 N=400 python scripts/recovery_panel.py
Env:  J, W, N (warmup/samples), SEED, PHI (true NB2 concentration), OUTDIR.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from traffic import statespace
from traffic.config import FactoredPriorConfig, MCMCConfig
from traffic.recovery import recover_once, route_mass


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ss = statespace.default(); S, K, L = ss.S, ss.K, ss.L
    prior = FactoredPriorConfig(mu_g=0.0, sigma_g=0.4, alpha_off=1.0, alpha_stay=4.0,
                                beta=1.0, sigma_phi=1.0)
    J = int(os.environ.get("J", "800"))
    phi = float(os.environ.get("PHI", "10.0"))
    seed = int(os.environ.get("SEED", "0"))
    cfg = MCMCConfig(num_warmup=int(os.environ.get("W", "400")),
                     num_samples=int(os.environ.get("N", "400")), num_chains=2, seed=seed)
    outdir = os.environ.get("OUTDIR", "figures/validation/recovery")
    os.makedirs(outdir, exist_ok=True)

    res, truth, m, _ = recover_once(jax.random.PRNGKey(seed), prior, cfg, ss, J=J, phi=phi)
    print(f"synthetic J={J} L={L} phi_true={phi}")
    print(f"fit: div={res.num_divergences} r_hat={res.r_hat_max:.3f} "
          f"ess_min={res.ess_min:.0f} n_draws={res.n_draws}")
    print(f"  g   : corr={m['g']['corr']:.3f}  cov90={m['g']['coverage']:.2f}  rmse_log={m['g']['rmse_log']:.3f}")
    print(f"  pi  : corr={m['pi']['corr']:.3f}  cov90={m['pi']['coverage']:.2f}  tv={m['pi']['tv']:.3f}")
    print(f"  Phi+: corr={m['Phi_on']['corr']:.3f}  cov90={m['Phi_on']['coverage']:.2f}  "
          f"tv={m['Phi_on']['tv']:.3f}  (n_on={m['Phi_on']['n']})")
    print(f"  Phi-: d_prior={m['Phi_off']['d_prior']:.3f}  ent/logK={m['Phi_off']['entropy_ratio']:.3f}  "
          f"(n_off={m['Phi_off']['n']})")
    print(f"  M   : corr={m['M']['corr']:.3f}  top10={m['M']['corr_top']:.3f}  "
          f"cov90={m['M']['coverage']:.2f}  relFro={m['M']['rel_fro']:.3f}")
    print(f"  phi : true={m['phi']['true']:.2f}  post_mean={m['phi']['post_mean']:.2f}  "
          f"covered={m['phi']['covered']}")

    g_h = res.factors["g"].mean(0); pi_h = res.factors["pi"].mean(0); Phi_h = res.factors["Phi"].mean(0)
    M_h = res.M_hat
    on = route_mass(truth["g"], truth["pi"]) >= m["route"]["tau_on"]
    Phi_t = truth["Phi"]

    def diag(ax, lo, hi):
        ax.plot([lo, hi], [lo, hi], "k:", lw=1, zorder=0)

    fig, ax = plt.subplots(2, 3, figsize=(15, 9.5))

    # g (log-log)
    a = ax[0, 0]
    a.scatter(truth["g"], g_h, s=12, color="#4c72b0", alpha=.7)
    lo, hi = 0.9 * min(truth["g"].min(), g_h.min()), 1.1 * max(truth["g"].max(), g_h.max())
    diag(a, lo, hi); a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("true g"); a.set_ylabel("posterior mean g")
    a.set_title(f"expansion g   corr={m['g']['corr']:.3f}, cov90={m['g']['coverage']:.2f}")

    # pi
    a = ax[0, 1]
    a.scatter(truth["pi"].ravel(), pi_h.ravel(), s=10, color="#dd8452", alpha=.55)
    diag(a, 0, 1); a.set_xlabel("true pi"); a.set_ylabel("posterior mean pi")
    a.set_title(f"trafficking pi   corr={m['pi']['corr']:.3f}, cov90={m['pi']['coverage']:.2f}")

    # Phi on-route
    a = ax[0, 2]
    a.scatter(Phi_t[on].ravel(), Phi_h[on].ravel(), s=10, color="#55a868", alpha=.5)
    diag(a, 0, 1); a.set_xlabel("true Phi (on-route)"); a.set_ylabel("posterior mean Phi")
    a.set_title(f"switching Phi | on-route   corr={m['Phi_on']['corr']:.3f}, "
                f"cov90={m['Phi_on']['coverage']:.2f}")

    # Phi off-route (starved): should collapse to 1/K
    a = ax[1, 0]
    off = ~on
    if off.any():
        a.scatter(Phi_t[off].ravel(), Phi_h[off].ravel(), s=10, color="#8172b3", alpha=.4)
    a.axhline(1.0 / K, color="r", ls="--", lw=1, label="prior mean 1/K")
    diag(a, 0, 1); a.set_xlabel("true Phi (starved)"); a.set_ylabel("posterior mean Phi")
    a.set_title(f"switching Phi | starved   d_prior={m['Phi_off']['d_prior']:.3f}, "
                f"ent/logK={m['Phi_off']['entropy_ratio']:.2f}")
    a.legend(fontsize=8, loc="upper left")

    # M (log-log, top-decile highlighted)
    a = ax[1, 1]
    mt, mh = truth["M"].ravel(), M_h.ravel()
    hi_m = mt >= np.quantile(mt, 0.9)
    a.scatter(mt[~hi_m] + 1e-6, mh[~hi_m] + 1e-6, s=6, color="#bbbbbb", alpha=.4, label="bulk")
    a.scatter(mt[hi_m] + 1e-6, mh[hi_m] + 1e-6, s=10, color="#c44e52", alpha=.7, label="top-decile route mass")
    lo = 1e-4
    diag(a, lo, 1.2 * max(mt.max(), mh.max())); a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("true M entry"); a.set_ylabel("posterior mean M entry")
    a.set_title(f"operator M   corr={m['M']['corr']:.3f}, top10={m['M']['corr_top']:.3f}")
    a.legend(fontsize=8, loc="upper left")

    # global phi posterior
    a = ax[1, 2]
    phi_draws = np.exp(np.asarray(res.dispersion["params"]["log_r"]))
    a.hist(phi_draws, bins=40, color="#4c72b0", alpha=.7)
    a.axvline(truth["phi"], color="r", ls="--", lw=1.5, label=f"true phi={truth['phi']:.1f}")
    a.set_xlabel("NB2 concentration phi (= exp log_r)"); a.set_ylabel("posterior draws")
    a.set_title(f"dispersion phi   covered90={m['phi']['covered']}")
    a.legend(fontsize=8)

    fig.suptitle(f"Parameter recovery under known truth (J={J}, phi={phi}): posterior mean vs truth "
                 f"[div={res.num_divergences}, r_hat={res.r_hat_max:.3f}]", y=1.0)
    fig.tight_layout()
    f = os.path.join(outdir, "recovery_panel.png")
    fig.savefig(f, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("wrote:", f)


if __name__ == "__main__":
    main()
