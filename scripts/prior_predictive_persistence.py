"""Prior-predictive check: what does the current Gamma(1,1) operator prior BELIEVE about
tissue persistence before it sees any data?

iid Gamma(1,1) on the L=39 entries of each row of M => the row-normalized transition T_row
~ Dirichlet(1,...,1) = UNIFORM over the 39 destination states. So a priori P(descendant stays
in the source tissue) = K/L = 13/39 = 1/3 for every source tissue -- a strongly ANTI-persistence
prior. This script samples the prior, pushes it through tissue_traffic, and overlays the empirical
(data) persistence and the fitted posterior persistence to show the shrinkage.

Env: TRAFFIC_H5 (required), FIT (default nb_global_putative.npz), OUTDIR, N (default 2000).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traffic import data, io, statespace  # noqa: E402
from traffic.readouts import decompose, tissue_traffic  # noqa: E402

RNG = np.random.default_rng(7)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h5 = os.environ.get("TRAFFIC_H5")
    if not h5:
        raise SystemExit("set TRAFFIC_H5")
    fit_path = os.environ.get("FIT", "nb_global_putative.npz")
    outdir = os.environ.get("OUTDIR", "figures/validation/migration")
    N = int(os.environ.get("N", "2000"))
    os.makedirs(outdir, exist_ok=True)

    ss = statespace.default(); S, K, L = ss.S, ss.K, ss.L; tis = list(ss.tissues)
    obs = data.build(data.load_obs_table(h5), ss)
    fit = io.load_fit(fit_path)
    Xt, Y, D = obs.Xtilde, obs.Y, obs.D; J = Xt.shape[0]
    obsv = D > 0; Yt = np.where(obsv, Y, 0.0)
    src_tis = Xt.reshape(J, S, K).sum(2).argmax(1)
    Xr = obs.X.reshape(J, S, K).sum(0); rho = Xr / np.maximum(Xr.sum(1, keepdims=True), 1e-12)

    # DATA-WEIGHTED prediction persistence (matches the source-conditioned PPC): aggregate the
    # predicted destination mass over the real clones per source tissue, row-normalize, take diagonal.
    def dw_persist(M):
        mean = np.where(obsv, D * (Xt @ M), 0.0).reshape(J, S, K).sum(2)   # [J,S] predicted tissue mass
        out = np.zeros(S)
        for a in range(S):
            sel = src_tis == a; v = mean[sel].sum(0); out[a] = (v / max(v.sum(), 1e-12))[a]
        return out

    # empirical (data) persistence: same aggregation on the observed counts
    obs_tm = Yt.reshape(J, S, K).sum(2)
    emp = np.zeros(S)
    for a in range(S):
        sel = src_tis == a; v = obs_tm[sel].sum(0); emp[a] = (v / max(v.sum(), 1e-12))[a]

    post = dw_persist(fit.M_hat)                                            # fitted posterior mean

    # prior-predictive: sample M ~ Gamma(1,1), push through the SAME data-weighted prediction
    Npp = min(N, 400)
    prior = np.empty((Npp, S))
    for i in range(Npp):
        prior[i] = dw_persist(RNG.gamma(1.0, 1.0, (L, L)))
    pmean = prior.mean(0)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    parts = ax.violinplot([prior[:, a] for a in range(S)], positions=range(S), showextrema=False, widths=0.7)
    for b in parts["bodies"]:
        b.set_facecolor("#c44e52"); b.set_alpha(0.35)
    ax.axhline(float(pmean.mean()), color="#c44e52", ls=":", lw=1.5, label=f"prior-predictive mean ≈ {pmean.mean():.2f}")
    ax.scatter(range(S), post, s=140, color="#4c72b0", marker="s", zorder=5, label="fitted posterior")
    ax.scatter(range(S), emp, s=160, color="#111111", marker="D", zorder=6, label="empirical (data)")
    for a in range(S):
        ax.annotate("", xy=(a, emp[a]), xytext=(a, post[a]),
                    arrowprops=dict(arrowstyle="<->", color="grey", lw=1))
        ax.text(a + 0.08, (emp[a] + post[a]) / 2, f"shrink\n{emp[a]-post[a]:+.2f}", fontsize=7, color="grey", va="center")
    ax.set_xticks(range(S)); ax.set_xticklabels(tis); ax.set_ylim(0, 1)
    ax.set_ylabel("P(descendant stays in source tissue)")
    ax.set_title("Prior is biologically backwards: Gamma(1,1) operator expects ~1/3 persistence (uniform scatter)\n"
                 "data wants ~0.6-0.9; the posterior is shrunk toward the anti-persistence prior")
    ax.legend(loc="center right")
    fig.tight_layout(); f = os.path.join(outdir, "prior_predictive_persistence.png"); fig.savefig(f, dpi=140); plt.close(fig)

    print(f"prior-predictive persistence mean: " + "  ".join(f"{tis[a]}={pmean[a]:.3f}" for a in range(S)))
    print(f"  analytic prior mean K/L = {K/L:.3f}")
    print(f"empirical (data)  persistence: " + "  ".join(f"{tis[a]}={emp[a]:.3f}" for a in range(S)))
    print(f"posterior (fit)   persistence: " + "  ".join(f"{tis[a]}={post[a]:.3f}" for a in range(S)))
    print(f"shrinkage (emp - post):        " + "  ".join(f"{tis[a]}={emp[a]-post[a]:+.3f}" for a in range(S)))
    print(f"wrote {f}")


if __name__ == "__main__":
    main()
