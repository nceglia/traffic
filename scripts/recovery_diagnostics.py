"""Recovery diagnostics (Goal 1, "learned the data" figures). One fit at known truth, three panels:

  A. Posterior z-score vs SHRINKAGE scatter -- the canonical Schad/Betancourt/Vasishth (2021)
     diagnostic that separates "the model is confident" from "the model is right". x = shrinkage
     s=1-var_post/var_prior (right = data informative), y = z=(mean_post-truth)/sd_post (0 = on-truth).
     Ideal cloud: right & y~0. Starved-Phi / pi->0 directions sit at LEFT (prior-limited, not broken).
  B. Contraction: distribution of shrinkage per factor -- most parameters contract strongly; the
     starved directions do not, by construction.
  C. Forest / caterpillar: truth marker inside sorted posterior intervals, per parameter, with the
     empirical 90% coverage annotated.

Run:  J=800 W=500 N=500 python -u scripts/recovery_diagnostics.py
Env:  J, W, N, SEED, PHI, OUTDIR.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from traffic import statespace
from traffic.config import FactoredPriorConfig, MCMCConfig
from traffic.recovery import prior_variance, recover_once, route_mass, shrinkage_zscore

TAU = 0.02
TISSUE_COL = {0: "#4c72b0", 1: "#dd8452", 2: "#55a868"}     # per-tissue label color (diagnostic)
_PH_SHORT = ["TEMRA", "TEXeff", "TEXterm", "TRM", "Mem", "Naive", "TEXprog",
             "4Naive", "4Act", "4QExh", "4Exh", "Treg", "Th"]


def _state_labels(ss):
    """[L] compact tissue:phenotype labels for the state space (falls back to raw names)."""
    if ss.K == len(_PH_SHORT):
        return [f"{ss.tissues[a]}:{_PH_SHORT[u]}" for a in range(ss.S) for u in range(ss.K)]
    return ss.state_labels()


def _forest(ax, draws, truth, ss, title):
    """Sorted posterior 50/90% intervals + median, state labels colored by tissue, truth marked
    green inside / red outside the 90% interval; annotate 90% coverage."""
    labels = _state_labels(ss); tof = np.asarray(ss.tissue_of)
    order = np.argsort(truth)
    med = np.median(draws, 0)
    lo50, hi50 = np.quantile(draws, [.25, .75], 0)
    lo90, hi90 = np.quantile(draws, [.05, .95], 0)
    t = np.asarray(truth)
    y = np.arange(len(t))
    for i, o in enumerate(order):
        ax.plot([lo90[o], hi90[o]], [i, i], color="#cccccc", lw=1, zorder=1)
        ax.plot([lo50[o], hi50[o]], [i, i], color="#4c72b0", lw=3, zorder=2)
    ax.scatter(med[order], y, s=9, color="#1f3b70", zorder=3, label="posterior (median, 50%, 90%)")
    covered = (lo90[order] <= t[order]) & (t[order] <= hi90[order])
    ax.scatter(t[order][covered], y[covered], marker="D", s=16, color="#55a868", zorder=4,
               label="truth inside 90%")
    ax.scatter(t[order][~covered], y[~covered], marker="D", s=40, color="#c44e52", zorder=5,
               edgecolor="k", lw=.4, label="truth OUTSIDE 90%")
    ax.set_ylim(-1, len(t)); ax.set_yticks(y)
    ax.set_yticklabels([labels[o] for o in order], fontsize=5)
    for tick, o in zip(ax.get_yticklabels(), order):
        tick.set_color(TISSUE_COL.get(int(tof[o]), "#000000"))
    ax.set_title(f"{title}   (90% CI covers {covered.mean():.0%})", fontsize=10)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    ss = statespace.default(); S, K, L = ss.S, ss.K, ss.L
    prior = FactoredPriorConfig(mu_g=0.0, sigma_g=0.4, alpha_off=1.0, alpha_stay=4.0,
                                beta=1.0, sigma_phi=1.0)
    J = int(os.environ.get("J", "800")); phi = float(os.environ.get("PHI", "10.0"))
    seed = int(os.environ.get("SEED", "0"))
    cfg = MCMCConfig(num_warmup=int(os.environ.get("W", "500")),
                     num_samples=int(os.environ.get("N", "500")), num_chains=2, seed=seed)
    outdir = os.environ.get("OUTDIR", "figures/validation/recovery")
    os.makedirs(outdir, exist_ok=True)

    res, truth, m, _ = recover_once(jax.random.PRNGKey(seed), prior, cfg, ss, J=J, phi=phi)
    print(f"fit: div={res.num_divergences} r_hat={res.r_hat_max:.3f} n_draws={res.n_draws}", flush=True)
    pv = prior_variance(prior, ss, n=8000, seed=1)
    sz = shrinkage_zscore(res, truth, pv)

    # ---- A. z-score vs shrinkage (headline) ----
    on = (route_mass(truth["g"], truth["pi"]) >= TAU)[:, :, None] & np.ones((1, 1, K), bool)
    groups = [
        ("g (expansion)", sz["g"]["s"].ravel(), sz["g"]["z"].ravel(), "#4c72b0", 14),
        ("pi (trafficking)", sz["pi"]["s"].ravel(), sz["pi"]["z"].ravel(), "#dd8452", 9),
        ("Phi on-route", sz["Phi"]["s"][on].ravel(), sz["Phi"]["z"][on].ravel(), "#55a868", 8),
        ("Phi starved", sz["Phi"]["s"][~on].ravel(), sz["Phi"]["z"][~on].ravel(), "#8172b3", 8),
        ("log_r (dispersion)", sz["log_r"]["s"], sz["log_r"]["z"], "#000000", 60),
    ]
    fig, ax = plt.subplots(figsize=(9, 7))
    for lbl, s, z, c, ms in groups:
        s = np.asarray(s); z = np.asarray(z)
        ok = np.isfinite(s) & np.isfinite(z)
        ax.scatter(s[ok], np.clip(z[ok], -6, 6), s=ms, color=c, alpha=.55, label=f"{lbl} (n={ok.sum()})",
                   edgecolors="none")
    ax.axhline(0, color="k", lw=.8); ax.axhline(2, color="grey", ls=":", lw=.8); ax.axhline(-2, color="grey", ls=":", lw=.8)
    ax.axvline(0, color="k", lw=.5)
    ax.set_xlim(-0.15, 1.05); ax.set_ylim(-6.3, 6.3)
    ax.set_xlabel("posterior shrinkage   s = 1 - var_post/var_prior   (right = data informative)")
    ax.set_ylabel("posterior z-score   z = (mean_post - truth)/sd_post   (0 = on truth)")
    ax.text(0.99, 0.03, "ideal:\ninformative\n& on-truth", ha="right", va="bottom",
            transform=ax.transAxes, fontsize=8, color="#333333")
    ax.text(0.01, 0.03, "prior-limited\n(unidentified)", ha="left", va="bottom",
            transform=ax.transAxes, fontsize=8, color="#333333")
    ax.text(0.99, 0.97, "confident\nbut biased", ha="right", va="top",
            transform=ax.transAxes, fontsize=8, color="#a11")
    ax.legend(loc="center left", fontsize=8, framealpha=.9)
    ax.set_title(f"Posterior z-score vs shrinkage — did the model learn AND land on truth? "
                 f"(J={J}, {res.n_draws} draws)")
    fig.tight_layout(); fa = os.path.join(outdir, "recovery_shrinkage_zscore.png")
    fig.savefig(fa, dpi=140, bbox_inches="tight"); plt.close(fig)

    # ---- B. contraction (shrinkage distribution per factor) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [np.asarray(g[1])[np.isfinite(g[1])] for g in groups]
    labels = [g[0] for g in groups]
    parts = ax.violinplot([d for d in data], showmeans=True, showextrema=False)
    for b, g in zip(parts["bodies"], groups):
        b.set_facecolor(g[3]); b.set_alpha(.5)
    ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.axhline(1.0, color="grey", ls=":", lw=.8); ax.axhline(0.0, color="grey", ls=":", lw=.8)
    ax.set_ylabel("posterior shrinkage s"); ax.set_ylim(-0.1, 1.05)
    ax.set_title("Prior->posterior contraction by parameter block "
                 "(1 = data-dominated; ~0 = prior-limited)")
    fig.tight_layout(); fb = os.path.join(outdir, "recovery_contraction.png")
    fig.savefig(fb, dpi=140, bbox_inches="tight"); plt.close(fig)

    # ---- C. forest with truth markers (g and pi_stay: both length L, interpretable) ----
    pi_stay_draws = np.asarray(res.factors["pi"])[:, np.arange(L), ss.tissue_of]   # [N,L]
    pi_stay_true = np.asarray(truth["pi"])[np.arange(L), ss.tissue_of]
    fig, ax = plt.subplots(1, 2, figsize=(13, 12))
    _forest(ax[0], np.asarray(res.factors["g"]), np.asarray(truth["g"]), ss, "g (expansion) per state")
    _forest(ax[1], pi_stay_draws, pi_stay_true, ss, "pi_stay (persistence) per state")
    ax[0].legend(fontsize=7, loc="lower right")
    ax[0].set_xlabel("g"); ax[1].set_xlabel("stay probability")
    tissue_handles = [Patch(color=TISSUE_COL[i], label=ss.tissues[i]) for i in range(ss.S)]
    ax[1].legend(handles=tissue_handles, fontsize=7, loc="lower right", title="label color = tissue")
    fig.suptitle("Truth inside sorted posterior intervals — states labeled, colored by tissue "
                 "(red diamond = truth outside 90% CI)")
    fig.tight_layout(); fc = os.path.join(outdir, "recovery_forest.png")
    fig.savefig(fc, dpi=140, bbox_inches="tight"); plt.close(fig)

    print("wrote:", fa, fb, fc, flush=True)


if __name__ == "__main__":
    main()
