"""SBC hero figures from stored ranks (scripts/sbc_check.py -> OUTDIR/sbc_results.npz), no re-fit:

  A. Overlaid all-scalar ECDF-difference on ONE shared Sailynoja simultaneous band -- the compact
     calibration hero panel: every parameter's rank ECDF staying inside one band is the single-glance
     "the whole posterior is calibrated" statement.
  B. Coverage curve with the binomial (Monte-Carlo) uncertainty that R rounds actually buy: per-scalar
     empirical coverage against a +/-1.96 se ribbon around the diagonal. Shows the small nominal-vs-
     empirical gaps are consistent with calibration at this round budget, not miscalibration.

Run: python -u scripts/sbc_figures.py   (reads OUTDIR/sbc_results.npz)
Env: OUTDIR (default figures/validation/sbc).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from traffic import sbc


def _ediff(r, Lp, grid):
    return np.searchsorted(np.sort(r / Lp), grid, side="right") / r.size - grid


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = os.environ.get("OUTDIR", "figures/validation/sbc")
    d = np.load(os.path.join(outdir, "sbc_results.npz"), allow_pickle=True)
    ranks = d["ranks"]; Lp = int(d["L_post"]); names = list(d["scalar_names"])
    R, P = ranks.shape
    levels = d["levels"].tolist() if "levels" in d.files else [0.5, 0.8, 0.9, 0.95]
    print(f"loaded {R} rounds x {P} scalars, L_post={Lp}", flush=True)

    # ---- A. overlaid ECDF-difference, one shared simultaneous band ----
    # The band is the null envelope (depends only on R, L_post, alpha) -- compute once, then each
    # scalar's own ECDF-difference is overlaid on it.
    grid, _, lo, hi = sbc.ecdf_diff_band(ranks[:, 0], Lp, n_sim=3000, alpha=0.05, seed=1)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.fill_between(grid, lo, hi, color="grey", alpha=.3, label="95% simultaneous band (null)")
    n_exit = 0
    for i, nm in enumerate(names):
        obs = _ediff(ranks[:, i], Lp, grid)
        out = bool(np.any((obs < lo) | (obs > hi)))
        n_exit += out
        ax.plot(grid, obs, lw=1.1, alpha=.85, color="#c44e52" if out else "#4c72b0",
                label=nm if out else None)
    ax.axhline(0, color="k", lw=.5)
    ax.set_xlabel("fractional rank"); ax.set_ylabel("ECDF(rank) - uniform")
    ax.set_title(f"SBC calibration: all {P} ranked scalars vs one 95% simultaneous band "
                 f"({n_exit} exiting, {R} rounds)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fa = os.path.join(outdir, "sbc_ecdf_overlay.png")
    fig.savefig(fa, dpi=140); plt.close(fig)

    # ---- A'. per-scalar ECDF-difference grid (regenerates sbc_ecdf_bands.png from stored ranks) ----
    ncol = 4; nrow = int(np.ceil(P / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.6 * nrow))
    for i, nm in enumerate(names):
        a = ax.flat[i]
        _, obs, blo, bhi = sbc.ecdf_diff_band(ranks[:, i], Lp, n_sim=1000, alpha=0.05, seed=1)
        a.fill_between(grid, blo, bhi, color="grey", alpha=.3)
        a.plot(grid, obs, color="#c44e52", lw=1); a.axhline(0, color="k", lw=.5)
        a.set_title(nm, fontsize=8)
    for j in range(P, nrow * ncol):
        ax.flat[j].axis("off")
    fig.suptitle("SBC ECDF-difference with 95% simultaneous band (Sailynoja); inside = calibrated")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "sbc_ecdf_bands.png"), dpi=140); plt.close(fig)

    # ---- B. coverage with the Monte-Carlo uncertainty R rounds buy ----
    cov = sbc.interval_coverage(ranks / Lp, levels)                 # {q: [P]}
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    qq = np.linspace(0.4, 1.0, 60)
    se = np.sqrt(qq * (1 - qq) / R)
    ax.fill_between(qq, qq - 1.96 * se, qq + 1.96 * se, color="grey", alpha=.25,
                    label=f"+/-1.96 se at R={R} (calibrated band)")
    for q in levels:
        c = np.asarray(cov[q])
        jit = (np.arange(P) - P / 2) * 0.002
        ax.scatter(np.full(P, q) + jit, c, s=16, color="#4c72b0", alpha=.6,
                   label="per-scalar coverage" if q == levels[0] else None)
        ax.scatter([q], [c.mean()], s=70, color="#c44e52", zorder=5, marker="_",
                   label="mean" if q == levels[0] else None)
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlim(0.4, 1.0); ax.set_ylim(0.3, 1.02)
    ax.set_xlabel("nominal central interval"); ax.set_ylabel("empirical coverage")
    ax.set_title(f"Interval coverage vs Monte-Carlo uncertainty ({R} rounds, {P} scalars)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fb = os.path.join(outdir, "sbc_coverage_band.png")
    fig.savefig(fb, dpi=140); plt.close(fig)

    print("wrote:", fa, fb, flush=True)


if __name__ == "__main__":
    main()
