"""Within-tissue phenotypic switching as a circular transition graph (model operator).

The model counterpart of scratch/phenotype_transitions_putative.py (which draws EMPIRICAL
dominant-phenotype switch counts). Here the edges come from the fitted, validated
within-tissue switching operator H[a] = within_tissue_switching(T)[a] -- P(dest phenotype v |
stayed in tissue a, source phenotype u) -- weighted by source occupancy rho, so an edge is the
occupancy-weighted VOLUME of u->v switching within a tissue. Share-supported (redistribution /
composition; see traffic/readouts.py CONTRACT). Same circular CD4/CD8 layout, node coloring, and
cross-lineage / to-naive / other-switch edge typing as the empirical figure, for direct comparison.

Posterior twist over the empirical version: edges resolved by the posterior (95% CrI of the flux
excludes ~0) are drawn SOLID; unresolved edges are faded -- so you can see which switches the
data actually pins down.

Panels: GLOBAL (tissue-averaged) + within blood / CSF / tumor.
Env: FIT (default nb_global_putative.npz), OUTDIR (default figures/validation/switching), DRAWS (300), TOPN (24).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

from traffic import io, statespace
from traffic.readouts import within_tissue_switching, decompose

RNG = np.random.default_rng(0)

# biological CD4-arc-then-CD8-arc order (our 13-phenotype scheme), matching the empirical layout.
ORDER = ["CD4_Naive", "CD4_Activated", "CD4_Quiescent_Exhausted", "CD4_Exhausted", "CD4_Th", "CD4_Treg",
         "CD8_Quiescent_Naive", "CD8_Quiescent_Memory", "CD8_Quiescent_TEXprog", "CD8_Activated_TEXeff",
         "CD8_Activated_TEXterm", "CD8_Activated_TRM", "CD8_Activated_TEMRA"]
SHORT = {"CD4_Naive": "4·Nv", "CD4_Activated": "4·Act", "CD4_Quiescent_Exhausted": "4·QEx",
         "CD4_Exhausted": "4·Ex", "CD4_Th": "4·Th", "CD4_Treg": "4·Treg",
         "CD8_Quiescent_Naive": "8·Nv", "CD8_Quiescent_Memory": "8·Mem", "CD8_Quiescent_TEXprog": "8·TEXpr",
         "CD8_Activated_TEXeff": "8·TEXeff", "CD8_Activated_TEXterm": "8·TEXtm",
         "CD8_Activated_TRM": "8·TRM", "CD8_Activated_TEMRA": "8·TEMRA"}
NAIVE = {"CD4_Naive", "CD8_Quiescent_Naive"}
DISP = {"PBMC": "blood", "CSF": "CSF", "TP": "tumor"}


def _lin(p):
    return p[:3]


def _cat(a, b):
    if _lin(a) != _lin(b):
        return "cross-lineage"
    if b in NAIVE and a != b:
        return "to-naive"
    return "ok"


def _flux(T, ss, rho):
    """Per-tissue occupancy-weighted within-tissue switching flux [S,K,K] (diagonal zeroed)."""
    H = within_tissue_switching(T, ss)
    F = np.zeros((ss.S, ss.K, ss.K))
    for a in range(ss.S):
        F[a] = rho[a][:, None] * H[a]
        np.fill_diagonal(F[a], 0.0)
    return F


def _draw(ax, W, Wlo, names, title, topn):
    """W, Wlo are [n,n] in ORDER; names are ORDER phenotype names."""
    n = len(names)
    ang = np.pi / 2 - 2 * np.pi * np.arange(n) / n
    pos = {i: (np.cos(ang[i]), np.sin(ang[i])) for i in range(n)}
    thru = W.sum(0) + W.sum(1)
    off = sorted([(W[a, b], a, b) for a in range(n) for b in range(n) if a != b and W[a, b] > 0],
                 reverse=True)[:topn]
    mx = max((w for w, _, _ in off), default=1e-9)
    ccol = {"cross-lineage": "#e8202a", "to-naive": "#f08a24", "ok": "#7c8a99"}
    for w, a, b in off:
        cat = _cat(names[a], names[b]); resolved = Wlo[a, b] > 0.02 * mx
        ax.add_patch(FancyArrowPatch(
            pos[a], pos[b], connectionstyle="arc3,rad=0.16", arrowstyle="-|>",
            mutation_scale=10 + 8 * w / mx, lw=0.6 + 5.5 * w / mx, color=ccol[cat],
            alpha=(.85 if cat != "ok" else .55) if resolved else 0.18,
            linestyle="-" if resolved else (0, (2, 2)),
            shrinkA=13, shrinkB=14, zorder=3 if cat != "ok" else 2))
    for i in range(n):
        x, y = pos[i]
        ax.scatter(x, y, s=70 + 240 * thru[i] / max(thru.max(), 1e-9),
                   color="#2c7fb8" if _lin(names[i]) == "CD4" else "#d8543a", zorder=4, ec="k", lw=.5)
        ax.text(x * 1.2, y * 1.2, SHORT.get(names[i], names[i]), ha="center", va="center", fontsize=7, zorder=5)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.45, 1.45); ax.set_aspect("equal"); ax.axis("off")
    sw = W.sum()
    xl = sum(W[a, b] for a in range(n) for b in range(n) if a != b and _lin(names[a]) != _lin(names[b]))
    nv = sum(W[a, b] for a in range(n) for b in range(n) if a != b and names[b] in NAIVE and a != b)
    ax.set_title(f"{title}\n{xl/max(sw,1e-9)*100:.0f}% cross-lineage · {nv/max(sw,1e-9)*100:.0f}% to-naive "
                 f"(switch mass; edges: solid=resolved)", fontsize=10)


def main():
    ss = statespace.default(); S, K = ss.S, ss.K
    fit_path = os.environ.get("FIT", "nb_global_putative.npz")
    outdir = os.environ.get("OUTDIR", "figures/validation/switching")
    n_draws = int(os.environ.get("DRAWS", "300")); topn = int(os.environ.get("TOPN", "24"))
    os.makedirs(outdir, exist_ok=True)
    fit = io.load_fit(fit_path)
    phenos = [str(p) for p in (getattr(fit, "phenotypes", None) if getattr(fit, "phenotypes", None) is not None else ss.phenotypes)]
    tis = [str(t) for t in (getattr(fit, "tissues", None) if getattr(fit, "tissues", None) is not None else ss.tissues)]
    rho = np.asarray(fit.rho) if getattr(fit, "rho", None) is not None else np.full((S, K), 1.0 / K)
    perm = [phenos.index(p) for p in ORDER if p in phenos]                 # reindex to biological ORDER
    names = [phenos[i] for i in perm]
    print(f"fit={fit_path}  states={fit.M_hat.shape[0]}  ordered {len(names)} phenotypes")

    idx = np.sort(RNG.choice(fit.samples.shape[0], min(n_draws, fit.samples.shape[0]), replace=False))
    FL = np.empty((len(idx), S, K, K))
    for i, si in enumerate(idx):
        FL[i] = _flux(decompose(fit.samples[si])[0], ss, rho)
    fl_m = FL.mean(0); fl_lo = np.quantile(FL, .025, 0)                    # [S,K,K]
    glob_m = fl_m.mean(0); glob_lo = fl_lo.mean(0)                         # tissue-averaged

    def to_order(M):
        return M[np.ix_(perm, perm)]

    panels = [("GLOBAL (tissue-averaged)", to_order(glob_m), to_order(glob_lo))]
    for a, t in enumerate(tis):
        panels.append((f"within {DISP.get(t, t)}", to_order(fl_m[a]), to_order(fl_lo[a])))

    fig, axes = plt.subplots(2, 2, figsize=(13, 13))
    for ax, (name, W, Wlo) in zip(axes.ravel(), panels):
        _draw(ax, W, Wlo, names, name, topn)
    fig.legend(handles=[Line2D([0], [0], color="#e8202a", lw=3, label="cross-lineage"),
                        Line2D([0], [0], color="#f08a24", lw=3, label="to-naive"),
                        Line2D([0], [0], color="#7c8a99", lw=3, label="other switch"),
                        Line2D([0], [0], color="#555", lw=2, ls=(0, (2, 2)), label="unresolved (95% CrI ~ 0)")],
               loc="lower center", ncol=4, frameon=False, fontsize=10)
    fig.suptitle("Model within-tissue phenotypic switching (operator H·rho; edge = occupancy-weighted switch mass)", fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    f = os.path.join(outdir, "switching_transition_graph.png")
    fig.savefig(f, dpi=140); plt.close(fig)
    print(f"wrote {f}")


if __name__ == "__main__":
    main()
