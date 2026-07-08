"""2x2 validation figure from validate_clonality_drift.py outputs (reads CSVs + H(g) posterior; no recompute).

Arc:  A model claim -> B empirical confirmation -> C detected signal (Tumor hero) -> D confounds removed.
Substitutions (data availability): drift is Shannon-clonality/step (Gini is in the CSV too but slopes were
saved as Shannon); Panel C is resident-restricted at OBSERVED depth (depth-robustness was shown separately
by the rarefaction sweep, TP rising at N=100/150/200).

Env: FIT (factored fit), OUTDIR (dir holding the three CSVs; figure written here).
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traffic import io, statespace

TL = {"PBMC": "Blood", "CSF": "CSF", "TP": "Tumor"}
TCOL = {"PBMC": "#b2182b", "CSF": "#2166ac", "TP": "#929292"}
D = os.environ.get("OUTDIR", "figures/analysis/migratory_resident_factored")

# ---- H(g) posterior per tissue ----
ss = statespace.default(); S, K = ss.S, ss.K; tc = list(ss.tissues)
g = np.asarray(io.load_fit(os.environ.get("FIT", "/Users/ceglian/Data/factored_fit.npz")).factors["g"], float).reshape(-1, S, K)
p = g / g.sum(2, keepdims=True); Hn = -(p * np.log(p)).sum(2) / np.log(K)
Hg = {tc[a]: (Hn[:, a].mean(), np.percentile(Hn[:, a], 2.5), np.percentile(Hn[:, a], 97.5)) for a in range(S)}

# ---- CSVs (missing quantity check) ----
need = ["clonality_by_sample.csv", "resident_clonality_by_sample.csv", "drift_slopes.csv"]
missing = [f for f in need if not os.path.exists(os.path.join(D, f))]
if missing:
    print("MISSING CSVs (re-run validate_clonality_drift.py):", missing); sys.exit(1)
res = pd.read_csv(os.path.join(D, "resident_clonality_by_sample.csv"))
dr = pd.read_csv(os.path.join(D, "drift_slopes.csv"))
order = sorted(tc, key=lambda t: Hg[t][0])                    # H ascending: TP (skewed) -> CSF (even)

fig, ax = plt.subplots(2, 2, figsize=(14.5, 11.5))

# ===== A: model claim -- H(g) per tissue =====
a = ax[0, 0]
for i, ti in enumerate(order):
    hm, hl, hh = Hg[ti]
    a.errorbar(i, hm, yerr=[[hm - hl], [hh - hm]], fmt="o", ms=15, color=TCOL[ti], ecolor=TCOL[ti], capsize=5, elinewidth=1.9)
    a.annotate(f"{hm:.2f}", (i, hh), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=10, fontweight="bold", color=TCOL[ti])
a.set_xlim(-.5, 2.5); a.set_xticks(range(3)); a.set_xticklabels([TL[t] for t in order], fontsize=12, fontweight="bold")
a.set_ylabel("clonality-skewing rate  H(g)", fontsize=11)
a.set_title("A · Model claim — expansion-skew per tissue", fontsize=12, fontweight="bold", loc="left")
a.text(.98, .04, "low H = concentrated expansion → sharpens repertoire", transform=a.transAxes, fontsize=8.5, color="#555", ha="right")
a.text(.98, .96, "high H = even expansion → preserves", transform=a.transAxes, fontsize=8.5, color="#555", ha="right", va="top")

# ===== B: empirical confirmation -- resident drift per tissue =====
b = ax[0, 1]
for i, ti in enumerate(order):
    sl = dr[dr.tissue == ti].slope_resident.dropna(); m = sl.median(); n = len(sl)
    b.errorbar(i, m, yerr=[[m - sl.min()], [sl.max() - m]] if n > 1 else 0, fmt="o", ms=15,
               color=TCOL[ti], ecolor=TCOL[ti], capsize=5, elinewidth=1.9)
    b.annotate(f"n={n}" + ("  ⚠ underpowered" if ti == "CSF" else ""), (i, m), xytext=(10, 0),
               textcoords="offset points", fontsize=8.5, va="center", color=TCOL[ti])
b.axhline(0, color="#999", ls=":", lw=1); b.set_xlim(-.5, 2.5); b.set_xticks(range(3))
b.set_xticklabels([TL[t] for t in order], fontsize=12, fontweight="bold")
b.set_ylabel("resident clonality drift  (Shannon / timepoint)", fontsize=11)
b.set_title("B · Empirical confirmation — migration-controlled drift", fontsize=12, fontweight="bold", loc="left")
b.text(.02, .96, "prediction holds: the low-H(g) tissue (Tumor) shows the steepest rise", transform=b.transAxes, fontsize=8.5, color="#555", va="top")

# ===== C: the detected signal -- Tumor resident trajectory =====
c = ax[1, 0]
tp = res[res.tissue == "TP"]
for pt, gp in tp.groupby("patient"):
    gp = gp.sort_values("timepoint"); c.plot(gp.timepoint, gp.clonality, "-o", color=TCOL["TP"], alpha=.35, lw=1.2, ms=5)
mt = tp.groupby("timepoint").clonality.mean()
c.plot(mt.index, mt.values, "-o", color="#000", lw=3.2, ms=10, zorder=5, label="mean")
c.set_xlabel("timepoint", fontsize=11); c.set_ylabel("resident clonality  (Shannon)", fontsize=11)
c.set_title("C · Detected signal — Tumor resident clonality climbs", fontsize=12, fontweight="bold", loc="left")
c.text(.02, .96, "resident-restricted (migration removed) · depth-robust by rarefaction sweep (TP rising at N=100/150/200)",
       transform=c.transAxes, fontsize=8.5, color="#555", va="top")
c.legend(loc="lower right", fontsize=9, frameon=False)

# ===== D: confounds removed -- naive vs resident dumbbell (vertical, same orientation/order as A,B) =====
d = ax[1, 1]
for i, ti in enumerate(order):
    nv = dr[dr.tissue == ti].slope_naive.median(); rs = dr[dr.tissue == ti].slope_resident.median()
    d.plot([i, i], [nv, rs], "-", color=TCOL[ti], lw=2.2, alpha=.5, zorder=1)
    d.plot(i, nv, "o", ms=12, mfc="white", mec=TCOL[ti], mew=1.6, zorder=3)          # naive (before control)
    d.plot(i, rs, "o", ms=15, color=TCOL[ti], zorder=4)                              # resident (the claim)
    flip = "\n(sign-flip!)" if (nv > 0) != (rs > 0) else ""
    d.annotate(f"Δ={nv - rs:+.3f}{flip}", (i, max(nv, rs)), xytext=(0, 9), textcoords="offset points",
               ha="center", fontsize=8.5, color=TCOL[ti], fontweight=("bold" if flip else "normal"))
d.axhline(0, color="#999", ls=":", lw=1); d.set_xlim(-.5, 2.5); d.set_xticks(range(3))
d.set_xticklabels([TL[t] for t in order], fontsize=12, fontweight="bold")
d.set_ylabel("clonality drift  (Shannon / timepoint)", fontsize=11)
d.set_title("D · Confounds removed — naive ○ vs resident ● (gap = migration)", fontsize=12, fontweight="bold", loc="left")
d.legend(handles=[Line2D([0], [0], marker="o", color="w", mfc="white", mec="0.4", mew=1.5, ms=11, label="naive (all clones)"),
                  Line2D([0], [0], marker="o", color="w", mfc="0.4", ms=13, label="resident (migration-controlled)")],
         loc="lower right", fontsize=9, frameon=False)

fig.suptitle("Model expansion-skew H(g) predicts empirical clonality drift  —  claim → confirmation → signal → confound control",
             fontsize=13.5, fontweight="bold", y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fp = os.path.join(D, "clonality_validation_2x2.png")
fig.savefig(fp, dpi=150, bbox_inches="tight"); print("wrote", fp)
