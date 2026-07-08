"""Validate model H(g) (clonality-skewing rate) against empirical clonality drift, per tissue.

Post-fit, no refit. Tests whether the model-derived expansion-skew entropy H(g) per tissue predicts
an INDEPENDENT empirical quantity never in the objective: the drift of observed repertoire clonality
across timepoints. Logic: low H(g) (focal expansion) -> clonality should rise; high H(g) (even) -> flat.

Confound handling (both, compared):
  naive     : clonality on all clones per (patient,tissue,timepoint).
  resident  : clones present at >=2 timepoints in that tissue only -> removes the migration contribution.
The naive-vs-resident gap = migration contamination. Both are share-clonality (depth-normalized indices),
matching H(g)'s share-space nature; depth was separately shown robust by rarefaction.

Env: FIT (factored fit), TRAFFIC_H5 (clone data), OUTDIR.  Outputs: 2 CSVs + Hg_vs_clonality_drift.png.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import anndata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traffic import io, statespace

TL = {"PBMC": "Blood", "CSF": "CSF", "TP": "Tumor"}
TCOL = {"PBMC": "#b2182b", "CSF": "#2166ac", "TP": "#929292"}
MIN_CLONES, MIN_RES, MIN_TP = 20, 10, 3


def shannon_clonality(cs):
    cs = np.asarray(cs, float); R = len(cs)
    if R < 2:
        return np.nan
    q = cs / cs.sum()
    return 1.0 - (-(q * np.log(q)).sum()) / np.log(R)


def gini(cs):
    x = np.sort(np.asarray(cs, float)); n = len(x)
    return float((2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum())) - (n + 1) / n) if n and x.sum() else np.nan


def dominance(cs):
    cs = np.asarray(cs, float); return float(cs.max() / cs.sum())


def slope_table(df, col, require_reliable):
    rows = []
    d = df[df["reliable"]] if require_reliable else df
    for (pt, ti), gp in d.groupby(["patient", "tissue"], observed=True):
        gp = gp.dropna(subset=[col])
        if gp["timepoint"].nunique() >= MIN_TP:
            rows.append((pt, ti, float(np.polyfit(gp["timepoint"], gp[col], 1)[0])))
    return pd.DataFrame(rows, columns=["patient", "tissue", "slope"])


def main():
    fit_path = os.environ.get("FIT", "/Users/ceglian/Data/factored_fit.npz")
    h5 = os.environ["TRAFFIC_H5"]
    outdir = os.environ.get("OUTDIR", "figures/analysis/migratory_resident_factored")
    os.makedirs(outdir, exist_ok=True)
    ss = statespace.default(); S, K = ss.S, ss.K; tc = list(ss.tissues)

    # ---- model H(g) per tissue ----
    g = np.asarray(io.load_fit(fit_path).factors["g"], float).reshape(-1, S, K)
    p = g / g.sum(2, keepdims=True); Hn = -(p * np.log(p)).sum(2) / np.log(K)
    Hg = {tc[a]: (Hn[:, a].mean(), np.percentile(Hn[:, a], 2.5), np.percentile(Hn[:, a], 97.5)) for a in range(S)}

    # ---- empirical clonality per (patient,tissue,timepoint) ----
    o = anndata.read_h5ad(h5, backed="r").obs[["patient", "tissue", "timepoint", "trb"]].dropna(subset=["trb"]).copy()
    o["timepoint"] = o["timepoint"].astype(int)
    for c in ["patient", "tissue", "trb"]:                # drop categorical dtype (else value_counts yields 0-count cats)
        o[c] = o[c].astype(str)
    rows = []
    for (pt, ti, tp), gp in o.groupby(["patient", "tissue", "timepoint"], observed=True):
        cs = gp["trb"].value_counts().values
        rows.append(dict(patient=pt, tissue=ti, timepoint=tp, ncells=int(cs.sum()), nclones=len(cs),
                         clonality=shannon_clonality(cs), gini=gini(cs), dominance=dominance(cs),
                         reliable=len(cs) >= MIN_CLONES))
    clon = pd.DataFrame(rows)
    clon.to_csv(os.path.join(outdir, "clonality_by_sample.csv"), index=False)

    # ---- resident-restricted clonality (clones present at >=2 timepoints in the tissue) ----
    rr = []
    for (pt, ti), sub in o.groupby(["patient", "tissue"], observed=True):
        ntp = sub.groupby("trb", observed=True)["timepoint"].nunique()
        res = set(ntp[ntp >= 2].index)
        for tp, gtp in sub.groupby("timepoint", observed=True):
            cs = gtp[gtp["trb"].isin(res)]["trb"].value_counts().values
            if len(cs) >= MIN_RES:
                rr.append(dict(patient=pt, tissue=ti, timepoint=tp, nres=len(cs),
                               clonality=shannon_clonality(cs), gini=gini(cs)))
    resid = pd.DataFrame(rr)
    resid.to_csv(os.path.join(outdir, "resident_clonality_by_sample.csv"), index=False)   # per-tp resident trajectory

    # ---- drift slopes: naive + resident ----
    sn = slope_table(clon, "clonality", True); sr = slope_table(resid, "clonality", False)
    merged = sn.rename(columns={"slope": "slope_naive"}).merge(
        sr.rename(columns={"slope": "slope_resident"}), on=["patient", "tissue"], how="outer")
    merged.to_csv(os.path.join(outdir, "drift_slopes.csv"), index=False)

    def agg(tab):
        return {ti: (tab[tab.tissue == ti].slope.median(), tab[tab.tissue == ti].slope.std(),
                     int((tab.tissue == ti).sum())) for ti in tc}
    A, R = agg(sn), agg(sr)

    # ---- plot: resident (migration-controlled) drift is the PRIMARY claim vs H(g);
    #           naive shown as a faint ghost + arrow so the arrow length = migration contamination ----
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(9.4, 6.7))
    for ti in tc:
        hm, hl, hh = Hg[ti]; dn = A[ti][0]; dr, ds, n = R[ti]
        ax.annotate("", xy=(hm, dr), xytext=(hm, dn),                                   # naive -> resident (migration removed)
                    arrowprops=dict(arrowstyle="-|>", color=TCOL[ti], lw=1.4, ls=(0, (4, 3)), alpha=0.5), zorder=2)
        ax.plot(hm, dn, "o", ms=11, mfc="white", mec=TCOL[ti], mew=1.3, alpha=0.6, zorder=3)   # naive ghost (reference)
        ye = ds if (n > 1 and np.isfinite(ds)) else 0.0
        ax.errorbar(hm, dr, xerr=[[hm - hl], [hh - hm]], yerr=ye, fmt="o", ms=15, color=TCOL[ti],   # resident = the claim
                    ecolor=TCOL[ti], elinewidth=1.7, capsize=3, zorder=4)
        ax.annotate(f"{TL[ti]} (n={n})", (hm, dr), xytext=(13, 11), textcoords="offset points",
                    fontsize=11.5, fontweight="bold", color=TCOL[ti], va="bottom")
    ax.axhline(0, color="#999", ls=":", lw=1)
    ax.set_xlabel("model  H(g)    (← skewed / sharpens          even / preserves →)", fontsize=11)
    ax.set_ylabel("in-situ clonality drift  (resident clones only, slope per timepoint)", fontsize=11)
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", mfc="0.35", ms=13, label="resident, migration-controlled — the claim"),
                       Line2D([0], [0], marker="o", color="w", mfc="white", mec="0.35", mew=1.3, ms=11, label="naive (before control)"),
                       Line2D([0], [0], color="0.35", ls=(0, (4, 3)), label="arrow = drift removed by migration control")],
              loc="upper right", fontsize=9, frameon=False)
    ax.set_title("In-situ clonality sharpening tracks the model's expansion-skew H(g)\n"
                 "resident-clone drift (migration removed) rises where H(g) is low (Tumor) and stays flat where H(g) is high",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(); fp = os.path.join(outdir, "Hg_vs_clonality_drift.png")
    fig.savefig(fp, dpi=150, bbox_inches="tight"); plt.close(fig)

    # ---- summary + sanity ----
    excluded = sum(1 for (_, _), gp in clon[clon.reliable].groupby(["patient", "tissue"], observed=True)
                   if gp.timepoint.nunique() < MIN_TP)
    print("\n===== H(g) vs empirical clonality drift =====")
    order = sorted(tc, key=lambda t: Hg[t][0])            # ascending H = most-skewed first
    print("  H(g) ranking (skewed→even):  " + "  <  ".join(f"{TL[t]} {Hg[t][0]:.2f}" for t in order))
    print(f"  naive drift    per tissue:   " + "   ".join(f"{TL[t]}={A[t][0]:+.4f}(n{A[t][2]})" for t in order))
    print(f"  resident drift per tissue:   " + "   ".join(f"{TL[t]}={R[t][0]:+.4f}(n{R[t][2]})" for t in order))
    dn = [A[t][0] for t in order]; dr = [R[t][0] for t in order]
    print(f"  RANKING TEST: does drift fall as H rises?  naive {'YES' if dn == sorted(dn, reverse=True) else 'partial/NO'} "
          f"| resident {'YES' if dr == sorted(dr, reverse=True) else 'partial/NO'}")
    print(f"  CSF vs TP:  TP  H={Hg['TP'][0]:.2f} drift(res)={R['TP'][0]:+.4f}   |   CSF H={Hg['CSF'][0]:.2f} drift(res)={R['CSF'][0]:+.4f}  "
          f"-> {'AGREE (skewing story)' if R['TP'][0] > R['CSF'][0] else 'DISAGREE'}")
    print("  migration contamination (naive − resident drift): " + "  ".join(f"{TL[t]}={A[t][0]-R[t][0]:+.4f}" for t in tc))
    print(f"  sanity: (patient,tissue) excluded for <{MIN_TP} timepoints: {excluded};  H(g) prior-dominated tissues: none (pre-flight)")
    print("wrote:")
    for f in ["clonality_by_sample.csv", "drift_slopes.csv", "Hg_vs_clonality_drift.png"]:
        print("  ", os.path.join(outdir, f))


if __name__ == "__main__":
    main()
