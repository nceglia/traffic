"""Migratory vs resident phenotype analysis (factored-model, joint-flow centered).

Everything is built from the fitted redistribution operator T (row-stochastic; T[(a,u),(b,v)] =
P(a cell in source (tissue a, phenotype u) redistributes to (tissue b, phenotype v))). When the fit
is factored, T = pi (x) Phi exactly and the readouts come straight off the factors; otherwise T is
recovered by decompose(M). Occupancy rho(a,u)=P(phenotype u | tissue a) turns conditionals into the
honest JOINT flow -- so rare source states can't masquerade as important (the base-rate fix).

The core object is the joint flow  J(a,u -> b,v) = rho(a,u) * T[(a,u),(b,v)]  (sums to 1). Every
figure is a view of it:

  joint_flow_map          the whole J: 39x39, tissue-blocked. Diagonal blocks = RESIDENT (stayed in
                          tissue), off-blocks = MIGRATORY; occupancy-weighted (source rho / dest
                          in-flow marginals framing the matrix).
  migration_flux_arrival  MIGRATORY zoom: per source phenotype, flux bar rho*go (how much migrates)
                          + P(arrive as (dest tissue, phenotype) | migrated). Block-diagonals = kept
                          identity on the move.
  residence_flux_switch   RESIDENT zoom: flux bar rho*(1-go) + within-tissue switching Phi. Diagonal
                          = persist (kept identity in place), off-diagonal = plastic.
  clonality_skewing_entropy entropy H(g) of the per-phenotype expansion factors, per tissue. Low H =
                          focal expansion (sharpens phenotypic clonality); high H = even (preserves it).
                          H normalizes g's scale away, so it is legitimately CROSS-TISSUE comparable even
                          though the raw g magnitudes are not (see readouts.py CONTRACT).
  calibration_by_phenotype (needs TRAFFIC_H5) per-phenotype stay/go calibration + AUC.

  go[a,u]      = 1 - sum_v T[(a,u),(a,v)]      = P(leave tissue | phenotype)     (= 1 - pi(stay))
  persist[a,u] = T[z,z]/(1-go)                 = P(keep phenotype | stayed)      (= Phi(stay,keep))

Env: FIT (default nb_global_putative.npz), TRAFFIC_H5 (optional, enables calibration),
     OUTDIR (default figures/analysis/migratory_resident_phenotypes), DRAWS (default 400).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traffic import io, statespace
from traffic.readouts import decompose

RNG = np.random.default_rng(0)
TL = {"PBMC": "Blood", "CSF": "CSF", "TP": "Tumor"}
TCOL = {"PBMC": "#b2182b", "CSF": "#2166ac", "TP": "#929292"}
_PHENO_COLORS = {"CD8_Activated_TEMRA": "#08306b", "CD8_Activated_TEXeff": "#41b6c4",
    "CD8_Activated_TEXterm": "#54278f", "CD8_Activated_TRM": "#1b7837", "CD8_Quiescent_Memory": "#5e8ec1",
    "CD8_Quiescent_Naive": "#a6cee3", "CD8_Quiescent_TEXprog": "#9e9ac8", "CD4_Naive": "#ffe08a",
    "CD4_Activated": "#ffd92f", "CD4_Quiescent_Exhausted": "#e377c2", "CD4_Exhausted": "#c51b8a",
    "CD4_Treg": "#fd8d3c", "CD4_Th": "#e31a1c"}
_PH_SHORT = {"CD8_Activated_TEMRA": "TEMRA", "CD8_Activated_TEXeff": "TEXeff", "CD8_Activated_TEXterm": "TEXterm",
    "CD8_Activated_TRM": "TRM", "CD8_Quiescent_Memory": "Mem", "CD8_Quiescent_Naive": "Naive",
    "CD8_Quiescent_TEXprog": "TEXprog", "CD4_Naive": "4Naive", "CD4_Activated": "4Act",
    "CD4_Quiescent_Exhausted": "4QExh", "CD4_Exhausted": "4Exh", "CD4_Treg": "Treg", "CD4_Th": "Th"}


def operator_draws(fit, ss, n_draws):
    """Per-draw operator quantities. Uses factors directly when present, else decompose(M)."""
    S, K, L = ss.S, ss.K, ss.L; ar = np.arange(L); aa = ar // K
    fac = getattr(fit, "factors", None)
    ok_fac = isinstance(fac, dict) and all(k in fac for k in ("pi", "Phi", "g"))
    nsamp = fit.samples.shape[0]
    idx = np.sort(RNG.choice(nsamp, min(n_draws, nsamp), replace=False)); N = len(idx)
    rho = np.asarray(fit.rho, float); rn = rho / rho.sum(1, keepdims=True); rglob = rho.reshape(L) / rho.sum()
    GO = np.empty((N, L)); G = np.empty((N, L)); ARR = np.zeros((L, S, K)); STAY = np.zeros((L, K)); J = np.zeros((L, L))
    for k, i in enumerate(idx):
        if ok_fac:
            piF = np.asarray(fac["pi"][i], float); PhiF = np.asarray(fac["Phi"][i], float)
            T = (piF[:, :, None] * PhiF).reshape(L, L); g = np.asarray(fac["g"][i], float)
        else:
            M = np.asarray(fit.samples[i], float); T = decompose(M)[0]; g = M.sum(1)
        Tr = T.reshape(L, S, K)
        GO[k] = 1.0 - Tr[ar, aa].sum(1); G[k] = g
        A = Tr.copy(); A[ar, aa] = 0.0                                  # migrant destinations
        ARR += A / np.maximum(A.sum((1, 2), keepdims=True), 1e-12)
        Sb = Tr[ar, aa, :]; STAY += Sb / np.maximum(Sb.sum(1, keepdims=True), 1e-12)
        J += rglob[:, None] * T
    return dict(S=S, K=K, L=L, rho=rho, rn=rn, GO=GO, G=G, N=N,
                ARR=(ARR / N).reshape(S, K, S, K), STAY=(STAY / N).reshape(S, K, K), J=J / N,
                go=GO.mean(0).reshape(S, K), used_factors=ok_fac)


def fig_joint_flow(q, tis, ph, fp):
    S, K, L, J = q["S"], q["K"], q["L"], q["J"]
    resid = float(sum(J[z, (z // K) * K:(z // K) * K + K].sum() for z in range(L)))
    src, dst = J.sum(1), J.sum(0)
    fig = plt.figure(figsize=(14.5, 13.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 22], height_ratios=[1, 22], wspace=0.015, hspace=0.015)
    axT, axL, ax = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    vmax = J.max(); im = ax.imshow(np.where(J > 0, J, np.nan), cmap="magma",
                                   norm=LogNorm(vmin=vmax * 3e-4, vmax=vmax), aspect="equal")
    ax.set_xticks(range(L)); ax.set_xticklabels([ph[z % K] for z in range(L)], rotation=90, fontsize=6.3)
    ax.set_yticks(range(L)); ax.set_yticklabels([ph[z % K] for z in range(L)], fontsize=6.3)
    for a in range(1, S):
        ax.axhline(a * K - .5, color="w", lw=2.5); ax.axvline(a * K - .5, color="w", lw=2.5)
    for a in range(S):
        ax.add_patch(Rectangle((a * K - .5, a * K - .5), K, K, fill=False, ec="#39FF14", lw=2.4))
        ax.text(a * K + K / 2 - .5, L + 2.4, TL[tis[a]], ha="center", fontsize=12, fontweight="bold", color=TCOL[tis[a]])
        ax.text(L + 1.2, a * K + K / 2 - .5, TL[tis[a]], va="center", rotation=-90, fontsize=12, fontweight="bold", color=TCOL[tis[a]])
    axL.barh(range(L), src, color=[TCOL[tis[z // K]] for z in range(L)]); axL.set_ylim(L - .5, -.5)
    axL.invert_xaxis(); axL.axis("off"); axL.set_title("source\nρ", fontsize=8)
    axT.bar(range(L), dst, color=[TCOL[tis[z // K]] for z in range(L)]); axT.set_xlim(-.5, L - .5)
    axT.axis("off"); axT.set_title("destination in-flow", fontsize=9, fontweight="bold")
    ax.set_xlabel("DESTINATION (arrive as)", fontsize=11, fontweight="bold")
    ax.set_ylabel("SOURCE (leave as)", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=[axT, ax], fraction=0.03, pad=0.02, label="joint flow  P(source→dest)  (log)")
    fig.suptitle(f"Integrated joint-flow map:  P(source→dest) = ρ·T   (green = resident blocks; "
                 f"resident {resid:.0%} / migratory {1-resid:.0%})", fontsize=12.5, fontweight="bold", y=0.95)
    fig.savefig(fp, dpi=150, bbox_inches="tight"); plt.close(fig)


def _flux_panel(fig, gs_row, flux_a, order, cols, ph, tis, a, xlabel, title):
    axb = fig.add_subplot(gs_row[0]); yy = np.arange(len(order))
    axb.barh(yy, flux_a[order], color=[cols[u] for u in order], edgecolor="k", lw=.4)
    for r, u in enumerate(order):
        axb.text(flux_a[u] + flux_a.max() * .02, r, f"{flux_a[u]:.03f}", va="center", fontsize=6.5)
    axb.set_yticks(yy); axb.set_yticklabels([ph[u] for u in order], fontsize=8)
    axb.set_ylim(len(order) - .5, -.5); axb.set_xlim(0, flux_a.max() * 1.3); axb.invert_xaxis()
    axb.set_title(title, fontsize=13, fontweight="bold", loc="right", color=TCOL[tis[a]])
    axb.set_xlabel(xlabel, fontsize=9)
    return fig.add_subplot(gs_row[1])


def fig_migration(q, tis, ph, cols, fp):
    S, K = q["S"], q["K"]; rn, go, ARR = q["rn"], q["go"], q["ARR"]
    flux = rn * go
    fig = plt.figure(figsize=(15.5, 15)); gs = fig.add_gridspec(S, 2, width_ratios=[1, 4.6], hspace=0.33, wspace=0.02)
    for a in range(S):
        others = [b for b in range(S) if b != a]; order = np.argsort(flux[a])[::-1]
        axh = _flux_panel(fig, (gs[a, 0], gs[a, 1]), flux[a], order, cols, ph, tis, a,
                          "migrant flux  ρ·go" if a == S - 1 else "", f"source = {TL[tis[a]]}")
        H = np.concatenate([ARR[a][order][:, b, :][:, order] for b in others], axis=1)   # cols in flux order
        im = axh.imshow(H, cmap="magma", aspect="auto", vmin=0, vmax=max(H.max(), 1e-9))
        axh.set_ylim(K - .5, -.5); axh.set_yticks([])
        axh.set_xticks(range(2 * K)); axh.set_xticklabels([ph[u] for u in order] * 2, rotation=90, fontsize=6)
        for bi, b in enumerate(others):
            if bi:
                axh.axvline(bi * K - .5, color="w", lw=3)
            axh.text(bi * K + K / 2 - .5, -1.4, f"→ {TL[tis[b]]}", ha="center", fontsize=11, fontweight="bold", color=TCOL[tis[b]])
            for r in range(K):
                axh.add_patch(Rectangle((bi * K + r - .5, r - .5), 1, 1, fill=False, ec="#39FF14", lw=1.1))
        if a == 0:
            fig.colorbar(im, ax=axh, fraction=0.02, pad=0.01, label="P(arrive as … | migrated)")
    fig.suptitle("Migratory events: how much each phenotype migrates (ρ·go)  +  where it goes & what it becomes "
                 "(P(arrive|migrate); block-diagonal = kept identity)", fontsize=12.5, fontweight="bold", y=0.995)
    fig.savefig(fp, dpi=145, bbox_inches="tight"); plt.close(fig)


def fig_residence(q, tis, ph, cols, fp):
    S, K = q["S"], q["K"]; rn, go, STAY = q["rn"], q["go"], q["STAY"]
    rflux = rn * (1 - go)
    fig = plt.figure(figsize=(11.5, 15)); gs = fig.add_gridspec(S, 2, width_ratios=[1, 2.3], hspace=0.34, wspace=0.02)
    for a in range(S):
        order = np.argsort(rflux[a])[::-1]
        axh = _flux_panel(fig, (gs[a, 0], gs[a, 1]), rflux[a], order, cols, ph, tis, a,
                          "resident flux  ρ·(1−go)" if a == S - 1 else "", f"resident in {TL[tis[a]]}")
        H = STAY[a][np.ix_(order, order)]                                # diagonal = persist
        im = axh.imshow(H, cmap="magma", aspect="auto", vmin=0, vmax=max(H.max(), 1e-9))
        axh.set_ylim(K - .5, -.5); axh.set_yticks([])
        axh.set_xticks(range(K)); axh.set_xticklabels([ph[u] for u in order], rotation=90, fontsize=7)
        for r in range(K):
            axh.add_patch(Rectangle((r - .5, r - .5), 1, 1, fill=False, ec="#39FF14", lw=1.3))
        axh.text(K / 2 - .5, -1.1, "becomes (in place) →", ha="center", fontsize=10, fontweight="bold")
        if a == 0:
            fig.colorbar(im, ax=axh, fraction=0.03, pad=0.01, label="P(become … | stayed)")
    fig.suptitle("Resident events: how much each phenotype stays (ρ·(1−go))  +  what it becomes in place "
                 "(within-tissue Φ; green diagonal = kept identity, off-diagonal = plastic)", fontsize=12, fontweight="bold", y=0.995)
    fig.savefig(fp, dpi=145, bbox_inches="tight"); plt.close(fig)


def fig_skew_entropy(q, tis, ph, cols, fp):
    """Clonality-skewing rate: entropy H of the normalized per-phenotype expansion factors, per tissue.

    p_a(u) = g_a(u)/Σ_u g_a(u);  H_a = -Σ p log p  (nats),  normalized by log K.  Low H = focal
    expansion (sharpens clonality); high H = even. H is scale-invariant -> cross-tissue comparable.
    """
    S, K, G = q["S"], q["K"], q["G"]; N = G.shape[0]
    g = G.reshape(N, S, K); p = g / g.sum(2, keepdims=True)
    H = -(p * np.log(np.maximum(p, 1e-12))).sum(2); Hn = H / np.log(K)   # [N,S]
    Hn_m, lo, hi = Hn.mean(0), np.quantile(Hn, .025, 0), np.quantile(Hn, .975, 0)
    pm = p.mean(0)                                                       # [S,K] mean expansion share
    order = np.argsort(Hn_m)                                             # skewed (low H) -> even (high H)
    pw = []
    for a in range(S):
        for b in range(a + 1, S):
            d = H[:, a] - H[:, b]; pw.append(f"P({TL[tis[a]]}>{TL[tis[b]]})={(d > 0).mean():.0%}")
    fig = plt.figure(figsize=(14.5, 5.4)); gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25], wspace=0.24)
    axA, axB = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    for r, a in enumerate(order):
        c = TCOL[tis[a]]
        axA.plot([lo[a], hi[a]], [r, r], "-", color=c, lw=3.5, solid_capstyle="round", zorder=2)
        axA.plot(Hn_m[a], r, "o", ms=13, color=c, mec="k", zorder=3)
        axA.text(hi[a] + .012, r, f"{Hn_m[a]:.2f}", va="center", fontsize=10, fontweight="bold")
        du = pm[a].argmax(); axA.text(0.02, r + 0.30, f"skew onto {ph[du]} ({pm[a, du]:.0%})", fontsize=8.5, color="0.35")
    axA.set_yticks(range(S)); axA.set_yticklabels([TL[tis[a]] for a in order], fontsize=12, fontweight="bold")
    axA.set_ylim(-.5, S - .5); axA.set_xlim(0, 1)
    axA.set_xlabel("normalized expansion entropy H(g)\n← skewed (sharpens clonality)        even (preserves) →", fontsize=10)
    axA.set_title("clonality-skewing rate per tissue  (scale-invariant → cross-tissue valid)\n" + "   ".join(pw), fontsize=9.5, fontweight="bold")
    im = axB.imshow(pm, cmap="magma", aspect="auto", vmin=0, vmax=pm.max())
    axB.set_yticks(range(S)); axB.set_yticklabels([TL[tis[a]] for a in range(S)], fontsize=11, fontweight="bold")
    axB.set_xticks(range(K)); axB.set_xticklabels(ph, rotation=90, fontsize=8)
    axB.set_title("expansion share  p = g / Σg   (concentrated = skewed = low H)", fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=axB, fraction=0.05, pad=0.02, label="share of tissue expansion")
    fig.suptitle("Expansion skew: entropy of the per-phenotype expansion factors, per tissue "
                 "(low H = focal expansion sharpens clonality; scale-invariant → cross-tissue comparable)",
                 fontsize=11, fontweight="bold", y=1.03)
    fig.tight_layout(); fig.savefig(fp, dpi=145, bbox_inches="tight"); plt.close(fig)


def fig_calibration(fit, ss, ph, cols, h5, fp):
    from migration_validation import auc, cboot_ci
    import migration_validation as _mv
    from traffic import data
    S, K = ss.S, ss.K
    obs = data.build(data.load_obs_table(h5), ss); _mv._PAT = np.asarray(obs.patient)
    Xt, Yb, Db = obs.Xtilde, obs.Y, obs.D; Jn = Xt.shape[0]; obsv = Db > 0; Yt = np.where(obsv, Yb, 0.0)
    mean_model = np.where(obsv, Db * (Xt @ fit.M_hat), 0.0)
    src_tis = obs.src_tissue; src_phe = obs.X.reshape(Jn, S, K)[np.arange(Jn), src_tis, :].argmax(1)
    obs_tm = Yt.reshape(Jn, S, K).sum(2); prd_tm = mean_model.reshape(Jn, S, K).sum(2)
    surv = obs_tm.sum(1) > 0; pf = prd_tm / np.maximum(prd_tm.sum(1, keepdims=True), 1e-12)
    pleave = 1 - pf[np.arange(Jn), src_tis]; leave_lbl = (obs_tm[np.arange(Jn), src_tis] < obs_tm.sum(1) - 1e-9).astype(int)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    for u in range(K):
        sel = surv & (src_phe == u)
        if sel.sum() < 20:
            continue
        ax[0].scatter(float(pleave[sel].mean()), float(leave_lbl[sel].mean()), s=np.clip(sel.sum() / 20, 12, 240),
                      color=cols[u], edgecolor="k", lw=.5, zorder=3)
        ax[0].annotate(ph[u], (float(pleave[sel].mean()), float(leave_lbl[sel].mean())), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    ax[0].plot([0, 1], [0, 1], "k--", lw=1); ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1)
    ax[0].set_xlabel("model predicted P(leave)"); ax[0].set_ylabel("observed leave rate")
    ax[0].set_title("A. Per-phenotype stay/go calibration (size = n; on diagonal = calibrated)")
    us, aus, los, his, ns = [], [], [], [], []
    for u in range(K):
        ii = np.where(surv & (src_phe == u))[0]
        if ii.size < 30 or leave_lbl[ii].sum() in (0, ii.size):
            continue
        a_ = auc(pleave[ii], leave_lbl[ii]); lo, hi = cboot_ci(lambda idx: auc(pleave[idx], leave_lbl[idx]), ii)
        us.append(u); aus.append(a_); los.append(max(0, a_ - lo) if np.isfinite(lo) else 0)
        his.append(max(0, hi - a_) if np.isfinite(hi) else 0); ns.append(ii.size)
    xu = np.arange(len(us))
    ax[1].bar(xu, aus, color=[cols[u] for u in us], edgecolor="k", lw=.4, yerr=[los, his], capsize=3)
    ax[1].axhline(0.5, color="k", ls=":"); ax[1].set_ylim(0.4, 1.0); ax[1].set_xticks(xu)
    ax[1].set_xticklabels([f"{ph[u]}\n(n={n})" for u, n in zip(us, ns)], fontsize=7, rotation=90)
    ax[1].set_ylabel("stay/go AUC"); ax[1].set_title("B. Per-phenotype leave/stay discrimination (patient-cluster 95% CI)")
    fig.suptitle("Is the residence/migration call CALIBRATED for each phenotype? (trust the call where it is)", y=1.0)
    fig.tight_layout(); fig.savefig(fp, dpi=140, bbox_inches="tight"); plt.close(fig)


def main():
    ss = statespace.default()
    fit_path = os.environ.get("FIT", "nb_global_putative.npz")
    outdir = os.environ.get("OUTDIR", "figures/analysis/migratory_resident_phenotypes")
    n_draws = int(os.environ.get("DRAWS", "400"))
    os.makedirs(outdir, exist_ok=True)
    fit = io.load_fit(fit_path)
    phenos = [str(p) for p in (getattr(fit, "phenotypes", None) if getattr(fit, "phenotypes", None) is not None else ss.phenotypes)]
    ph = [_PH_SHORT.get(p, p) for p in phenos]; cols = [_PHENO_COLORS.get(p, "#999") for p in phenos]
    tis = [str(t) for t in ss.tissues]
    q = operator_draws(fit, ss, n_draws)
    print(f"fit={fit_path}  states={fit.M_hat.shape[0]}  draws={q['N']}  factors={'yes' if q['used_factors'] else 'no (decompose)'}")

    written = []
    for name, fn in [("joint_flow_map", fig_joint_flow), ("migration_flux_arrival", fig_migration),
                     ("residence_flux_switch", fig_residence), ("clonality_skewing_entropy", fig_skew_entropy)]:
        fp = os.path.join(outdir, name + ".png")
        (fn(q, tis, ph, fp) if fn is fig_joint_flow else fn(q, tis, ph, cols, fp))
        written.append(fp)

    h5 = os.environ.get("TRAFFIC_H5")
    if h5:
        fp = os.path.join(outdir, "calibration_by_phenotype.png")
        fig_calibration(fit, ss, ph, cols, h5, fp); written.append(fp)
    else:
        print("  (set TRAFFIC_H5 to also produce calibration_by_phenotype)")

    print("wrote:")
    for w in written:
        print(" ", w)


if __name__ == "__main__":
    main()
