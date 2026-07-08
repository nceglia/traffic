"""Phenotypic-switching validation inventory (batch 1: full fit + posterior draws).

Validates the operator's WITHIN-TISSUE phenotype redistribution -- do a clone's
descendants that STAY in a tissue change phenotype as the model predicts -- from every
angle the full fit supports. Switching is a COMPOSITION quantity, so it is fully
share-supported (see traffic/readouts.py CONTRACT): this axis is on solid ground, unlike
absolute expansion. One standalone PNG per check to figures/validation/switching/.

On top of grouped statistics, several checks are explicitly Bayesian and per-CLONE:
  clone_posterior_predictive : sample individual clones, show the full posterior-predictive
                               within-tissue phenotype distribution vs what they actually did.
  clone_ppc_coverage         : per-clone posterior-predictive coverage of the observed
                               dominant destination phenotype, by clone size (calibration).

Holdout influence + out-of-sample switching skill is batch 2 (separate script).

Env: TRAFFIC_H5 (required), FIT (default nb_global_putative.npz),
     OUTDIR (default figures/validation/switching), PPC_DRAWS (default 400).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import migration_validation as _mv  # noqa: E402
from migration_validation import boot_ci, cboot_ci, wilson, auc  # noqa: E402
from predictability_breakdown import decompose_prediction, _js_rows  # noqa: E402

from traffic import data, io, statespace  # noqa: E402
from traffic.readouts import _js, decompose, within_tissue_switching  # noqa: E402

RNG = np.random.default_rng(0)

_PH_SHORT = {
    "CD8_Activated_TEMRA": "TEMRA", "CD8_Activated_TEXeff": "TEXeff",
    "CD8_Activated_TEXterm": "TEXterm", "CD8_Activated_TRM": "TRM",
    "CD8_Quiescent_Memory": "Mem", "CD8_Quiescent_Naive": "Naive",
    "CD8_Quiescent_TEXprog": "TEXprog", "CD4_Naive": "4Naive",
    "CD4_Activated": "4Act", "CD4_Quiescent_Exhausted": "4QExh",
    "CD4_Exhausted": "4Exh", "CD4_Treg": "Treg", "CD4_Th": "Th",
}

# curated biologically-directed transitions (source_idx, dest_idx, label); indices match
# ss.phenotypes order: 0 TEMRA 1 TEXeff 2 TEXterm 3 TRM 4 Mem 5 Naive 6 TEXprog
#                      7 4Naive 8 4Act 9 4QExh 10 4Exh 11 Treg 12 Th
_CURATED = [
    (5, 4, "Naive->Mem"), (5, 6, "Naive->TEXprog"), (6, 1, "TEXprog->TEXeff"),
    (6, 2, "TEXprog->TEXterm"), (1, 2, "TEXeff->TEXterm"), (4, 0, "Mem->TEMRA"),
    (0, 1, "TEMRA->TEXeff"), (3, 2, "TRM->TEXterm"),
    (7, 8, "4Naive->4Act"), (8, 9, "4Act->4QExh"), (8, 10, "4Act->4Exh"),
    (9, 10, "4QExh->4Exh"), (8, 11, "4Act->Treg"),
]
# progression-toward-exhaustion edges for the tissue-contrast score
_PROG = [(6, 1), (6, 2), (1, 2), (3, 2), (8, 9), (8, 10), (9, 10)]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h5 = os.environ.get("TRAFFIC_H5")
    if not h5:
        raise SystemExit("set TRAFFIC_H5")
    fit_path = os.environ.get("FIT", "nb_global_putative.npz")
    outdir = os.environ.get("OUTDIR", "figures/validation/switching")
    n_ppc = int(os.environ.get("PPC_DRAWS", "400"))
    os.makedirs(outdir, exist_ok=True)

    ss = statespace.default()
    obs = data.build(data.load_obs_table(h5), ss)
    _mv._PAT = np.asarray(obs.patient)                            # honest sampling unit: patient-cluster CIs
    fit = io.load_fit(fit_path)
    S, K, L = ss.S, ss.K, ss.L
    tis = list(ss.tissues)
    ph = [_PH_SHORT.get(str(p), str(p)) for p in ss.phenotypes]
    Xt, Y, D = obs.Xtilde, obs.Y, obs.D
    J = Xt.shape[0]
    phi = float(np.exp(np.asarray(fit.dispersion["params"]["log_r"]).mean())) if fit.dispersion else np.inf
    obsv = D > 0
    Yt = np.where(obsv, Y, 0.0)
    pi = Yt.sum(0); pi = pi / max(pi.sum(), 1e-12)
    print(f"fit={fit_path} J={J} phi={phi:.3f} K={K} outdir={outdir}")

    mean_model = np.where(obsv, D * (Xt @ fit.M_hat), 0.0)
    mean_static = np.where(obsv, D * Xt, 0.0)
    mean_pooled = np.where(obsv, D * (Xt.sum(1, keepdims=True) * pi[None, :]), 0.0)
    m = decompose_prediction(Xt, Y, D, mean_model, ss, phi)
    p = decompose_prediction(Xt, Y, D, mean_static, ss, phi)
    c = decompose_prediction(Xt, Y, D, mean_pooled, ss, phi)

    # ---- within-(source)-tissue phenotype composition, per clone ------------------
    src_tis = obs.src_tissue                 # dominant source tissue (raw-count; see docs/DATA.md)
    aj = np.arange(J)

    def block(arr):
        """Each clone's source-tissue K-phenotype slice, [J,K]."""
        return arr.reshape(J, S, K)[aj, src_tis, :]

    src_cnt = block(obs.X)                                         # raw source phenotype counts
    obs_ph = block(Yt)                                            # observed dest phenotype counts (stayed in tissue)
    prd_ph = block(mean_model)                                    # predicted
    nrm = lambda A: A / np.maximum(A.sum(1, keepdims=True), 1e-12)
    src_n, obs_n, prd_n = nrm(src_cnt), nrm(obs_ph), nrm(prd_ph)
    surv_sw = obs_ph.sum(1) > 0                                    # descendants stayed in source tissue
    src_phe = src_cnt.argmax(1)                                    # dominant source phenotype
    n_src = obs.n_src.astype(int)
    Xr = obs.X.reshape(J, S, K).sum(0); rho = Xr / np.maximum(Xr.sum(1, keepdims=True), 1e-12)

    swi_ch = np.full(J, np.nan)
    if surv_sw.any():
        swi_ch[surv_sw] = _js_rows(src_n[surv_sw], obs_n[surv_sw]) # did phenotype actually change
    mover = surv_sw & (swi_ch > 0.05)                             # switchers (skill only meaningful here)

    grp = np.where(n_src == 1, "singleton", np.where(n_src < 50, "small", "large"))
    ABL = ["1", "2", "3-4", "5-9", "10-49", "50+"]
    abin_of = lambda n: "1" if n <= 1 else "2" if n <= 2 else "3-4" if n <= 4 else "5-9" if n <= 9 else "10-49" if n <= 49 else "50+"
    abin = np.array([abin_of(n) for n in n_src])

    # posterior draw subset (shared by every Bayesian panel)
    idx = np.sort(RNG.choice(fit.samples.shape[0], min(n_ppc, fit.samples.shape[0]), replace=False))

    T_hat, _ = decompose(fit.M_hat)
    H = within_tissue_switching(T_hat, ss)                         # [S,K,K] model operator
    # posterior draws of the switching operator (shared by CrI panels) + empirical operator
    Hdraws = np.empty((len(idx), S, K, K))
    for s_, si in enumerate(idx):
        Hdraws[s_] = within_tissue_switching(decompose(fit.samples[si])[0], ss)
    Hobs_all = np.full((S, K, K), np.nan)
    for a in range(S):
        for u in range(K):
            sel = surv_sw & (src_tis == a) & (src_phe == u)
            if sel.any():
                v = obs_ph[sel].sum(0); Hobs_all[a, u] = v / max(v.sum(), 1e-12)
    written = []

    # ===================== FIG 1: within-tissue switching operator ============= #
    # empirical H_obs[a][u,v] = dest phenotype composition of clones whose dominant source
    # phenotype is u in tissue a (aggregated), vs the model operator H[a].
    fig, ax = plt.subplots(S, 3, figsize=(16, 4.6 * S))
    for a in range(S):
        Hobs = np.full((K, K), np.nan); nu = np.zeros(K, int)
        for u in range(K):
            sel = surv_sw & (src_tis == a) & (src_phe == u)
            nu[u] = int(sel.sum())
            if sel.any():
                v = obs_ph[sel].sum(0); Hobs[u] = v / max(v.sum(), 1e-12)
        for k_, (M_, ttl, cmap, vlim) in enumerate([
                (Hobs, "observed", "viridis", (0, 1)),
                (H[a], "model", "viridis", (0, 1)),
                (H[a] - Hobs, "model - observed", "RdBu_r", (-0.5, 0.5))]):
            im = ax[a, k_].imshow(M_, cmap=cmap, vmin=vlim[0], vmax=vlim[1], aspect="auto")
            ax[a, k_].set_xticks(range(K)); ax[a, k_].set_xticklabels(ph, rotation=90, fontsize=6)
            ax[a, k_].set_yticks(range(K)); ax[a, k_].set_yticklabels(ph, fontsize=6)
            ax[a, k_].set_title(f"{tis[a]}: {ttl}", fontsize=10)
            fig.colorbar(im, ax=ax[a, k_], fraction=0.046)
        ax[a, 0].set_ylabel("source phenotype")
    fig.suptitle("Within-tissue phenotype switching operator (row-stochastic; diagonal = phenotype persistence)", y=1.0)
    fig.tight_layout(); f = os.path.join(outdir, "switching_matrices.png"); fig.savefig(f, dpi=140, bbox_inches="tight"); plt.close(fig)
    written.append(f)

    # ===================== FIG 2: phenotype persistence + posterior CrI ========= #
    # per (tissue, phenotype): P(keep phenotype | stay in tissue) = H[a][u,u], with 95% CrI.
    Pdiag = np.diagonal(Hdraws, axis1=2, axis2=3)                 # [draws,S,K] persistence per draw
    fig, ax = plt.subplots(S, 1, figsize=(11, 3.2 * S), sharex=True)
    for a in range(S):
        md = Pdiag[:, a, :].mean(0); lo = np.quantile(Pdiag[:, a, :], .025, 0); hi = np.quantile(Pdiag[:, a, :], .975, 0)
        nu = np.array([int((surv_sw & (src_tis == a) & (src_phe == u)).sum()) for u in range(K)])
        col = ["#55a868" if nn >= 20 else "#c8c8c8" for nn in nu]
        ax[a].bar(range(K), md, color=col, yerr=[md - lo, hi - md], capsize=2)
        ax[a].set_ylim(0, 1.02); ax[a].set_ylabel("P(persist)")
        ax[a].set_title(f"{tis[a]}  --  phenotype stickiness (grey = <20 source clones)", fontsize=10)
        for u in range(K):
            ax[a].text(u, .02, f"n={nu[u]}", ha="center", va="bottom", fontsize=6, rotation=90)
    ax[-1].set_xticks(range(K)); ax[-1].set_xticklabels(ph, rotation=90, fontsize=8)
    fig.suptitle("Phenotype persistence with 95% posterior credible intervals (1 = never switches)", y=1.0)
    fig.tight_layout(); f = os.path.join(outdir, "persistence_by_phenotype.png"); fig.savefig(f, dpi=140, bbox_inches="tight"); plt.close(fig)
    written.append(f)

    # ===================== FIG 3: switching skill by clone size ================ #
    def skill_ci(null_err, base):
        b = np.where(base & np.isfinite(m["switching"]) & np.isfinite(null_err))[0]
        if b.size < 10:
            return np.nan, np.nan, np.nan
        pt = 1 - np.median(m["switching"][b]) / max(np.median(null_err[b]), 1e-9)
        lo, hi = cboot_ci(lambda idx: 1 - np.median(m["switching"][idx]) / max(np.median(null_err[idx]), 1e-9), b)
        return pt, lo, hi
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.arange(len(ABL))
    pts, los, his = [], [], []
    for lb in ABL:
        s, lo, hi = skill_ci(c["switching"], mover & (abin == lb))
        pts.append(s); los.append(s - lo if np.isfinite(lo) else np.nan); his.append(hi - s if np.isfinite(hi) else np.nan)
    ax.errorbar(xs, pts, yerr=[los, his], fmt="-o", color="#dd8452", capsize=3, lw=2, label="skill vs pooled (population-average)")
    st = [skill_ci(p["switching"], mover & (abin == lb))[0] for lb in ABL]
    ax.plot(xs, st, "--", color="#aaaaaa", lw=1, label="vs static / no-switch (weak ref)")
    ax.axhline(0, color="k", lw=0.8); ax.set_ylim(-1.15, 1.15)
    ax.set_xticks(xs); ax.set_xticklabels([f"{l}\n(n={int((mover&(abin==l)).sum())})" for l in ABL], fontsize=8)
    ax.set_xlabel("source clone size"); ax.set_ylabel("switching skill vs pooled null")
    ax.set_title("Clone-specific phenotype switching vs clone size (95% bootstrap CI; switchers only)")
    ax.legend(loc="upper left")
    fig.tight_layout(); f = os.path.join(outdir, "skill_by_clonesize.png"); fig.savefig(f, dpi=140); plt.close(fig)
    written.append(f)

    # ===================== FIG 4: per-tissue phenotype-composition calibration = #
    fig, ax = plt.subplots(1, S, figsize=(6 * S, 5), sharey=True)
    for a in range(S):
        sel = np.where(surv_sw & (src_tis == a))[0]
        oagg = obs_ph[sel].sum(0); oagg = oagg / max(oagg.sum(), 1e-12)
        pagg = prd_ph[sel].sum(0); pagg = pagg / max(pagg.sum(), 1e-12)
        w = .4
        ax[a].bar(np.arange(K) - w / 2, oagg, w, color="#8172b3", label="observed")
        ax[a].bar(np.arange(K) + w / 2, pagg, w, color="#dd8452", label="predicted")
        ax[a].set_xticks(range(K)); ax[a].set_xticklabels(ph, rotation=90, fontsize=7)
        ax[a].set_title(f"{tis[a]}  (n={sel.size} clones)")
    ax[0].set_ylabel("aggregate destination phenotype share"); ax[0].legend()
    fig.suptitle("Within-tissue phenotype composition: observed vs predicted (aggregate over clones)")
    fig.tight_layout(); f = os.path.join(outdir, "phenotype_calibration.png"); fig.savefig(f, dpi=140); plt.close(fig)
    written.append(f)

    # ===================== FIG 5: stay-vs-switch discrimination (ROC/AUC) ====== #
    switch_lbl = (obs_n.argmax(1) != src_phe).astype(int)          # observed: dominant dest phenotype != source
    pswitch = 1 - prd_n[aj, src_phe]                               # predicted P(not keeping source phenotype)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for lg, col in [("all", "k"), ("singleton", "#c44e52"), ("small", "#4c72b0"), ("large", "#55a868")]:
        sel = surv_sw if lg == "all" else (surv_sw & (grp == lg))
        sc = pswitch[sel]; lb = switch_lbl[sel]
        if sel.sum() < 20 or lb.sum() == 0 or lb.sum() == lb.size:
            continue
        a_ = auc(sc, lb); ii = np.where(sel)[0]
        lo, hi = cboot_ci(lambda idx: auc(pswitch[idx], switch_lbl[idx]), ii)
        order = np.argsort(-sc); l = lb[order]
        tpr = np.cumsum(l) / max(l.sum(), 1); fpr = np.cumsum(1 - l) / max((1 - l).sum(), 1)
        ax[0].plot(np.r_[0, fpr], np.r_[0, tpr], color=col, label=f"{lg} AUC={a_:.2f} [{lo:.2f},{hi:.2f}] n={sel.sum()}")
    ax[0].plot([0, 1], [0, 1], "k:", lw=1); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR")
    ax[0].set_title("Stay-vs-switch ROC by clone size"); ax[0].legend(fontsize=8)
    bins = np.linspace(0, 1, 9); mids = .5 * (bins[1:] + bins[:-1])
    op, plo, phi_ = [], [], []
    for i in range(len(bins) - 1):
        inb = surv_sw & (pswitch >= bins[i]) & (pswitch < bins[i + 1])
        pv, lo, hi = wilson(int(switch_lbl[inb].sum()), int(inb.sum()))
        op.append(pv); plo.append(pv - lo if np.isfinite(pv) else np.nan); phi_.append(hi - pv if np.isfinite(pv) else np.nan)
    ax[1].plot([0, 1], [0, 1], "k:", lw=1)
    ax[1].errorbar(mids, op, yerr=[plo, phi_], fmt="o-", color="#dd8452", capsize=3)
    ax[1].set_xlabel("predicted P(switch)"); ax[1].set_ylabel("observed switch frequency")
    ax[1].set_title("Switch-probability reliability (Wilson CI)")
    fig.tight_layout(); f = os.path.join(outdir, "stay_vs_switch.png"); fig.savefig(f, dpi=140); plt.close(fig)
    written.append(f)

    # ===================== FIG 6: PPC on the aggregate phenotype marginal ====== #
    fig, ax = plt.subplots(1, S, figsize=(6 * S, 5))
    for a in range(S):
        sel = surv_sw & (src_tis == a)
        oagg = obs_ph[sel].sum(0); ofrac = oagg / max(oagg.sum(), 1e-12)
        reps = np.empty((len(idx), K))
        for i, si in enumerate(idx):
            mu = np.where(obsv, D * (Xt @ fit.samples[si]), 0.0)
            pp = phi / (phi + mu)
            yi = np.where(obsv, RNG.negative_binomial(np.maximum(phi, 1e-6), np.clip(pp, 1e-9, 1)), 0)
            yb = yi.reshape(J, S, K)[aj, src_tis, :][sel].sum(0)
            reps[i] = yb / max(yb.sum(), 1e-12)
        parts = ax[a].violinplot([reps[:, u] for u in range(K)], positions=range(K), showextrema=False)
        for b in parts["bodies"]:
            b.set_facecolor("#4c72b0"); b.set_alpha(.4)
        ax[a].plot(range(K), ofrac, "rD", ms=6)
        ax[a].set_xticks(range(K)); ax[a].set_xticklabels(ph, rotation=90, fontsize=7)
        ax[a].set_title(f"{tis[a]}  (violin = model replicates, red = observed)")
    ax[0].set_ylabel("aggregate phenotype share")
    fig.suptitle("Posterior-predictive check: within-tissue phenotype marginal")
    fig.tight_layout(); f = os.path.join(outdir, "ppc_phenotype_marginal.png"); fig.savefig(f, dpi=140); plt.close(fig)
    written.append(f)

    # ===================== FIG 7: per-CLONE posterior predictive (Bayesian) ==== #
    # sample individual clones and show the full posterior-predictive within-tissue
    # phenotype distribution (over draws) vs what the clone actually did.
    pick = []
    for lg, mk in [("switcher", mover), ("persister", surv_sw & ~mover)]:
        cand = np.where(mk & (n_src >= 10))[0]
        if cand.size:
            pick += list(cand[np.argsort(-n_src[cand])[:3]])
    pick = pick[:6]
    if pick:
        fig, ax = plt.subplots(2, 3, figsize=(17, 8))
        for k_, j in enumerate(pick):
            a = src_tis[j]
            pp_ph = np.empty((len(idx), K))
            for i, si in enumerate(idx):
                mu = D[j] * (Xt[j] @ fit.samples[si])
                blk = mu.reshape(S, K)[a]; pp_ph[i] = blk / max(blk.sum(), 1e-12)
            axk = ax.flat[k_]
            parts = axk.violinplot([pp_ph[:, u] for u in range(K)], positions=range(K), showextrema=False)
            for b in parts["bodies"]:
                b.set_facecolor("#dd8452"); b.set_alpha(.45)
            axk.plot(range(K), obs_n[j], "kD", ms=6, label="observed")
            axk.plot(range(K), src_n[j], "b_", ms=14, mew=2, label="source")
            axk.set_xticks(range(K)); axk.set_xticklabels(ph, rotation=90, fontsize=6)
            lab = "switcher" if mover[j] else "persister"
            axk.set_title(f"{tis[a]} {ph[src_phe[j]]} clone (n_src={n_src[j]}, {lab})", fontsize=9)
            if k_ == 0:
                axk.legend(fontsize=7)
        fig.suptitle("Per-clone posterior predictive: model's within-tissue phenotype distribution "
                     "(orange) vs observed (black) and source (blue)", y=1.0)
        fig.tight_layout(); f = os.path.join(outdir, "clone_posterior_predictive.png"); fig.savefig(f, dpi=140, bbox_inches="tight"); plt.close(fig)
        written.append(f)

    # ===================== FIG 8: per-clone PPC coverage (Bayesian calibration) = #
    # for each clone: posterior-predictive distribution over the DOMINANT dest phenotype
    # (from NB replicates); is the observed dominant inside the 90% predictive credible set?
    cov_pool = np.where(surv_sw)[0]
    if cov_pool.size > 4000:
        cov_pool = np.sort(RNG.choice(cov_pool, 4000, replace=False))
    dom_pp = np.zeros((cov_pool.size, K))
    for si in idx:
        mu = np.where(obsv, D * (Xt @ fit.samples[si]), 0.0)
        pp = phi / (phi + mu)
        yi = np.where(obsv, RNG.negative_binomial(np.maximum(phi, 1e-6), np.clip(pp, 1e-9, 1)), 0)
        yb = yi.reshape(J, S, K)[aj, src_tis, :][cov_pool]                # [P,K]
        dom = yb.argmax(1); has = yb.sum(1) > 0
        dom_pp[np.arange(cov_pool.size)[has], dom[has]] += 1
    dom_pp /= np.maximum(dom_pp.sum(1, keepdims=True), 1e-12)
    obs_dom = obs_n[cov_pool].argmax(1)
    covered = np.zeros(cov_pool.size, bool)
    for r in range(cov_pool.size):
        order = np.argsort(-dom_pp[r]); cs = np.cumsum(dom_pp[r][order])
        kset = order[:np.searchsorted(cs, 0.90) + 1]
        covered[r] = obs_dom[r] in set(kset.tolist())
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ov = covered.mean()
    ax[0].bar([0], [ov], color="#4c72b0", width=.5)
    ax[0].axhline(0.90, color="r", ls="--", label="nominal 0.90")
    ax[0].set_ylim(0, 1.02); ax[0].set_xticks([0]); ax[0].set_xticklabels([f"all\n(n={cov_pool.size})"])
    ax[0].set_ylabel("coverage of observed dominant phenotype"); ax[0].set_title(f"A. Overall 90% PPC coverage = {ov:.2f}")
    ax[0].legend()
    covs, ns = [], []
    for lb in ABL:
        sel = abin[cov_pool] == lb; ns.append(int(sel.sum()))
        covs.append(covered[sel].mean() if sel.any() else np.nan)
    ax[1].bar(range(len(ABL)), covs, color="#55a868")
    ax[1].axhline(0.90, color="r", ls="--")
    ax[1].set_xticks(range(len(ABL))); ax[1].set_xticklabels([f"{l}\n{n}" for l, n in zip(ABL, ns)], fontsize=8)
    ax[1].set_ylim(0, 1.02); ax[1].set_ylabel("coverage"); ax[1].set_xlabel("clone size")
    ax[1].set_title("B. Coverage by clone size")
    fig.suptitle("Per-clone posterior-predictive coverage of the observed dominant destination phenotype (90% credible set)")
    fig.tight_layout(); f = os.path.join(outdir, "clone_ppc_coverage.png"); fig.savefig(f, dpi=140); plt.close(fig)
    written.append(f)

    # ===================== FIG 9: dominant-phenotype confusion per tissue ====== #
    fig, ax = plt.subplots(1, S, figsize=(6.2 * S, 5.6))
    for a in range(S):
        sel = np.where(mover & (src_tis == a))[0]
        if sel.size < 5:
            ax[a].set_title(f"{tis[a]} (n={sel.size}, too few)"); continue
        o = obs_n[sel].argmax(1); pr = prd_n[sel].argmax(1)
        C = np.zeros((K, K)); np.add.at(C, (o, pr), 1)
        Cn = C / np.maximum(C.sum(1, keepdims=True), 1)
        acc = np.trace(C) / max(C.sum(), 1)
        im = ax[a].imshow(Cn, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax[a].set_xticks(range(K)); ax[a].set_xticklabels(ph, rotation=90, fontsize=6)
        ax[a].set_yticks(range(K)); ax[a].set_yticklabels(ph, fontsize=6)
        ax[a].set_xlabel("predicted dominant phenotype"); ax[a].set_ylabel("observed dominant phenotype")
        ax[a].set_title(f"{tis[a]}: switch recall (acc={acc:.2f}, n={sel.size})", fontsize=10)
        fig.colorbar(im, ax=ax[a], fraction=0.046)
    fig.suptitle("Dominant destination-phenotype confusion among switchers (row-normalized = recall)")
    fig.tight_layout(); f = os.path.join(outdir, "confusion_dominant_phenotype.png"); fig.savefig(f, dpi=140, bbox_inches="tight"); plt.close(fig)
    written.append(f)

    # ===================== FIG 10: directed differentiation flux =============== #
    # net directed phenotype flux within each tissue: rho-weighted (H[a][u,v]-H[a][v,u]).
    # positive u->v = net differentiation from u to v. Observed vs model.
    fig, ax = plt.subplots(S, 2, figsize=(13, 4.6 * S))
    for a in range(S):
        # observed within-tissue phenotype operator
        Hobs = np.zeros((K, K))
        for u in range(K):
            sel = surv_sw & (src_tis == a) & (src_phe == u)
            if sel.any():
                v = obs_ph[sel].sum(0); Hobs[u] = v / max(v.sum(), 1e-12)
        for k_, (Hmat, ttl) in enumerate([(Hobs, "observed"), (H[a], "model")]):
            net = (rho[a][:, None] * Hmat) - (rho[a][None, :] * Hmat.T)   # net u->v flux
            vmax = float(np.nanpercentile(np.abs(net), 98)) or 0.02
            im = ax[a, k_].imshow(net, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            ax[a, k_].set_xticks(range(K)); ax[a, k_].set_xticklabels(ph, rotation=90, fontsize=6)
            ax[a, k_].set_yticks(range(K)); ax[a, k_].set_yticklabels(ph, fontsize=6)
            ax[a, k_].set_title(f"{tis[a]}: net differentiation flux ({ttl})", fontsize=10)
            ax[a, k_].set_xlabel("to phenotype"); ax[a, k_].set_ylabel("from phenotype")
            fig.colorbar(im, ax=ax[a, k_], fraction=0.046)
    fig.suptitle("Directed differentiation: net within-tissue phenotype flux (red = net source->dest), observed vs model", y=1.0)
    fig.tight_layout(); f = os.path.join(outdir, "directed_differentiation.png"); fig.savefig(f, dpi=140, bbox_inches="tight"); plt.close(fig)
    written.append(f)

    # ===================== FIG 11: top-k accuracy among switchers ============== #
    fig, ax = plt.subplots(figsize=(9, 5))
    ks = [1, 2, 3]; width = 0.26
    for a in range(S):
        sel = np.where(mover & (src_tis == a))[0]
        if sel.size < 5:
            accs = [np.nan] * 3
        else:
            od = obs_n[sel].argmax(1); rank = np.argsort(-prd_n[sel], axis=1)
            accs = [float(np.mean([od[i] in rank[i, :k] for i in range(sel.size)])) for k in ks]
        ax.bar(np.arange(3) + a * width, accs, width, label=f"{tis[a]} (n={sel.size})")
    ax.set_xticks(np.arange(3) + width); ax.set_xticklabels([f"top-{k}" for k in ks])
    ax.set_ylim(0, 1.02); ax.set_ylabel("switcher dominant-destination accuracy")
    ax.set_title("Top-k accuracy: is the true switch destination in the model's top-k? (calibrated-not-sharp)")
    ax.legend()
    fig.tight_layout(); f = os.path.join(outdir, "topk_accuracy.png"); fig.savefig(f, dpi=140); plt.close(fig)
    written.append(f)

    # ===================== FIG 12: curated directed transitions (obs vs CrI) === #
    fig, ax = plt.subplots(1, S, figsize=(7 * S, 5.6), sharey=True)
    elabs = [lab for _, _, lab in _CURATED]; xe = np.arange(len(_CURATED)); w = 0.4
    for a in range(S):
        o_pt = [Hobs_all[a, u, v] for u, v, _ in _CURATED]
        m_pt = np.array([Hdraws[:, a, u, v].mean() for u, v, _ in _CURATED])
        m_lo = np.array([Hdraws[:, a, u, v].mean() - np.quantile(Hdraws[:, a, u, v], .025) for u, v, _ in _CURATED])
        m_hi = np.array([np.quantile(Hdraws[:, a, u, v], .975) - Hdraws[:, a, u, v].mean() for u, v, _ in _CURATED])
        ax[a].bar(xe - w / 2, o_pt, w, color="#8172b3", label="observed")
        ax[a].bar(xe + w / 2, m_pt, w, color="#dd8452", yerr=[m_lo, m_hi], capsize=2, label="model (95% CrI)")
        ax[a].set_xticks(xe); ax[a].set_xticklabels(elabs, rotation=90, fontsize=7); ax[a].set_title(tis[a])
    ax[0].set_ylabel("P(source -> dest | stay in tissue)"); ax[0].legend(fontsize=8)
    fig.suptitle("Curated directed transitions: observed vs model switching probability (95% posterior CrI)")
    fig.tight_layout(); f = os.path.join(outdir, "curated_transitions.png"); fig.savefig(f, dpi=140, bbox_inches="tight"); plt.close(fig)
    written.append(f)

    # ===================== FIG 13: switching skill by source phenotype ========= #
    fig, ax = plt.subplots(figsize=(11, 5))
    pts, los, his, ns = [], [], [], []
    for u in range(K):
        base = mover & (src_phe == u)
        s, lo, hi = skill_ci(c["switching"], base)
        pts.append(s); los.append(s - lo if np.isfinite(lo) else np.nan); his.append(hi - s if np.isfinite(hi) else np.nan)
        ns.append(int(base.sum()))
    ax.bar(range(K), pts, color=["#55a868" if n >= 30 else "#c8c8c8" for n in ns], yerr=[los, his], capsize=3)
    ax.axhline(0, color="k", lw=.8); ax.set_xticks(range(K))
    ax.set_xticklabels([f"{ph[u]}\n(n={ns[u]})" for u in range(K)], rotation=90, fontsize=7)
    ax.set_ylabel("switching skill vs pooled"); ax.set_ylim(-1.15, 1.15)
    ax.set_title("Which source phenotypes' switching is predictable (skill vs pooled; grey = <30 switchers)")
    fig.tight_layout(); f = os.path.join(outdir, "skill_by_source_phenotype.png"); fig.savefig(f, dpi=140, bbox_inches="tight"); plt.close(fig)
    written.append(f)

    # ===================== FIG 14: tissue-specific differentiation contrast ==== #
    def prog_score(Hmat, a):
        return float(sum(rho[a][u] * (Hmat[a][u, v] - Hmat[a][v, u]) for u, v in _PROG))
    sc_draws = np.array([[prog_score(Hdraws[s_], a) for a in range(S)] for s_ in range(len(idx))])   # [draws,S]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    md = sc_draws.mean(0); lo = np.quantile(sc_draws, .025, 0); hi = np.quantile(sc_draws, .975, 0)
    ax[0].bar(range(S), md, color=["#c44e52", "#4c72b0", "#8172b3"][:S], yerr=[md - lo, hi - md], capsize=4)
    ax[0].axhline(0, color="k", lw=.8); ax[0].set_xticks(range(S)); ax[0].set_xticklabels(tis)
    ax[0].set_ylabel("net progression-toward-exhaustion flux"); ax[0].set_title("A. Differentiation drive by tissue (95% CrI)")
    iTP = tis.index("TP") if "TP" in tis else S - 1; iPB = tis.index("PBMC") if "PBMC" in tis else 0
    ax[0].text(0.02, 0.96, f"P(TP > PBMC) = {(sc_draws[:, iTP] > sc_draws[:, iPB]).mean():.2f}",
               transform=ax[0].transAxes, fontsize=9, va="top")
    Djs = np.zeros((S, S))
    for a in range(S):
        for b in range(S):
            if a != b:
                vals = [_js(Hobs_all[a][u], Hobs_all[b][u]) for u in range(K)
                        if np.isfinite(Hobs_all[a][u]).all() and np.isfinite(Hobs_all[b][u]).all()]
                Djs[a, b] = float(np.mean(vals)) if vals else np.nan
    im = ax[1].imshow(Djs, cmap="viridis")
    ax[1].set_xticks(range(S)); ax[1].set_xticklabels(tis); ax[1].set_yticks(range(S)); ax[1].set_yticklabels(tis)
    for a in range(S):
        for b in range(S):
            if np.isfinite(Djs[a, b]):
                ax[1].text(b, a, f"{Djs[a, b]:.2f}", ha="center", va="center", color="w")
    ax[1].set_title("B. Switching-operator dissimilarity across tissues (mean row-JS)")
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    fig.suptitle("Tissue-specific differentiation structure")
    fig.tight_layout(); f = os.path.join(outdir, "tissue_contrast.png"); fig.savefig(f, dpi=140); plt.close(fig)
    written.append(f)

    # ===================== FIG 15: PBMC effector over-prediction (survivorship?) = #
    iPBMC = tis.index("PBMC") if "PBMC" in tis else 0
    fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for k_, (glab, gmask) in enumerate([("small (n<50)", n_src < 50), ("large (n>=50)", n_src >= 50)]):
        sel = np.where(surv_sw & (src_tis == iPBMC) & gmask)[0]
        if sel.size < 3:
            ax[k_].set_title(f"PBMC {glab} (n={sel.size}, too few)"); continue
        oagg = obs_ph[sel].sum(0); oagg = oagg / max(oagg.sum(), 1e-12)
        pagg = prd_ph[sel].sum(0); pagg = pagg / max(pagg.sum(), 1e-12)
        gap = pagg - oagg                                          # + = model over-predicts
        w = .4
        ax[k_].bar(np.arange(K) - w / 2, oagg, w, color="#8172b3", label="observed")
        ax[k_].bar(np.arange(K) + w / 2, pagg, w, color="#dd8452", label="predicted")
        ax[k_].set_xticks(range(K)); ax[k_].set_xticklabels(ph, rotation=90, fontsize=7)
        ax[k_].set_title(f"PBMC {glab} (n={sel.size})  |  TEXeff+TEXterm over-pred = {gap[1] + gap[2]:+.3f}")
    ax[0].set_ylabel("aggregate destination phenotype share"); ax[0].legend()
    fig.suptitle("PBMC CD8-effector over-prediction: does the over-call shrink for large clones? (shrink => survivorship/censoring)")
    fig.tight_layout(); f = os.path.join(outdir, "pbmc_effector_check.png"); fig.savefig(f, dpi=140, bbox_inches="tight"); plt.close(fig)
    written.append(f)

    # ===================== FIG 16: per-clone P(switch) with posterior CrI ====== #
    # (matches migration's per-clone leave-probability panel) propagate posterior M-draws to a
    # per-clone switch-probability distribution -> 95% credible interval per clone, sorted,
    # with observed switch/keep outcomes.
    svv = np.where(surv_sw)[0]
    Xs, Ds, ov = Xt[svv], D[svv], obsv[svv]
    st_sv, sp_sv = src_tis[svv], src_phe[svv]
    PS = np.empty((len(idx), svv.size))
    for s_, si in enumerate(idx):
        mu = np.where(ov, Ds * (Xs @ fit.samples[si]), 0.0).reshape(svv.size, S, K)
        blk = mu[np.arange(svv.size), st_sv, :]
        muf = blk / np.maximum(blk.sum(1, keepdims=True), 1e-12)
        PS[s_] = 1 - muf[np.arange(svv.size), sp_sv]
    psmean = PS.mean(0); pslo = np.quantile(PS, .025, 0); pshi = np.quantile(PS, .975, 0)
    sw = switch_lbl[svv]; order = np.argsort(psmean); xr = np.arange(svv.size)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.fill_between(xr, pslo[order], pshi[order], color="#dd8452", alpha=0.25, label="95% posterior CrI")
    ax.plot(xr, psmean[order], color="#8c4a1f", lw=1, label="posterior mean P(switch)")
    o = sw[order]
    ax.scatter(xr[o == 1], np.full(int((o == 1).sum()), 1.03), s=1.2, color="#c44e52", zorder=3, label="observed: switched")
    ax.scatter(xr[o == 0], np.full(int((o == 0).sum()), -0.03), s=1.2, color="#55a868", zorder=3, label="observed: kept")
    ax.axhline(1.0, color="#dddddd", lw=.6); ax.axhline(0.0, color="#dddddd", lw=.6)
    ax.set_ylim(-0.15, 1.15); ax.set_xlabel("survivor clones (sorted by posterior mean P(switch))")
    ax.set_ylabel("P(switch phenotype | stay in tissue)")
    ax.set_title(f"Per-clone switch probability with posterior credible intervals (n={svv.size} survivors)")
    ax.legend(fontsize=8, loc="center left")
    fig.tight_layout(); f = os.path.join(outdir, "switch_probability_perclone.png"); fig.savefig(f, dpi=140); plt.close(fig)
    written.append(f)

    # ===================== FIG 17: within- vs across-tissue switching =========== #
    # A clone's phenotype can change IN-PLACE (within-tissue switching, what this whole
    # inventory validates) or WHILE MIGRATING (cross-tissue). These are different events.
    # From T_hat, per source state: (A) what fraction of phenotype-CHANGE mass is in-place vs
    # via migration, and (B) is the phenotype FATE the same whether a cell stays or migrates
    # (JS: low = one tissue-agnostic program -> within-tissue validation is representative;
    # high = staying and migrating switch to DIFFERENT phenotypes -> not comparable).
    frac_within = np.full((S, K), np.nan); js_fate = np.full((S, K), np.nan)
    for a in range(S):
        for u in range(K):
            row = T_hat[a * K + u].reshape(S, K)
            same = row[a].copy()                                 # dest phenotypes, same tissue
            other = row.sum(0) - same                            # summed over other tissues, per phenotype
            within_sw = same.sum() - same[u]                     # stay, phenotype != source
            cross_sw = other.sum() - other[u]                    # migrate, phenotype != source
            tot = within_sw + cross_sw
            frac_within[a, u] = within_sw / tot if tot > 1e-9 else np.nan
            sf = same.copy(); sf[u] = 0; mf = other.copy(); mf[u] = 0
            if sf.sum() > 1e-9 and mf.sum() > 1e-9:
                js_fate[a, u] = _js(sf, mf)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    for k_, (M_, ttl, cmap, vlim) in enumerate([
            (frac_within, "A. Fraction of phenotype-change mass that is WITHIN-tissue\n(1 = all in-place; 0 = all via migration)", "RdBu_r", (0, 1)),
            (js_fate, "B. JS(stay-fate, migrate-fate) per source state\n(0 = same program staying vs migrating; high = different)", "magma", (0, 1))]):
        im = ax[k_].imshow(M_, cmap=cmap, vmin=vlim[0], vmax=vlim[1], aspect="auto")
        ax[k_].set_xticks(range(K)); ax[k_].set_xticklabels(ph, rotation=90, fontsize=7)
        ax[k_].set_yticks(range(S)); ax[k_].set_yticklabels(tis); ax[k_].set_title(ttl, fontsize=10)
        for a in range(S):
            for u in range(K):
                if np.isfinite(M_[a, u]):
                    ax[k_].text(u, a, f"{M_[a, u]:.2f}", ha="center", va="center", fontsize=6,
                                color="w" if (vlim[1] - M_[a, u]) < 0.4 or M_[a, u] < 0.2 else "k")
        fig.colorbar(im, ax=ax[k_], fraction=0.046)
    fig.suptitle("Within- vs across-tissue phenotype switching: is the within-tissue validation representative?")
    fig.tight_layout(); f = os.path.join(outdir, "within_vs_across_switching.png"); fig.savefig(f, dpi=140, bbox_inches="tight"); plt.close(fig)
    written.append(f)

    print("wrote:")
    for w in written:
        print(" ", w)


if __name__ == "__main__":
    main()
