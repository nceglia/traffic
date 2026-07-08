"""Migration validation inventory (batch 1: full fit + posterior draws).

Validates the operator's CROSS-TISSUE redistribution (destination tissue marginal)
from every angle we can with the full fit alone. Writes one standalone PNG per check
to figures/validation/migration/ (batch 2 -- holdout influence -- is separate).

Clone-size groups (per source count n_src):  singleton n=1  |  small 2-49  |  large >=50.

Env: TRAFFIC_H5 (required), FIT (default nb_global_putative.npz),
     OUTDIR (default figures/validation/migration), PPC_DRAWS (default 400).
"""
import os
import sys

import numpy as np
from scipy.stats import rankdata, nbinom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predictability_breakdown import decompose_prediction, _js_rows  # noqa: E402

from traffic import data, io, statespace  # noqa: E402
from traffic.readouts import _js, decompose, tissue_traffic  # noqa: E402

RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------- #
# small stats helpers (CIs)
# --------------------------------------------------------------------------- #
def boot_ci(fn, n, n_boot=300, alpha=0.05):
    """Bootstrap (lo, hi) of a statistic fn(idx) over n items, resampling indices."""
    vals = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        v = fn(idx)
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < 10:
        return np.nan, np.nan
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


_PAT = None   # set to the per-clone patient array in main(); the honest sampling unit


def cboot_ci(fn, base_idx, n_boot=300, alpha=0.05):
    """Patient-CLUSTER bootstrap (lo, hi): resample patients (clones correlate within patient,
    so a clone bootstrap is anti-conservative). fn takes ABSOLUTE clone indices."""
    base_idx = np.asarray(base_idx)
    pats = np.unique(_PAT[base_idx])
    groups = {pp: base_idx[_PAT[base_idx] == pp] for pp in pats}
    vals = []
    for _ in range(n_boot):
        samp = RNG.choice(pats, len(pats), replace=True)
        idx = np.concatenate([groups[pp] for pp in samp])
        v = fn(idx)
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < 10:
        return np.nan, np.nan
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)


def auc(score, label):
    label = np.asarray(label); score = np.asarray(score)
    npos = int(label.sum()); nneg = int((label == 0).sum())
    if npos == 0 or nneg == 0:
        return np.nan
    r = rankdata(score)
    return (r[label == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def main():
    global _PAT
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h5 = os.environ.get("TRAFFIC_H5")
    if not h5:
        raise SystemExit("set TRAFFIC_H5")
    fit_path = os.environ.get("FIT", "nb_global_putative.npz")
    outdir = os.environ.get("OUTDIR", "figures/validation/migration")
    n_ppc = int(os.environ.get("PPC_DRAWS", "400"))
    os.makedirs(outdir, exist_ok=True)

    ss = statespace.default()
    obs = data.build(data.load_obs_table(h5), ss)
    fit = io.load_fit(fit_path)
    S, K, L = ss.S, ss.K, ss.L
    tis = list(ss.tissues)
    Xt, Y, D = obs.Xtilde, obs.Y, obs.D
    J = Xt.shape[0]
    phi = float(np.exp(np.asarray(fit.dispersion["params"]["log_r"]).mean())) if fit.dispersion else np.inf
    obsv = D > 0
    Yt = np.where(obsv, Y, 0.0)
    pi = Yt.sum(0); pi = pi / max(pi.sum(), 1e-12)
    print(f"fit={fit_path} J={J} phi={phi:.3f} outdir={outdir}")

    mean_model = np.where(obsv, D * (Xt @ fit.M_hat), 0.0)
    mean_static = np.where(obsv, D * Xt, 0.0)
    mean_pooled = np.where(obsv, D * (Xt.sum(1, keepdims=True) * pi[None, :]), 0.0)

    m = decompose_prediction(Xt, Y, D, mean_model, ss, phi)
    p = decompose_prediction(Xt, Y, D, mean_static, ss, phi)
    c = decompose_prediction(Xt, Y, D, mean_pooled, ss, phi)
    surv = m["surv"]

    # per-clone tissue marginals (counts) + source tissue
    src_tis = obs.src_tissue   # raw-count attribution (not Xtilde); see docs/DATA.md
    obs_tm = Yt.reshape(J, S, K).sum(2)                 # [J,S] observed dest tissue counts
    prd_tm = mean_model.reshape(J, S, K).sum(2)         # [J,S] predicted
    n_src = obs.n_src.astype(int)
    grp = np.where(n_src == 1, "singleton", np.where(n_src < 50, "small", "large"))
    patient = np.asarray(obs.patient); _PAT = patient   # honest sampling unit for cluster CIs

    # mover mask (migration only interesting where tissue distribution actually shifts)
    src_t = obs.X.reshape(J, S, K).sum(2); src_t = src_t / np.maximum(src_t.sum(1, keepdims=True), 1e-12)
    obs_t = obs_tm / np.maximum(obs_tm.sum(1, keepdims=True), 1e-12)
    mig_ch = np.full(J, np.nan); mig_ch[surv] = _js_rows(src_t[surv], obs_t[surv])
    mov = surv & (mig_ch > 0.05)

    # rho = source phenotype occupancy per tissue (for the operator read-out)
    Xr = obs.X.reshape(J, S, K).sum(0)                  # [S,K] total source mass
    rho = Xr / np.maximum(Xr.sum(1, keepdims=True), 1e-12)

    written = []

    # ===================== FIG 1: observed vs model traffic matrices ============ #
    def agg_traffic(tm):
        P = np.zeros((S, S))
        for a in range(S):
            sel = surv & (src_tis == a)
            if sel.any():
                v = tm[sel].sum(0); P[a] = v / max(v.sum(), 1e-12)
        return P
    P_obs = agg_traffic(obs_tm); P_mod = agg_traffic(prd_tm)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for a, (M_, ttl) in enumerate([(P_obs, "observed"), (P_mod, "model"), (P_mod - P_obs, "model - observed")]):
        im = ax[a].imshow(M_, cmap="RdBu" if a == 2 else "viridis",
                          vmin=-0.3 if a == 2 else 0, vmax=0.3 if a == 2 else 1)
        ax[a].set_xticks(range(S)); ax[a].set_xticklabels(tis); ax[a].set_yticks(range(S)); ax[a].set_yticklabels(tis)
        ax[a].set_xlabel("destination"); ax[a].set_ylabel("source"); ax[a].set_title(f"{ttl}")
        for i in range(S):
            for j in range(S):
                ax[a].text(j, i, f"{M_[i, j]:.2f}", ha="center", va="center",
                           color="w" if (a < 2 and M_[i, j] < 0.6) else "k", fontsize=9)
        fig.colorbar(im, ax=ax[a], fraction=0.046)
    fig.suptitle("Migration: tissue->tissue trafficking (rows sum to 1; diagonal = persistence)")
    fig.tight_layout(); f1 = os.path.join(outdir, "traffic_matrices.png"); fig.savefig(f1, dpi=140); plt.close(fig)
    written.append(f1)

    # ===================== FIG 2: operator posterior credible intervals ========= #
    idx = np.sort(RNG.choice(fit.samples.shape[0], min(n_ppc, fit.samples.shape[0]), replace=False))
    Pdraws = np.empty((len(idx), S, S))
    for s_, si in enumerate(idx):
        T_s, _ = decompose(fit.samples[si])
        Pdraws[s_] = tissue_traffic(T_s, ss, rho)
    Pmean = Pdraws.mean(0); Plo = np.quantile(Pdraws, 0.025, 0); Phi = np.quantile(Pdraws, 0.975, 0)
    fig, ax = plt.subplots(figsize=(8, 5))
    xlab = [f"{tis[a]}->{tis[b]}" for a in range(S) for b in range(S)]
    xs = np.arange(len(xlab))
    means = np.array([Pmean[a, b] for a in range(S) for b in range(S)])
    los = np.array([Pmean[a, b] - Plo[a, b] for a in range(S) for b in range(S)])
    his = np.array([Phi[a, b] - Pmean[a, b] for a in range(S) for b in range(S)])
    diag = np.array([a == b for a in range(S) for b in range(S)])
    ax.bar(xs[diag], means[diag], color="#55a868", label="persistence (stay)")
    ax.bar(xs[~diag], means[~diag], color="#4c72b0", label="migration (go)")
    ax.errorbar(xs, means, yerr=[los, his], fmt="none", ecolor="k", capsize=3, lw=1)
    ax.set_xticks(xs); ax.set_xticklabels(xlab, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("operator rate  (rho-weighted)"); ax.legend()
    ax.set_title("Migration operator with 95% posterior credible intervals")
    fig.tight_layout(); f2 = os.path.join(outdir, "operator_credible_intervals.png"); fig.savefig(f2, dpi=140); plt.close(fig)
    written.append(f2)

    # ===================== FIG 3: skill by clone size (both nulls) ============== #
    ABL = ["1", "2", "3-4", "5-9", "10-49", "50+"]
    def abin_of(n):
        return "1" if n <= 1 else "2" if n <= 2 else "3-4" if n <= 4 else "5-9" if n <= 9 else "10-49" if n <= 49 else "50+"
    abin = np.array([abin_of(n) for n in n_src])

    def skill_ci(null_err, base):
        b = np.where(base & np.isfinite(m["migration"]) & np.isfinite(null_err))[0]
        if b.size < 10:
            return np.nan, np.nan, np.nan
        pt = 1 - np.median(m["migration"][b]) / max(np.median(null_err[b]), 1e-9)
        lo, hi = cboot_ci(lambda idx: 1 - np.median(m["migration"][idx]) / max(np.median(null_err[idx]), 1e-9), b)
        return pt, lo, hi
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.arange(len(ABL))
    # pooled = the honest bar (clone must beat the population average); plotted prominently
    pts, los, his = [], [], []
    for lb in ABL:
        s, lo, hi = skill_ci(c["migration"], mov & (abin == lb))
        pts.append(s); los.append(s - lo if np.isfinite(lo) else np.nan); his.append(hi - s if np.isfinite(hi) else np.nan)
    ax.errorbar(xs, pts, yerr=[los, his], fmt="-o", color="#4c72b0", capsize=3, lw=2,
                label="skill vs pooled (population-average)")
    # static = weak reference (beats 'no change' for free); faint, context only
    st = [skill_ci(p["migration"], mov & (abin == lb))[0] for lb in ABL]
    ax.plot(xs, st, "--", color="#aaaaaa", lw=1, label="vs static / no-change (weak ref)")
    ax.axhline(0, color="k", lw=0.8); ax.set_ylim(-1.15, 1.15)
    ax.set_xticks(xs); ax.set_xticklabels([f"{l}\n(n={int((mov&(abin==l)).sum())})" for l in ABL], fontsize=8)
    ax.set_xlabel("source clone size"); ax.set_ylabel("migration skill vs pooled null")
    ax.set_title("Clone-specific migration is a large-clone signal (95% patient-cluster CI)")
    ax.text(0.02, 0.03, "skill = 1 - err_model / err_null\n1 = perfect,  0 = as good as null,  <0 = worse than null",
            transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(boxstyle="round", fc="white", ec="#cccccc", alpha=0.9))
    ax.legend(loc="upper left")
    fig.tight_layout(); f3 = os.path.join(outdir, "skill_by_clonesize.png"); fig.savefig(f3, dpi=140); plt.close(fig)
    written.append(f3)

    # ===================== FIG 4: route calibration (obs vs pred per route) ===== #
    fig, ax = plt.subplots(1, S, figsize=(15, 4.5), sharey=True)
    of = obs_tm / np.maximum(obs_tm.sum(1, keepdims=True), 1e-12)
    pf = prd_tm / np.maximum(prd_tm.sum(1, keepdims=True), 1e-12)
    for a in range(S):
        sel = np.where(surv & (src_tis == a))[0]
        xs = np.arange(S); w = 0.38
        o_pt = [of[sel, b].mean() for b in range(S)]; p_pt = [pf[sel, b].mean() for b in range(S)]
        o_ci = [cboot_ci(lambda idx, bb=b: of[idx, bb].mean(), sel) for b in range(S)]
        p_ci = [cboot_ci(lambda idx, bb=b: pf[idx, bb].mean(), sel) for b in range(S)]
        ax[a].bar(xs - w / 2, o_pt, w, color="#8172b3", label="observed",
                  yerr=[[o_pt[b] - o_ci[b][0] for b in range(S)], [o_ci[b][1] - o_pt[b] for b in range(S)]], capsize=3)
        ax[a].bar(xs + w / 2, p_pt, w, color="#dd8452", label="predicted",
                  yerr=[[p_pt[b] - p_ci[b][0] for b in range(S)], [p_ci[b][1] - p_pt[b] for b in range(S)]], capsize=3)
        ax[a].set_xticks(xs); ax[a].set_xticklabels(tis); ax[a].set_title(f"source = {tis[a]}  (n={sel.size})")
        ax[a].set_xlabel("destination")
    ax[0].set_ylabel("fraction of descendant mass"); ax[0].legend()
    fig.suptitle("Route calibration: observed vs predicted fraction of descendant mass per source tissue (diagonal = persistence)")
    fig.tight_layout(); f4 = os.path.join(outdir, "route_calibration.png"); fig.savefig(f4, dpi=140); plt.close(fig)
    written.append(f4)

    # ===================== FIG 5: stay-vs-go discrimination (ROC/AUC) ========== #
    leave_lbl = (obs_tm[np.arange(J), src_tis] < obs_tm.sum(1) - 1e-9).astype(int)   # any mass outside source
    pleave = 1 - pf[np.arange(J), src_tis]                                            # predicted P(leave)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for label_grp, col in [("all", "k"), ("singleton", "#c44e52"), ("small", "#4c72b0"), ("large", "#55a868")]:
        sel = surv if label_grp == "all" else (surv & (grp == label_grp))
        sc = pleave[sel]; lb = leave_lbl[sel]
        a_ = auc(sc, lb)
        idxs = np.where(sel)[0]
        lo, hi = cboot_ci(lambda idx: auc(pleave[idx], leave_lbl[idx]), idxs)
        # ROC curve
        order = np.argsort(-sc); l = lb[order]
        tpr = np.cumsum(l) / max(l.sum(), 1); fpr = np.cumsum(1 - l) / max((1 - l).sum(), 1)
        ax[0].plot(np.r_[0, fpr], np.r_[0, tpr], color=col,
                   label=f"{label_grp} AUC={a_:.2f} [{lo:.2f},{hi:.2f}] n={sel.sum()}")
    ax[0].plot([0, 1], [0, 1], "k:", lw=1); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR")
    ax[0].set_title("Stay-vs-go ROC by clone size"); ax[0].legend(fontsize=8)
    # reliability of P(leave)
    bins = np.linspace(0, 1, 9); mids = 0.5 * (bins[1:] + bins[:-1])
    who = surv
    op, plo, phi_ = [], [], []
    for i in range(len(bins) - 1):
        inb = who & (pleave >= bins[i]) & (pleave < bins[i + 1])
        k = int(leave_lbl[inb].sum()); nn = int(inb.sum())
        pv, lo, hi = wilson(k, nn)
        op.append(pv); plo.append(pv - lo if np.isfinite(pv) else np.nan); phi_.append(hi - pv if np.isfinite(pv) else np.nan)
    ax[1].plot([0, 1], [0, 1], "k:", lw=1)
    ax[1].errorbar(mids, op, yerr=[plo, phi_], fmt="o-", color="#4c72b0", capsize=3)
    ax[1].set_xlabel("predicted P(leave)"); ax[1].set_ylabel("observed leave frequency")
    ax[1].set_title("Leave-probability reliability (Wilson CI)")
    fig.tight_layout(); f5 = os.path.join(outdir, "stay_vs_go.png"); fig.savefig(f5, dpi=140); plt.close(fig)
    written.append(f5)

    # ===================== FIG 6: PPC on the aggregate tissue marginal ========= #
    reps = np.empty((n_ppc, S))
    obs_tm_agg = obs_tm.sum(0); obs_frac = obs_tm_agg / obs_tm_agg.sum()
    r_glob = phi
    for i, si in enumerate(idx):
        mu = np.where(obsv, D * (Xt @ fit.samples[si]), 0.0)
        pp = r_glob / (r_glob + mu)
        yi = np.where(obsv, RNG.negative_binomial(np.maximum(r_glob, 1e-6), np.clip(pp, 1e-9, 1)), 0)
        tm = yi.reshape(J, S, K).sum((0, 2)); reps[i] = tm / max(tm.sum(), 1e-12)
    fig, ax = plt.subplots(figsize=(8, 5))
    parts = ax.violinplot([reps[:, a] for a in range(S)], positions=range(S), showextrema=False)
    for b in parts["bodies"]:
        b.set_facecolor("#4c72b0"); b.set_alpha(0.4)
    ax.plot(range(S), obs_frac, "rD", ms=9, label="observed")
    for a in range(S):
        pv = 2 * min((reps[:, a] >= obs_frac[a]).mean(), (reps[:, a] <= obs_frac[a]).mean())
        ax.text(a, obs_frac[a], f" p={pv:.2f}", va="center", fontsize=9)
    ax.set_xticks(range(S)); ax.set_xticklabels(tis); ax.set_ylabel("aggregate tissue share")
    ax.set_title("Posterior-predictive check: aggregate tissue marginal (violin = model replicates)")
    ax.legend()
    fig.tight_layout(); f6 = os.path.join(outdir, "ppc_tissue_marginal.png"); fig.savefig(f6, dpi=140); plt.close(fig)
    written.append(f6)

    # ===================== FIG 7: destination-mass calibration (obs vs pred) === #
    # per (survivor clone, destination tissue): predicted vs observed FRACTION of descendant mass.
    of_all = (obs_tm[surv] / np.maximum(obs_tm[surv].sum(1, keepdims=True), 1e-12)).ravel()
    pf_all = pf[surv].ravel()
    r = float(np.corrcoef(pf_all, of_all)[0, 1])
    # patient-cluster bootstrap of the calibration correlation (resample patients, not clone×tissue cells)
    sv7 = np.where(surv)[0]; pat7 = patient[sv7]; upats = np.unique(pat7); rvals = []
    for _ in range(200):
        samp = RNG.choice(upats, len(upats), replace=True)
        ii = np.concatenate([np.where(pat7 == pp)[0] for pp in samp])
        rvals.append(float(np.corrcoef(pf[sv7[ii]].ravel(), of[sv7[ii]].ravel())[0, 1]))
    rlo, rhi = float(np.quantile(rvals, 0.025)), float(np.quantile(rvals, 0.975))
    fig, ax = plt.subplots(figsize=(7, 6))
    hb = ax.hexbin(pf_all, of_all, gridsize=40, bins="log", cmap="Blues", mincnt=1)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="identity (perfect)")
    bm = [of_all[(pf_all >= bins[i]) & (pf_all < bins[i + 1])].mean()
          if ((pf_all >= bins[i]) & (pf_all < bins[i + 1])).any() else np.nan for i in range(len(bins) - 1)]
    ax.plot(mids, bm, "r-o", lw=2, label="binned mean observed")
    ax.set_xlabel("predicted fraction of descendant mass"); ax.set_ylabel("observed fraction of descendant mass")
    ax.set_title(f"Destination-mass calibration  (Pearson r = {r:.2f} [{rlo:.2f}, {rhi:.2f}])")
    ax.legend(loc="upper left"); fig.colorbar(hb, ax=ax, label="clone×tissue count (log)")
    fig.tight_layout(); f7 = os.path.join(outdir, "destination_mass_scatter.png"); fig.savefig(f7, dpi=140); plt.close(fig)
    written.append(f7)

    # ===================== FIG 8: size-group summary ========================== #
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))
    groups = ["singleton", "small", "large"]
    # A: stay/go AUC vs clone size across finer bins (curve), + logistic-regression accuracy trend
    au_pt, au_lo, au_hi = [], [], []
    for lb in ABL:
        sel = surv & (abin == lb); idxs = np.where(sel)[0]
        if idxs.size < 20:
            au_pt.append(np.nan); au_lo.append(0); au_hi.append(0); continue
        a_ = auc(pleave[sel], leave_lbl[sel]); lo, hi = cboot_ci(lambda idx: auc(pleave[idx], leave_lbl[idx]), idxs)
        au_pt.append(a_); au_lo.append(a_ - lo if np.isfinite(lo) else 0); au_hi.append(hi - a_ if np.isfinite(hi) else 0)
    ax[0].errorbar(range(len(ABL)), au_pt, yerr=[au_lo, au_hi], fmt="-o", color="#55a868", capsize=3, lw=2)
    # logistic regression: P(stay/go predicted correctly at 0.5) ~ log(size); overlay accuracy trend
    pr = (pleave > 0.5).astype(int); correct = (pr == leave_lbl).astype(float)
    xs_l = np.log1p(n_src[surv]); yy = correct[surv]
    b1, b0 = np.polyfit(xs_l, yy, 1)   # linear proxy for the accuracy trend on log-size
    ax[0].axhline(0.5, color="k", ls=":"); ax[0].set_xticks(range(len(ABL))); ax[0].set_xticklabels(ABL, fontsize=8)
    ax[0].set_ylim(0.45, 1.02); ax[0].set_ylabel("stay/go AUC"); ax[0].set_xlabel("source clone size")
    ax[0].set_title(f"A. Discrimination vs clone size (acc~log-size slope {b1:+.03f})")
    # migration skill vs pooled per group
    sk = []
    for g in groups:
        s, lo, hi = skill_ci(c["migration"], mov & (grp == g)); sk.append((s, lo, hi))
    ax[1].bar(range(3), [s[0] for s in sk], color="#4c72b0",
              yerr=[[s[0] - s[1] if np.isfinite(s[1]) else 0 for s in sk], [s[2] - s[0] if np.isfinite(s[2]) else 0 for s in sk]], capsize=4)
    ax[1].axhline(0, color="k", ls=":"); ax[1].set_xticks(range(3)); ax[1].set_xticklabels(groups)
    ax[1].set_ylabel("skill vs pooled"); ax[1].set_title("B. Skill vs population-average")
    # counts + recapture
    ax[2].bar(range(3), [int((grp == g).sum()) for g in groups], color="#8172b3", alpha=0.7, label="clones")
    ax[2].bar(range(3), [int((surv & (grp == g)).sum()) for g in groups], color="#4c72b0", label="recaptured")
    ax[2].set_xticks(range(3)); ax[2].set_xticklabels(groups); ax[2].set_yscale("log"); ax[2].set_ylabel("count (log)")
    ax[2].set_title("C. Sample size & recapture"); ax[2].legend(fontsize=8)
    fig.suptitle("Migration by clone-size group (singleton n=1 / small 2-49 / large >=50)")
    fig.tight_layout(); f8 = os.path.join(outdir, "size_groups.png"); fig.savefig(f8, dpi=140); plt.close(fig)
    written.append(f8)

    # ===================== FIG 9: dominant-destination confusion matrix ======== #
    sv = np.where(surv)[0]; mv = np.where(mov)[0]
    def confusion(o, pr):
        C = np.zeros((S, S))
        np.add.at(C, (o, pr), 1)
        return C
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for k, (sel, ttl) in enumerate([(sv, "all survivors"), (mv, "movers only")]):
        o = obs_tm[sel].argmax(1); pr = prd_tm[sel].argmax(1)
        C = confusion(o, pr); Cn = C / np.maximum(C.sum(1, keepdims=True), 1)
        acc = np.trace(C) / max(C.sum(), 1)
        im = ax[k].imshow(Cn, cmap="Blues", vmin=0, vmax=1)
        for i in range(S):
            for j in range(S):
                ax[k].text(j, i, f"{Cn[i, j]:.2f}\n({int(C[i, j])})", ha="center", va="center",
                           fontsize=8, color="w" if Cn[i, j] > 0.5 else "k")
        ax[k].set_xticks(range(S)); ax[k].set_xticklabels(tis); ax[k].set_yticks(range(S)); ax[k].set_yticklabels(tis)
        ax[k].set_xlabel("predicted dominant destination"); ax[k].set_ylabel("observed dominant destination")
        ax[k].set_title(f"{ttl}  (accuracy={acc:.2f}, n={len(o)})")
        fig.colorbar(im, ax=ax[k], fraction=0.046)
    fig.suptitle("Dominant-destination confusion (row-normalized = recall; raw counts in parentheses)")
    fig.tight_layout(); f9 = os.path.join(outdir, "confusion_dominant_dest.png"); fig.savefig(f9, dpi=140); plt.close(fig)
    written.append(f9)

    # ===================== FIG 10: per-route flow-volume with CIs =============== #
    # fraction of ALL trafficked descendant mass carried by each route a->b (which routes dominate).
    def flow(tm, sel):
        F = np.zeros((S, S))
        for a in range(S):
            ss_ = sel[src_tis[sel] == a]
            if ss_.size:
                F[a] = tm[ss_].sum(0)
        return F / max(F.sum(), 1e-12)
    F_obs = flow(obs_tm, sv)
    o_lo = np.zeros((S, S)); o_hi = np.zeros((S, S))
    for a in range(S):
        for b in range(S):
            lo, hi = cboot_ci(lambda idx, aa=a, bb=b: flow(obs_tm, idx)[aa, bb], sv, n_boot=200)
            o_lo[a, b], o_hi[a, b] = lo, hi
    Fd = np.empty((len(idx), S, S))
    for s_, si in enumerate(idx):
        mu = np.where(obsv, D * (Xt @ fit.samples[si]), 0.0).reshape(J, S, K).sum(2)
        Fd[s_] = flow(mu, sv)
    Fm = Fd.mean(0); Fdlo = np.quantile(Fd, 0.025, 0); Fdhi = np.quantile(Fd, 0.975, 0)
    routes = [(a, b) for a in range(S) for b in range(S)]
    xs = np.arange(len(routes)); w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    o_pt = [F_obs[a, b] for a, b in routes]; p_pt = [Fm[a, b] for a, b in routes]
    ax.bar(xs - w / 2, o_pt, w, color="#8172b3", label="observed",
           yerr=[[max(0, o_pt[i] - o_lo[a, b]) for i, (a, b) in enumerate(routes)],
                 [max(0, o_hi[a, b] - o_pt[i]) for i, (a, b) in enumerate(routes)]], capsize=3)
    ax.bar(xs + w / 2, p_pt, w, color="#dd8452", label="predicted (95% CrI)",
           yerr=[[max(0, p_pt[i] - Fdlo[a, b]) for i, (a, b) in enumerate(routes)],
                 [max(0, Fdhi[a, b] - p_pt[i]) for i, (a, b) in enumerate(routes)]], capsize=3)
    ax.set_xticks(xs); ax.set_xticklabels([f"{tis[a]}->{tis[b]}" for a, b in routes], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("fraction of total trafficked mass"); ax.legend()
    ax.set_title("Per-route flow volume: observed (patient-cluster CI) vs predicted (posterior CrI)")
    fig.tight_layout(); f10 = os.path.join(outdir, "flow_volume_by_route.png"); fig.savefig(f10, dpi=140); plt.close(fig)
    written.append(f10)

    # ===================== FIG 11: prediction-entropy maps ===================== #
    pfrac = prd_tm / np.maximum(prd_tm.sum(1, keepdims=True), 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(pfrac > 0, pfrac * np.log2(pfrac), 0.0), 1)   # [0, log2(3)]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    # A: mean predicted entropy by source tissue x size bin
    H = np.full((S, len(ABL)), np.nan)
    for a in range(S):
        for j, lb in enumerate(ABL):
            sel = surv & (src_tis == a) & (abin == lb)
            if sel.sum() >= 5:
                H[a, j] = ent[sel].mean()
    im = ax[0].imshow(H, cmap="magma", vmin=0, vmax=np.log2(S), aspect="auto")
    ax[0].set_yticks(range(S)); ax[0].set_yticklabels(tis); ax[0].set_xticks(range(len(ABL))); ax[0].set_xticklabels(ABL)
    ax[0].set_xlabel("source clone size"); ax[0].set_ylabel("source tissue")
    for a in range(S):
        for j in range(len(ABL)):
            if np.isfinite(H[a, j]):
                ax[0].text(j, a, f"{H[a, j]:.2f}", ha="center", va="center", fontsize=8,
                           color="w" if H[a, j] < 0.8 else "k")
    ax[0].set_title("A. Predicted destination entropy (bits; high = uncertain)")
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    # B: does confidence (low entropy) track correctness? dominant-dest accuracy by entropy bin
    ebins = np.linspace(0, ent[surv].max() + 1e-9, 7); em = 0.5 * (ebins[1:] + ebins[:-1])
    correct = (obs_tm.argmax(1) == prd_tm.argmax(1)).astype(int)
    acc, lo_, hi_, cnt = [], [], [], []
    for i in range(len(ebins) - 1):
        inb = surv & (ent >= ebins[i]) & (ent < ebins[i + 1])
        pv, l, h = wilson(int(correct[inb].sum()), int(inb.sum()))
        acc.append(pv); lo_.append(pv - l if np.isfinite(pv) else np.nan); hi_.append(h - pv if np.isfinite(pv) else np.nan); cnt.append(int(inb.sum()))
    ax[1].errorbar(em, acc, yerr=[lo_, hi_], fmt="o-", color="#4c72b0", capsize=3)
    for x, y, n in zip(em, acc, cnt):
        if np.isfinite(y):
            ax[1].text(x, y, f" {n}", fontsize=7, va="bottom")
    ax[1].set_xlabel("predicted destination entropy (bits)"); ax[1].set_ylabel("dominant-destination accuracy")
    ax[1].set_ylim(0, 1.02); ax[1].set_title("B. Confidence calibration (low entropy should = high accuracy)")
    fig.suptitle("Migration prediction uncertainty (entropy of the predicted destination distribution)")
    fig.tight_layout(); f11 = os.path.join(outdir, "prediction_entropy.png"); fig.savefig(f11, dpi=140); plt.close(fig)
    written.append(f11)

    # ===================== FIG 12: per-clone P(leave) with posterior CrI ======= #
    # propagate the posterior M-draws to a per-clone leave-probability distribution ->
    # 95% credible interval per clone (parameter uncertainty), sorted, with observed outcome.
    svv = np.where(surv)[0]
    Xs = Xt[svv]; Ds = D[svv]; ov = obsv[svv]; st_sv = src_tis[svv]
    PL = np.empty((len(idx), svv.size))
    for s_, si in enumerate(idx):
        mu = np.where(ov, Ds * (Xs @ fit.samples[si]), 0.0).reshape(svv.size, S, K).sum(2)
        muf = mu / np.maximum(mu.sum(1, keepdims=True), 1e-12)
        PL[s_] = 1 - muf[np.arange(svv.size), st_sv]
    plmean = PL.mean(0); pllo = np.quantile(PL, 0.025, 0); plhi = np.quantile(PL, 0.975, 0)
    lv = leave_lbl[svv]; order = np.argsort(plmean); xr = np.arange(svv.size)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.fill_between(xr, pllo[order], plhi[order], color="#4c72b0", alpha=0.25, label="95% posterior CrI")
    ax.plot(xr, plmean[order], color="#25406b", lw=1, label="posterior mean P(leave)")
    o = lv[order]
    # density histograms of each class along the sorted axis, in the top/bottom margins (behind the strips)
    nb = 80
    hl, edges = np.histogram(xr[o == 1], bins=nb, range=(0, svv.size))
    hs, _ = np.histogram(xr[o == 0], bins=nb, range=(0, svv.size))
    ctr = 0.5 * (edges[1:] + edges[:-1]); bw = edges[1] - edges[0]; scl = 0.22
    hl_ = hl / max(hl.max(), 1) * scl; hs_ = hs / max(hs.max(), 1) * scl
    ax.bar(ctr, hl_, width=bw * 0.95, bottom=1.0, color="#c44e52", alpha=0.30, edgecolor="none", zorder=1)
    ax.bar(ctr, hs_, width=bw * 0.95, bottom=-hs_, color="#55a868", alpha=0.30, edgecolor="none", zorder=1)
    ax.scatter(xr[o == 1], np.full(int((o == 1).sum()), 1.03), s=1.2, color="#c44e52", zorder=3, label="observed: left")
    ax.scatter(xr[o == 0], np.full(int((o == 0).sum()), -0.03), s=1.2, color="#55a868", zorder=3, label="observed: stayed")
    ax.axhline(1.0, color="#dddddd", lw=0.6); ax.axhline(0.0, color="#dddddd", lw=0.6)
    ax.set_ylim(-0.30, 1.30); ax.set_xlabel("survivor clones (sorted by posterior mean P(leave))")
    ax.set_ylabel("P(leave source tissue)")
    ax.set_title(f"Per-clone leave probability with posterior credible intervals (n={svv.size} survivors)")
    ax.legend(fontsize=8, loc="center left")
    fig.tight_layout(); f12 = os.path.join(outdir, "leave_probability_perclone.png"); fig.savefig(f12, dpi=140); plt.close(fig)
    written.append(f12)

    print("wrote:")
    for w in written:
        print("  " + w)


if __name__ == "__main__":
    main()
