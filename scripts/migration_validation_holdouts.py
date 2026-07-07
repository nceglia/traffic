"""Migration validation inventory -- batch 2: holdout influence + out-of-sample skill.

Answers the patient/timepoint-dependency question by letting the holdouts do the work:
  holdout_influence   : per holdout, how much withholding it moves each tissue->tissue
                        route of the operator (full - withheld) -- which data is load-bearing.
  skill_by_holdout    : out-of-sample migration skill (vs pooled) + stay/go AUC on each
                        holdout's never-seen test slice, with CI -- generalization.

Env: TRAFFIC_H5 (required), FULL_FIT (default nb_global_putative.npz),
     SCORE_DIR (holdout npz dir), OUTDIR (default figures/validation/migration).
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from migration_validation import boot_ci, auc  # noqa: E402
from predictability_breakdown import decompose_prediction, _js_rows  # noqa: E402

from traffic import data, io, scoring, statespace  # noqa: E402
from traffic.readouts import decompose, tissue_traffic  # noqa: E402


def _phi(fit):
    d = fit.dispersion
    if not d:                      # Poisson -> no overdispersion
        return np.inf
    return float(np.exp(np.asarray(d["params"]["log_r"]).mean()))   # global NB2 concentration


def _mig_skill_auc(fit_h, obs, ss, mask, phi_h):
    """OOS migration skill (vs pooled) on movers + stay/go AUC on the test slice."""
    S, K = ss.S, ss.K
    Xt, Y, D = obs.Xtilde[mask], obs.Y[mask], obs.D[mask]
    Jh = Xt.shape[0]
    obsv = D > 0; Yt = np.where(obsv, Y, 0.0)
    pi_h = scoring.pooled_pi(obs, ~mask)
    mean_h = np.where(obsv, D * (Xt @ fit_h.M_hat), 0.0)
    mean_pl = np.where(obsv, D * (Xt.sum(1, keepdims=True) * pi_h[None, :]), 0.0)
    m = decompose_prediction(Xt, Y, D, mean_h, ss, phi_h)
    c = decompose_prediction(Xt, Y, D, mean_pl, ss, phi_h)
    surv = m["surv"]
    Xr = obs.X[mask].reshape(Jh, S, K).sum(2)
    src_tis = Xr.argmax(1)
    src_t = Xr / np.maximum(Xr.sum(1, keepdims=True), 1e-12)
    obs_tm = Yt.reshape(Jh, S, K).sum(2)
    obs_t = obs_tm / np.maximum(obs_tm.sum(1, keepdims=True), 1e-12)
    prd_tm = mean_h.reshape(Jh, S, K).sum(2)
    pf = prd_tm / np.maximum(prd_tm.sum(1, keepdims=True), 1e-12)
    mig_ch = np.full(Jh, np.nan)
    if surv.any():
        mig_ch[surv] = _js_rows(src_t[surv], obs_t[surv])
    mov = surv & (mig_ch > 0.05)
    # skill vs pooled on movers
    b = np.where(mov & np.isfinite(m["migration"]) & np.isfinite(c["migration"]))[0]
    if b.size >= 10:
        sk = 1 - np.median(m["migration"][b]) / max(np.median(c["migration"][b]), 1e-9)
        slo, shi = boot_ci(lambda ii: 1 - np.median(m["migration"][b[ii]]) / max(np.median(c["migration"][b[ii]]), 1e-9), b.size)
    else:
        sk = slo = shi = np.nan
    # stay/go AUC (survivors)
    sv = np.where(surv)[0]
    if sv.size >= 20:
        pleave = 1 - pf[np.arange(Jh), src_tis]
        leave = (obs_tm[np.arange(Jh), src_tis] < obs_tm.sum(1) - 1e-9).astype(int)
        au = auc(pleave[sv], leave[sv])
        alo, ahi = boot_ci(lambda ii: auc(pleave[sv[ii]], leave[sv[ii]]), sv.size)
    else:
        au = alo = ahi = np.nan
    return dict(skill=sk, s_lo=slo, s_hi=shi, auc=au, a_lo=alo, a_hi=ahi,
                n_mov=int(mov.sum()), n_surv=int(surv.sum()))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h5 = os.environ.get("TRAFFIC_H5")
    if not h5:
        raise SystemExit("set TRAFFIC_H5")
    full_path = os.environ.get("FULL_FIT", "nb_global_putative.npz")
    sdir = os.environ.get("SCORE_DIR", "holdout_fits/global")
    outdir = os.environ.get("OUTDIR", "figures/validation/migration")
    os.makedirs(outdir, exist_ok=True)

    ss = statespace.default(); S, K = ss.S, ss.K; tis = list(ss.tissues)
    obs = data.build(data.load_obs_table(h5), ss)
    full = io.load_fit(full_path)
    Xr = obs.X.reshape(obs.X.shape[0], S, K).sum(0); rho = Xr / np.maximum(Xr.sum(1, keepdims=True), 1e-12)
    P_full = tissue_traffic(decompose(full.M_hat)[0], ss, rho)

    npzs = sorted(glob.glob(os.path.join(sdir, "*.npz")))
    names, deltas, oos = [], [], []
    for p in npzs:
        fh = io.load_fit(p); name = scoring._short(p)
        Ph = tissue_traffic(decompose(fh.M_hat)[0], ss, rho)
        names.append(name); deltas.append(P_full - Ph)               # influence: full - withheld
        mask, mode = scoring.test_mask(fh.meta, obs)
        if mask.sum() > 0:
            oos.append((name, _mig_skill_auc(fh, obs, ss, mask, _phi(fh))))
        print(f"  {name:16} rel||dP_mig||={np.linalg.norm(P_full - Ph):.3f}  mode={mode}")

    # ===================== FIG: holdout influence on routes ==================== #
    routes = [(a, b) for a in range(S) for b in range(S)]
    Dmat = np.array([[d[a, b] for a, b in routes] for d in deltas])   # [holdouts, 9]
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(names) + 2))
    vmax = float(np.nanpercentile(np.abs(Dmat), 98)) or 0.05
    im = ax.imshow(Dmat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(routes))); ax.set_xticklabels([f"{tis[a]}->{tis[b]}" for a, b in routes], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(routes)):
            if abs(Dmat[i, j]) > vmax * 0.5:
                ax.text(j, i, f"{Dmat[i, j]:+.2f}", ha="center", va="center", fontsize=6.5,
                        color="w" if abs(Dmat[i, j]) > vmax * 0.75 else "k")
    ax.set_title("Holdout influence on migration routes  (full - withheld operator rate)\n"
                 "red = withholding this data LOWERS the route -> that data drives it")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Δ route rate")
    fig.tight_layout(); fa = os.path.join(outdir, "holdout_influence.png"); fig.savefig(fa, dpi=140); plt.close(fig)

    # ===================== FIG: out-of-sample skill + AUC per holdout ========== #
    oos = [o for o in oos if o[1]["n_surv"] >= 20]
    onames = [o[0] for o in oos]; xs = np.arange(len(onames))
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    sk = [o[1]["skill"] for o in oos]
    slo = [max(0, o[1]["skill"] - o[1]["s_lo"]) if np.isfinite(o[1]["s_lo"]) else 0 for o in oos]
    shi = [max(0, o[1]["s_hi"] - o[1]["skill"]) if np.isfinite(o[1]["s_hi"]) else 0 for o in oos]
    ax[0].bar(xs, sk, color="#4c72b0", yerr=[slo, shi], capsize=3)
    ax[0].axhline(0, color="k", lw=0.8); ax[0].set_xticks(xs); ax[0].set_xticklabels(onames, rotation=45, ha="right", fontsize=8)
    ax[0].set_ylabel("OOS migration skill vs pooled"); ax[0].set_ylim(-1.15, 1.15)
    ax[0].set_title("A. Out-of-sample migration skill (movers)")
    au = [o[1]["auc"] for o in oos]
    alo = [max(0, o[1]["auc"] - o[1]["a_lo"]) if np.isfinite(o[1]["a_lo"]) else 0 for o in oos]
    ahi = [max(0, o[1]["a_hi"] - o[1]["auc"]) if np.isfinite(o[1]["a_hi"]) else 0 for o in oos]
    ax[1].bar(xs, au, color="#55a868", yerr=[alo, ahi], capsize=3)
    ax[1].axhline(0.5, color="k", ls=":"); ax[1].set_xticks(xs); ax[1].set_xticklabels(onames, rotation=45, ha="right", fontsize=8)
    ax[1].set_ylabel("OOS stay/go AUC"); ax[1].set_ylim(0.4, 1.0)
    ax[1].set_title("B. Out-of-sample stay/go discrimination")
    fig.suptitle("Migration generalization across held-out patients / timepoints (95% bootstrap CI)")
    fig.tight_layout(); fb = os.path.join(outdir, "skill_by_holdout.png"); fig.savefig(fb, dpi=140); plt.close(fig)

    print(f"wrote:\n  {fa}\n  {fb}")


if __name__ == "__main__":
    main()
