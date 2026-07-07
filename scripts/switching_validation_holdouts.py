"""Phenotypic-switching validation -- batch 2: holdout influence + out-of-sample skill.

Mirror of migration_validation_holdouts for the WITHIN-TISSUE phenotype axis:
  holdout_influence : per holdout, how much withholding it moves each tissue's within-tissue
                      switching operator H[a] (Frobenius ||H_full - H_held||) -- which data
                      is load-bearing for switching.
  skill_by_holdout  : out-of-sample switching skill (vs pooled) on switchers + stay/switch AUC
                      on each holdout's never-seen test slice -- generalization.

REQUIRES holdout fits on the SAME statespace as the full fit (currently putative = 39 states,
3x13). The existing holdout_fits/ are 33-state (old 11-phenotype) and will NOT match -- regen
holdouts on the putative dataset first (GPU, like the temporal fits).

Env: TRAFFIC_H5 (required), FULL_FIT (default nb_global_putative.npz),
     SCORE_DIR (holdout npz dir), OUTDIR (default figures/validation/switching).
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from migration_validation import boot_ci, auc  # noqa: E402
from predictability_breakdown import decompose_prediction, _js_rows  # noqa: E402

from traffic import data, io, scoring, statespace  # noqa: E402
from traffic.readouts import decompose, within_tissue_switching  # noqa: E402


def _phi(fit):
    d = fit.dispersion
    if not d:                      # Poisson -> no overdispersion
        return np.inf
    return float(np.exp(np.asarray(d["params"]["log_r"]).mean()))   # global NB2 concentration


def _switch_skill_auc(fit_h, obs, ss, mask, phi_h):
    """OOS switching skill (vs pooled) on switchers + stay/switch AUC on the test slice."""
    S, K = ss.S, ss.K
    Xt, Y, D = obs.Xtilde[mask], obs.Y[mask], obs.D[mask]
    Jh = Xt.shape[0]; aj = np.arange(Jh)
    obsv = D > 0; Yt = np.where(obsv, Y, 0.0)
    pi_h = scoring.pooled_pi(obs, ~mask)
    mean_h = np.where(obsv, D * (Xt @ fit_h.M_hat), 0.0)
    mean_pl = np.where(obsv, D * (Xt.sum(1, keepdims=True) * pi_h[None, :]), 0.0)
    m = decompose_prediction(Xt, Y, D, mean_h, ss, phi_h)
    c = decompose_prediction(Xt, Y, D, mean_pl, ss, phi_h)

    src_tis = Xt.reshape(Jh, S, K).sum(2).argmax(1)
    blk = lambda A: A.reshape(Jh, S, K)[aj, src_tis, :]
    src_cnt = obs.X[mask].reshape(Jh, S, K)[aj, src_tis, :]
    obs_ph, prd_ph = blk(Yt), blk(mean_h)
    nrm = lambda A: A / np.maximum(A.sum(1, keepdims=True), 1e-12)
    obs_n, prd_n, src_n = nrm(obs_ph), nrm(prd_ph), nrm(src_cnt)
    surv_sw = obs_ph.sum(1) > 0
    src_phe = src_cnt.argmax(1)
    swi_ch = np.full(Jh, np.nan)
    if surv_sw.any():
        swi_ch[surv_sw] = _js_rows(src_n[surv_sw], obs_n[surv_sw])
    mov = surv_sw & (swi_ch > 0.05)

    b = np.where(mov & np.isfinite(m["switching"]) & np.isfinite(c["switching"]))[0]
    if b.size >= 10:
        sk = 1 - np.median(m["switching"][b]) / max(np.median(c["switching"][b]), 1e-9)
        slo, shi = boot_ci(lambda ii: 1 - np.median(m["switching"][b[ii]]) / max(np.median(c["switching"][b[ii]]), 1e-9), b.size)
    else:
        sk = slo = shi = np.nan
    sv = np.where(surv_sw)[0]
    if sv.size >= 20:
        pswitch = 1 - prd_n[aj, src_phe]
        switch_lbl = (obs_n.argmax(1) != src_phe).astype(int)
        au = auc(pswitch[sv], switch_lbl[sv])
        alo, ahi = boot_ci(lambda ii: auc(pswitch[sv[ii]], switch_lbl[sv[ii]]), sv.size)
    else:
        au = alo = ahi = np.nan
    return dict(skill=sk, s_lo=slo, s_hi=shi, auc=au, a_lo=alo, a_hi=ahi,
                n_mov=int(mov.sum()), n_surv=int(surv_sw.sum()))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h5 = os.environ.get("TRAFFIC_H5")
    if not h5:
        raise SystemExit("set TRAFFIC_H5")
    full_path = os.environ.get("FULL_FIT", "nb_global_putative.npz")
    sdir = os.environ.get("SCORE_DIR", "holdout_fits/global")
    outdir = os.environ.get("OUTDIR", "figures/validation/switching")
    os.makedirs(outdir, exist_ok=True)

    ss = statespace.default(); S, K = ss.S, ss.K; tis = list(ss.tissues)
    obs = data.build(data.load_obs_table(h5), ss)
    full = io.load_fit(full_path)
    if full.M_hat.shape[0] != ss.L:
        raise SystemExit(f"full fit is {full.M_hat.shape[0]} states but statespace is {ss.L} "
                         f"-- regen the fit/holdouts on the current dataset.")
    H_full = within_tissue_switching(decompose(full.M_hat)[0], ss)

    npzs = sorted(glob.glob(os.path.join(sdir, "*.npz")))
    names, infl, oos = [], [], []
    for p in npzs:
        fh = io.load_fit(p)
        if fh.M_hat.shape[0] != ss.L:
            print(f"  SKIP {os.path.basename(p)}: {fh.M_hat.shape[0]}-state (need {ss.L})"); continue
        name = scoring._short(p)
        Hh = within_tissue_switching(decompose(fh.M_hat)[0], ss)
        names.append(name)
        infl.append([float(np.linalg.norm(H_full[a] - Hh[a])) for a in range(S)])   # per-tissue operator shift
        mask, mode = scoring.test_mask(fh.meta, obs)
        if mask.sum() > 0:
            oos.append((name, _switch_skill_auc(fh, obs, ss, mask, _phi(fh))))
        print(f"  {name:16} ||dH||/tissue={np.round(infl[-1],3)}  mode={mode}")

    if not names:
        raise SystemExit("no statespace-matching holdout fits found -- regen holdouts on this dataset.")

    # ===================== FIG: holdout influence on switching operator ========= #
    Imat = np.array(infl)                                            # [holdouts, S]
    fig, ax = plt.subplots(figsize=(6, 0.5 * len(names) + 2))
    im = ax.imshow(Imat, cmap="magma", aspect="auto")
    ax.set_xticks(range(S)); ax.set_xticklabels(tis)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    for i in range(len(names)):
        for a in range(S):
            ax.text(a, i, f"{Imat[i, a]:.2f}", ha="center", va="center", fontsize=7,
                    color="w" if Imat[i, a] < Imat.max() * 0.6 else "k")
    ax.set_title("Holdout influence on within-tissue switching operator\n||H_full - H_withheld|| per tissue (larger = that data drives switching)")
    fig.colorbar(im, ax=ax, fraction=0.04, label="operator shift")
    fig.tight_layout(); fa = os.path.join(outdir, "holdout_influence.png"); fig.savefig(fa, dpi=140); plt.close(fig)

    # ===================== FIG: OOS switching skill + stay/switch AUC =========== #
    oos = [o for o in oos if o[1]["n_surv"] >= 20]
    onames = [o[0] for o in oos]; xs = np.arange(len(onames))
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    sk = [o[1]["skill"] for o in oos]
    slo = [max(0, o[1]["skill"] - o[1]["s_lo"]) if np.isfinite(o[1]["s_lo"]) else 0 for o in oos]
    shi = [max(0, o[1]["s_hi"] - o[1]["skill"]) if np.isfinite(o[1]["s_hi"]) else 0 for o in oos]
    ax[0].bar(xs, sk, color="#dd8452", yerr=[slo, shi], capsize=3)
    ax[0].axhline(0, color="k", lw=0.8); ax[0].set_xticks(xs); ax[0].set_xticklabels(onames, rotation=45, ha="right", fontsize=8)
    ax[0].set_ylabel("OOS switching skill vs pooled"); ax[0].set_ylim(-1.15, 1.15)
    ax[0].set_title("A. Out-of-sample switching skill (switchers)")
    au = [o[1]["auc"] for o in oos]
    alo = [max(0, o[1]["auc"] - o[1]["a_lo"]) if np.isfinite(o[1]["a_lo"]) else 0 for o in oos]
    ahi = [max(0, o[1]["a_hi"] - o[1]["auc"]) if np.isfinite(o[1]["a_hi"]) else 0 for o in oos]
    ax[1].bar(xs, au, color="#55a868", yerr=[alo, ahi], capsize=3)
    ax[1].axhline(0.5, color="k", ls=":"); ax[1].set_xticks(xs); ax[1].set_xticklabels(onames, rotation=45, ha="right", fontsize=8)
    ax[1].set_ylabel("OOS stay/switch AUC"); ax[1].set_ylim(0.4, 1.0)
    ax[1].set_title("B. Out-of-sample stay/switch discrimination")
    fig.suptitle("Switching generalization across held-out patients / timepoints (95% bootstrap CI)")
    fig.tight_layout(); fb = os.path.join(outdir, "skill_by_holdout.png"); fig.savefig(fb, dpi=140); plt.close(fig)

    print(f"wrote:\n  {fa}\n  {fb}")


if __name__ == "__main__":
    main()
