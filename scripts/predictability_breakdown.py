"""Where can we predict next-timepoint clonal behavior, and where can't we?

Scores the global fit's one-step-ahead prediction for every forward transition
and decomposes the destination prediction into three biologically distinct axes:

  expansion   within-SOURCE-TISSUE count fold-change (predicted vs observed count in the
              clone's source tissue) -- did the clone grow/shrink within its own tissue.
              Restricting to the source tissue cancels that tissue's depth in the ratio, so
              expansion is comparable across tissues (a within-tissue share fold-change, not
              absolute fecundity) and not entangled with cross-tissue sampling depth.
  migration   the destination *tissue marginal*  c^tis_j(s) = sum_k c_j(s,k)
              -- did cells redistribute across PBMC/CSF/TP as predicted.
  switching   the *within-tissue phenotype* composition  c_j(k|s), averaged over
              tissues weighted by observed tissue mass -- did phenotypes convert
              as predicted, holding tissue fixed.

Migration + switching are the exact KL chain-rule split of the destination
composition divergence  D(c_obs || c_pred) = D(tissue marginal) + E[D(pheno|tissue)]
(here computed with the bounded Jensen-Shannon analogue per piece); expansion is
the orthogonal magnitude axis. Each axis is scored against two nulls --
static (M = identity: "cells stay put") and pooled (population-average mean
composition) -- so "skill" = how far the model beats the relevant null.

Every axis is then stratified by clone abundance, patient, source tissue, and
source timepoint to show *where* prediction is trustworthy and where it is not.
Survivors only (clones recaptured at t+1) define the shape axes; recapture rate
is reported separately as the dominant predictability limit for rare clones.

Env:
  TRAFFIC_H5   dataset path (required)
  FIT          global fit npz (default nb_global_putative.npz)
  OUTDIR       output dir (default holdout_fits_putative/global/scoring/predictability)
"""
import csv
import os

import numpy as np

from traffic import data, io, statespace
from traffic.readouts import _js
from traffic.scoring import _wls_slope_se, _interval_cover

ABUND_LABELS = ["1", "2", "3-4", "5-9", "10-49", "50+"]


def abundance_bin(n):
    n = int(n)
    if n <= 1:
        return "1"
    if n <= 2:
        return "2"
    if n <= 4:
        return "3-4"
    if n <= 9:
        return "5-9"
    if n <= 49:
        return "10-49"
    return "50+"


def _js_rows(P, Q, eps=1e-12):
    """Row-wise Jensen-Shannon divergence (base 2, in [0,1]); P,Q: [n,d]."""
    P = P + eps; P = P / P.sum(1, keepdims=True)
    Q = Q + eps; Q = Q / Q.sum(1, keepdims=True)
    M = 0.5 * (P + Q)
    kl = lambda A, B: np.sum(A * np.log2(A / B), axis=1)
    return 0.5 * kl(P, M) + 0.5 * kl(Q, M)


def decompose_prediction(Xt, Y, D, mean, ss, phi):
    """Per-clone expansion / migration / switching error for a predicted mean.

    expansion : the within-SOURCE-TISSUE count magnitude, scored three ways over ALL
                profiled transitions (extinct included, NO survival conditioning -- which
                would impose recapture selection: deep samples surface low-share clones
                whose survivors are growers, biasing any survivor-conditioned metric):
                  exp_cov : coverage -- is obs_s inside the 90% NB predictive interval
                            (moment-matched to pred_s, Var_s). HEADLINE calibration; want ~0.90.
                  exp_dss : Dawid-Sebastiani proper score (obs_s-pred_s)^2/Var_s + log Var_s,
                            lower=better; compared to a null as a DIFFERENCE (not a 1-ratio,
                            since DSS is not non-negative). The proper skill-vs-null metric.
                  expansion : |standardized NB residual| |z| = |obs_s-pred_s|/sqrt(Var_s).
                            DIAGNOSTIC ONLY (kept for the depth-calibration gate); not a
                            proper score (a model can shrink it by inflating its variance).
                Var_s = pred_s + (Σ_k mean_k^2)/phi: obs_s = Σ_k y_k is a sum of K
                independent per-state NB2(mean_k, phi), so its variance is the SUM of
                per-state variances -- NOT pred_s + pred_s^2/phi.
    migration : JS on the destination tissue marginal (survivors -- a composition is
                undefined for an extinct clone).
    switching : tissue-mass-weighted JS of within-tissue phenotype composition (survivors).

    Returns dict of length-J arrays (NaN where undefined) plus masks `surv` (recaptured)
    and `prof_src` (source tissue profiled at t+1 -> expansion defined there).
    """
    S, K, L = ss.S, ss.K, ss.L
    J = Xt.shape[0]
    obsv = D > 0
    Yt = np.where(obsv, Y, 0.0)
    mean = np.where(obsv, mean, 0.0)
    N_obs = Yt.sum(1)
    N_pred = mean.sum(1)
    surv = N_obs > 0

    Yr = Yt.reshape(J, S, K)
    Mr = mean.reshape(J, S, K)
    Yt_tis = Yr.sum(2)            # [J,S] observed tissue mass
    Mt_tis = Mr.sum(2)            # [J,S] predicted tissue mass

    # expansion: UNCONDITIONAL standardized NB residual on the within-source-tissue count.
    # Restricting to the clone's dominant source tissue cancels that tissue's depth in the
    # comparison; standardizing by the NB sd makes it depth-calibrated; keeping extinct clones
    # (no `surv` filter -- only "source tissue profiled") removes the recapture selection that
    # made the old survivor-conditioned |log-ratio| slope with depth even when the fit is fine.
    src_tis = Xt.reshape(J, S, K).sum(2).argmax(1)         # [J] dominant source tissue
    prof_src = (D.reshape(J, S, K).sum(2) > 0)[np.arange(J), src_tis]   # source tissue sequenced at t+1
    pred_s = Mt_tis[np.arange(J), src_tis]                 # predicted dest count in source tissue
    obs_s = Yt_tis[np.arange(J), src_tis]                  # observed  dest count in source tissue
    # obs_s = Σ_k y_k is a sum of K independent per-state NB2(mean_k, phi), so its variance is the
    # SUM of per-state variances pred_s + (Σ_k mean_k^2)/phi -- NOT pred_s + pred_s^2/phi (the
    # variance of a single NB2 at the aggregated mean, which overstates it by the cross terms).
    m2_s = (Mr[np.arange(J), src_tis, :] ** 2).sum(1)      # Σ_k mean_k^2 in the source tissue
    var_s = np.maximum(pred_s + m2_s / phi, 1e-9)
    sd_s = np.sqrt(var_s)
    # |z|: diagnostic only (feeds the depth-calibration gate; not a proper score).
    expansion = np.full(J, np.nan)
    expansion[prof_src] = np.abs((obs_s[prof_src] - pred_s[prof_src]) / sd_s[prof_src])
    # DSS (proper score, lower=better) + 90% interval coverage (headline calibration).
    exp_dss = np.full(J, np.nan)
    exp_dss[prof_src] = ((obs_s[prof_src] - pred_s[prof_src]) ** 2 / var_s[prof_src]
                         + np.log(var_s[prof_src]))
    exp_cov = np.full(J, np.nan)
    exp_cov[prof_src] = _interval_cover(obs_s[prof_src], pred_s[prof_src],
                                        var_s[prof_src]).astype(float)

    # migration: JS on the tissue marginal (survivors)
    migration = np.full(J, np.nan)
    if surv.any():
        migration[surv] = _js_rows(Yt_tis[surv], Mt_tis[surv])

    # switching: tissue-mass-weighted JS of within-tissue phenotype composition
    switching = np.full(J, np.nan)
    sw_acc = np.zeros(J); w_acc = np.zeros(J)
    for s in range(S):
        mo = Yr[:, s, :].sum(1)          # observed mass in tissue s
        mp = Mr[:, s, :].sum(1)          # predicted mass in tissue s
        ok = (mo > 0) & (mp > 0)
        if ok.any():
            js_s = _js_rows(Yr[ok, s, :], Mr[ok, s, :])
            sw_acc[ok] += mo[ok] * js_s
            w_acc[ok] += mo[ok]
    have = w_acc > 0
    switching[have] = sw_acc[have] / w_acc[have]

    return {"N_obs": N_obs, "N_pred": N_pred, "surv": surv, "prof_src": prof_src,
            "expansion": expansion, "exp_dss": exp_dss, "exp_cov": exp_cov,
            "migration": migration, "switching": switching}


def stratum_summary(key, order, model_err, null_err, base_mask):
    """Per-stratum median model & null error and skill = 1 - med_model/med_null."""
    rows = []
    for lab in order:
        sel = base_mask & (key == lab) & np.isfinite(model_err)
        n = int(sel.sum())
        mm = float(np.median(model_err[sel])) if n else np.nan
        seln = base_mask & (key == lab) & np.isfinite(null_err)
        mn = float(np.median(null_err[seln])) if seln.sum() else np.nan
        # skill undefined when the null is ~perfect (mn≈0, e.g. the static null on
        # the no-change majority) -- guard so it doesn't blow up.
        skill = (1 - mm / mn) if (np.isfinite(mm) and np.isfinite(mn) and mn > 1e-4) else np.nan
        rows.append({"stratum": str(lab), "n": n, "model": mm, "null": mn, "skill": skill})
    return rows


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h5 = os.environ.get("TRAFFIC_H5")
    if not h5:
        raise SystemExit("set TRAFFIC_H5")
    fit_path = os.environ.get("FIT", "nb_global_putative.npz")
    outdir = os.environ.get("OUTDIR", "holdout_fits_putative/global/scoring/predictability")
    os.makedirs(outdir, exist_ok=True)

    ss = statespace.default()
    obs = data.build(data.load_obs_table(h5), ss)
    fit = io.load_fit(fit_path)
    S, K, L = ss.S, ss.K, ss.L
    Xt, Y, D = obs.Xtilde, obs.Y, obs.D
    J = Xt.shape[0]
    phi = float(np.exp(np.asarray(fit.dispersion["params"]["log_r"]).mean())) if fit.dispersion else np.inf
    print(f"fit={fit_path}  J={J}  L={L}  tissues={ss.tissues}  phi={phi:.3f}")

    # predicted destination means: model (M_hat), static null (M=I), pooled null (pi)
    obsv = D > 0
    Yt = np.where(obsv, Y, 0.0)
    pi = Yt.sum(0); pi = pi / max(pi.sum(), 1e-12)        # pooled population destination comp [L]
    mean_model = np.where(obsv, D * (Xt @ fit.M_hat), 0.0)
    mean_static = np.where(obsv, D * Xt, 0.0)
    mean_pooled = np.where(obsv, D * (Xt.sum(1, keepdims=True) * pi[None, :]), 0.0)

    m = decompose_prediction(Xt, Y, D, mean_model, ss, phi)
    p = decompose_prediction(Xt, Y, D, mean_static, ss, phi)    # static (no-change) null
    c = decompose_prediction(Xt, Y, D, mean_pooled, ss, phi)    # pooled population-average null

    # strata
    n_src = obs.n_src.astype(int)
    abin = np.array([abundance_bin(n) for n in n_src])
    patient = np.asarray(obs.patient)
    src_tp = obs.src_tp.astype(int)
    Xr_tis = obs.X.reshape(J, S, K).sum(2)               # [J,S] source tissue mass
    src_tissue = np.array([ss.tissues[i] for i in Xr_tis.argmax(1)])
    pat_order = sorted(set(patient.tolist()))
    tis_order = list(ss.tissues)
    tp_order = sorted(set(src_tp.tolist()))

    surv = m["surv"]
    # net growth direction accuracy (survivors): predicted vs observed expand/contract
    obs_lg = np.log((m["N_obs"] + 0.5) / (n_src + 0.5))
    pred_lg = np.log((m["N_pred"] + 0.5) / (n_src + 0.5))
    dir_ok = (np.sign(obs_lg) == np.sign(pred_lg))

    # multi-tissue destination flag (migration is only "interesting" off-diagonal)
    dest_ntis = (Yt.reshape(J, S, K).sum(2) > 0).sum(1)

    # --- model-independent "did the behavior actually occur?" signals ---------- #
    # Measured from source vs observed destination composition. We score each axis
    # on the subset where the behavior is present, so skill reflects events of
    # interest, not the no-change majority (which the static null predicts for
    # free). Thresholds are deliberately mild (JS>0.05; >2x size change).
    src_tis = Xr_tis / np.maximum(Xr_tis.sum(1, keepdims=True), 1e-12)        # [J,S]
    obs_tis = Yt.reshape(J, S, K).sum(2)
    obs_tis = obs_tis / np.maximum(obs_tis.sum(1, keepdims=True), 1e-12)
    src_phe = obs.X.reshape(J, S, K).sum(1)
    src_phe = src_phe / np.maximum(src_phe.sum(1, keepdims=True), 1e-12)      # [J,K]
    obs_phe = Yt.reshape(J, S, K).sum(1)
    obs_phe = obs_phe / np.maximum(obs_phe.sum(1, keepdims=True), 1e-12)
    mig_change = np.full(J, np.nan); swi_change = np.full(J, np.nan)
    if surv.any():
        mig_change[surv] = _js_rows(src_tis[surv], obs_tis[surv])
        swi_change[surv] = _js_rows(src_phe[surv], obs_phe[surv])
    # expansion is now an UNCONDITIONAL calibration residual, so it has no "event" sub-selection:
    # its base and mover masks are both "source tissue profiled" (all transitions we can score).
    # migration/switching keep the survivor base and a mover threshold (their no-change null is
    # trivially right on the non-mover majority, so skill is only meaningful where the event occurs).
    prof_src = m["prof_src"]
    mover = {
        "expansion": prof_src,                                # all profiled (no survival/event filter)
        "migration": surv & (mig_change > 0.05),              # tissue distribution shifted
        "switching": surv & (swi_change > 0.05),              # phenotype distribution shifted
    }
    axis_base = {"expansion": prof_src, "migration": surv, "switching": surv}

    axes = {"expansion": (m["expansion"], p["expansion"], c["expansion"]),
            "migration": (m["migration"], p["migration"], c["migration"]),
            "switching": (m["switching"], p["switching"], c["switching"])}

    # ---------------- CSVs: per-stratum summary for every axis x stratum-kind ----
    strata = {"abundance": (abin, ABUND_LABELS), "patient": (patient, pat_order),
              "source_tissue": (src_tissue, tis_order), "source_tp": (src_tp, tp_order)}
    summ_path = os.path.join(outdir, "predictability_by_stratum.csv")
    with open(summ_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["axis", "stratum_kind", "stratum", "n_survivors", "n_movers",
                    "model_err", "static_err", "pooled_err",
                    "skill_vs_static_all", "skill_vs_pooled_all",
                    "skill_vs_static_movers", "skill_vs_pooled_movers"])
        for ax_name, (em, ep, ec) in axes.items():
            mv = mover[ax_name]
            base = axis_base[ax_name]                            # expansion: profiled; shape axes: survivors
            for kind, (key, order) in strata.items():
                sp = stratum_summary(key, order, em, ep, base)       # all (per-axis base)
                sc = stratum_summary(key, order, em, ec, base)
                spm = stratum_summary(key, order, em, ep, mv)        # movers only
                scm = stratum_summary(key, order, em, ec, mv)
                for rp, rc, rpm, rcm in zip(sp, sc, spm, scm):
                    w.writerow([ax_name, kind, rp["stratum"], rp["n"], rpm["n"],
                                f"{rp['model']:.4f}", f"{rp['null']:.4f}", f"{rc['null']:.4f}",
                                f"{rp['skill']:.4f}", f"{rc['skill']:.4f}",
                                f"{rpm['skill']:.4f}", f"{rcm['skill']:.4f}"])
    print(f"wrote {summ_path}")

    # overall recapture rate by abundance (data-level predictability limit)
    recap = {lab: float((surv & (abin == lab)).sum()) / max((abin == lab).sum(), 1)
             for lab in ABUND_LABELS}

    cmap = plt.get_cmap("viridis")
    abin_colors = {lab: cmap(i / (len(ABUND_LABELS) - 1)) for i, lab in enumerate(ABUND_LABELS)}

    def grouped_box(ax, key, order, em, ep, ylabel, title, base):
        data_m = [em[base & (key == lab) & np.isfinite(em)] for lab in order]
        ax.boxplot(data_m, positions=np.arange(len(order)), widths=0.5,
                   showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="#4c72b0", alpha=0.6))
        med_p = [np.nanmedian(ep[base & (key == lab) & np.isfinite(ep)])
                 if (base & (key == lab)).any() else np.nan for lab in order]
        ax.plot(np.arange(len(order)), med_p, "rD", ms=6, label="static (no change) median")
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels([f"{o}\n(n={int((base&(key==o)).sum())})" for o in order],
                           fontsize=7)
        ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(fontsize=7)

    def by_group_bars(ax, key, order, em, ep, ec, base, ylabel, title):
        med = lambda arr, q: (np.nanmedian(arr[base & (key == q)])
                              if (base & (key == q)).any() else np.nan)
        xx = np.arange(len(order))
        ax.bar(xx - 0.25, [med(em, q) for q in order], 0.25, label="model", color="#4c72b0")
        ax.bar(xx, [med(ep, q) for q in order], 0.25, label="static", color="#c44e52", alpha=0.8)
        ax.bar(xx + 0.25, [med(ec, q) for q in order], 0.25, label="pooled population-average", color="#dd8452", alpha=0.8)
        ax.set_xticks(xx); ax.set_xticklabels([f"{q}\n(n={int((base&(key==q)).sum())})" for q in order], fontsize=7)
        ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(fontsize=7)

    def skill_heat(ax, key, order, em, ep, title, cond):
        rows = [stratum_summary(key, order, em, ep, cond)[i] for i in range(len(order))]
        vals = np.array([r["skill"] for r in rows])[None, :]
        im = ax.imshow(vals, cmap="RdBu", vmin=-0.5, vmax=0.5, aspect="auto")
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(order, fontsize=7, rotation=30, ha="right")
        ax.set_yticks([0]); ax.set_yticklabels(["skill"])
        for i, r in enumerate(rows):
            if np.isfinite(r["skill"]):
                ax.text(i, 0, f"{r['skill']:+.2f}\nn={r['n']}", ha="center", va="center", fontsize=7)
        ax.set_title(title)
        return im

    # ============================ FIGURE 1: EXPANSION ========================== #
    fig, ax = plt.subplots(2, 2, figsize=(13, 11))
    a = ax[0, 0]
    for lab in ABUND_LABELS:
        sel = surv & (abin == lab)
        a.scatter(m["N_obs"][sel] + 1, m["N_pred"][sel] + 1, s=7, alpha=0.3,
                  color=abin_colors[lab], label=lab)
    lim = [1, max(2, np.nanmax(m["N_obs"][surv]) + 1)]
    a.plot(lim, lim, "k--", lw=1); a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("observed destination total (+1)"); a.set_ylabel("predicted total (+1)")
    a.set_title("A. Expansion: predicted vs observed total (survivors, by abundance)")
    a.legend(fontsize=7, title="src clone size", markerscale=2)
    grouped_box(ax[0, 1], abin, ABUND_LABELS, m["expansion"], p["expansion"],
                "|standardized NB residual|  |z|  (lower=better)",
                "B. Within-tissue expansion residual |z| by clone abundance (all profiled)", mover["expansion"])
    by_group_bars(ax[1, 0], patient, pat_order, m["expansion"], p["expansion"], c["expansion"],
                  mover["expansion"], "median |z|", "C. Within-tissue expansion residual |z| by patient (all profiled)")
    d = ax[1, 1]
    rec = [recap[l] for l in ABUND_LABELS]
    diracc = [float(dir_ok[surv & (abin == l)].mean()) if (surv & (abin == l)).any() else np.nan
              for l in ABUND_LABELS]
    xx = np.arange(len(ABUND_LABELS))
    d.bar(xx, rec, 0.6, color="#8172b3", alpha=0.7, label="recapture rate (survival)")
    d.plot(xx, diracc, "go-", label="growth-direction accuracy (survivors)")
    d.set_xticks(xx); d.set_xticklabels(ABUND_LABELS, fontsize=8)
    d.set_ylim(0, 1.02); d.set_xlabel("source clone size")
    d.set_title("D. Recapture & growth-direction accuracy by abundance"); d.legend(fontsize=7)
    fig.suptitle("Expansion (within-tissue magnitude; headline = coverage@90 + DSS, |z| shown as diagnostic) -- global fit one-step prediction", y=0.995)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "expansion.png"), dpi=130); plt.close(fig)
    print(f"wrote {os.path.join(outdir, 'expansion.png')}")

    # ============================ FIGURE 2: SWITCHING ========================= #
    fig, ax = plt.subplots(2, 2, figsize=(13, 11))
    grouped_box(ax[0, 0], abin, ABUND_LABELS, m["switching"], p["switching"],
                "within-tissue phenotype JS (lower=better)",
                "A. Switching error by clone abundance (switchers)", mover["switching"])
    by_group_bars(ax[0, 1], patient, pat_order, m["switching"], p["switching"], c["switching"],
                  mover["switching"], "median phenotype JS", "B. Switching error by patient (switchers)")
    by_group_bars(ax[1, 0], src_tissue, tis_order, m["switching"], p["switching"], c["switching"],
                  mover["switching"], "median phenotype JS", "C. Switching error by source tissue (switchers)")
    im = skill_heat(ax[1, 1], abin, ABUND_LABELS, m["switching"], p["switching"],
                    "D. Switching skill vs static, SWITCHERS only\n(+ = model better, - = worse)",
                    mover["switching"])
    fig.colorbar(im, ax=ax[1, 1], fraction=0.046)
    fig.suptitle("Phenotypic switching (within-tissue phenotype conversion) -- global fit", y=0.995)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "switching.png"), dpi=130); plt.close(fig)
    print(f"wrote {os.path.join(outdir, 'switching.png')}")

    # ============================ FIGURE 3: MIGRATION ========================= #
    fig, ax = plt.subplots(2, 2, figsize=(13, 11))
    a = ax[0, 0]
    frac_multi = [float((dest_ntis[surv & (abin == l)] > 1).mean()) if (surv & (abin == l)).any() else np.nan
                  for l in ABUND_LABELS]
    xx = np.arange(len(ABUND_LABELS))
    a.bar(xx, frac_multi, 0.6, color="#55a868", alpha=0.75)
    a.set_xticks(xx); a.set_xticklabels(ABUND_LABELS, fontsize=8)
    a.set_ylabel("fraction of survivors in >1 tissue at t+1")
    a.set_xlabel("source clone size")
    a.set_title("A. How often migration is even observable (multi-tissue destination)")
    grouped_box(ax[0, 1], abin, ABUND_LABELS, m["migration"], p["migration"],
                "tissue-marginal JS (lower=better)",
                "B. Migration error by clone abundance (migrators)", mover["migration"])
    by_group_bars(ax[1, 0], patient, pat_order, m["migration"], p["migration"], c["migration"],
                  mover["migration"], "median tissue JS", "C. Migration error by patient (migrators)")
    im = skill_heat(ax[1, 1], abin, ABUND_LABELS, m["migration"], p["migration"],
                    "D. Migration skill vs static, MIGRATORS only\n(+ = model better, - = worse)",
                    mover["migration"])
    fig.colorbar(im, ax=ax[1, 1], fraction=0.046)
    fig.suptitle("Migration (cross-tissue redistribution) -- global fit one-step prediction", y=0.995)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "migration.png"), dpi=130); plt.close(fig)
    print(f"wrote {os.path.join(outdir, 'migration.png')}")

    # ===================== FIGURE 4: PREDICTABILITY SUMMARY =================== #
    # master heatmap: rows = strata (abundance bins + patients), cols = 3 axes,
    # color = skill vs static null; a trailing column shows recapture rate.
    row_keys = [("abund", l) for l in ABUND_LABELS] + [("pat", q) for q in pat_order]
    row_labels = [f"size {l}" for l in ABUND_LABELS] + [f"pt {q}" for q in pat_order]
    axis_order = ["expansion", "switching", "migration"]
    skill_grid = np.full((len(row_keys), len(axis_order)), np.nan)
    n_grid = np.zeros((len(row_keys), len(axis_order)), int)
    for ri, (kind, lab) in enumerate(row_keys):
        key = abin if kind == "abund" else patient
        for ci, ax_name in enumerate(axis_order):
            em, ep, _ = axes[ax_name]
            r = stratum_summary(key, [lab], em, ep, mover[ax_name])[0]
            skill_grid[ri, ci] = r["skill"]; n_grid[ri, ci] = r["n"]

    fig, axx = plt.subplots(1, 2, figsize=(12, 8), gridspec_kw={"width_ratios": [3, 1]})
    a = axx[0]
    im = a.imshow(skill_grid, cmap="RdBu", vmin=-0.5, vmax=0.5, aspect="auto")
    a.set_xticks(np.arange(len(axis_order)))
    a.set_xticklabels([s.capitalize() for s in axis_order], fontsize=10)
    a.set_yticks(np.arange(len(row_labels))); a.set_yticklabels(row_labels, fontsize=8)
    for ri in range(len(row_keys)):
        for ci in range(len(axis_order)):
            v = skill_grid[ri, ci]
            if np.isfinite(v):
                a.text(ci, ri, f"{v:+.2f}\nn={n_grid[ri, ci]}", ha="center", va="center",
                       fontsize=6.5, color="k")
    a.axhline(len(ABUND_LABELS) - 0.5, color="k", lw=1.5)
    a.set_title("Predictability skill vs static (no change)\n(expansion: all profiled, |z|; migration/switching: movers only)")
    fig.colorbar(im, ax=a, fraction=0.046, label="skill = 1 - err_model/err_static")
    rax = axx[1]
    rec = [recap[l] for l in ABUND_LABELS] + [np.nan] * len(pat_order)
    yy = np.arange(len(row_labels))
    rax.barh(yy, [r if np.isfinite(r) else 0 for r in rec], color="#8172b3", alpha=0.7)
    rax.set_yticks(yy); rax.set_yticklabels([]); rax.invert_yaxis()
    rax.set_xlim(0, 1); rax.set_xlabel("recapture rate")
    rax.set_title("survival\n(data limit)", fontsize=9)
    a.invert_yaxis()
    fig.suptitle("WHERE can we predict clonal trajectories? -- global fit", y=0.99, fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "predictability_summary.png"), dpi=130)
    plt.close(fig)
    print(f"wrote {os.path.join(outdir, 'predictability_summary.png')}")

    # console headline: overall (all survivors) vs movers-only skill
    print("\n=== axis medians: all (expansion: profiled; shape axes: survivors)  vs  movers-only ===")
    for ax_name, (em, ep, ec) in axes.items():
        mv = mover[ax_name]
        base = axis_base[ax_name]
        def med(arr, msk):
            s = msk & np.isfinite(arr)
            return np.median(arr[s]) if s.any() else np.nan
        sk_all = 1 - med(em, base) / med(ep, base) if med(ep, base) > 1e-4 else np.nan
        sk_mv = 1 - med(em, mv) / med(ep, mv) if med(ep, mv) > 1e-4 else np.nan
        print(f"  {ax_name:10} ALL: model={med(em, base):.3f} static={med(ep, base):.3f} "
              f"pooled={med(ec, base):.3f} skill_vs_static={sk_all:+.2f} (n={int((base&np.isfinite(em)).sum())})")
        print(f"  {'':10} MOVERS: model={med(em, mv):.3f} static={med(ep, mv):.3f} "
              f"pooled={med(ec, mv):.3f} skill_vs_static={sk_mv:+.2f} (n={int((mv&np.isfinite(em)).sum())})")
    print(f"  overall recapture rate = {surv.mean():.3f}")
    # (removed: per-tissue g^within -- a cross-pool magnitude comparison, invalid under
    #  compositional closure; see traffic/readouts.py CONTRACT. Report T-based redistribution
    #  and within-pool RELATIVE share change instead.)

    # expansion magnitude -- HEADLINE metrics: proper score (DSS) + calibration (coverage).
    # |z| above is the depth diagnostic only; these are the reportable expansion numbers.
    # DSS is a proper score (mean+variance) so model-vs-null is a DIFFERENCE null-minus-model
    # (>0 = model better), not a 1-ratio skill (DSS is not non-negative).
    ps = m["prof_src"]
    def _med(a, msk):
        sel = msk & np.isfinite(a)
        return np.median(a[sel]) if sel.any() else np.nan
    dss_m, dss_p, dss_c = m["exp_dss"], p["exp_dss"], c["exp_dss"]
    cov_m = float(np.nanmean(m["exp_cov"][ps])); cov_p = float(np.nanmean(p["exp_cov"][ps]))
    print("\n=== expansion magnitude (proper score + calibration; |z| is diagnostic only) ===")
    print(f"  coverage@90%: model={cov_m:.3f}  static={cov_p:.3f}   (want ~0.90; <0.90 = over-confident)")
    print(f"  median DSS  : model={_med(dss_m,ps):+.3f}  static={_med(dss_p,ps):+.3f}  pooled={_med(dss_c,ps):+.3f}  (lower=better)")
    print(f"  DSS skill (null - model, >0 = model adds value): "
          f"vs_static={_med(dss_p,ps)-_med(dss_m,ps):+.3f}  vs_pooled={_med(dss_c,ps)-_med(dss_m,ps):+.3f}"
          f"  (n={int((ps&np.isfinite(dss_m)).sum())})")

    # ===== ACCEPTANCE TEST (E): is the within-tissue magnitude depth-CALIBRATED? =====
    # Standardized NB residual z = (obs - pred)/sqrt(pred + (Σ_k mean_k^2)/phi), over ALL profiled forward
    # transitions (extinct included). NO survival conditioning -- conditioning on survival/recapture
    # is itself depth-correlated selection (deep samples surface low-share clones whose survivors are
    # growers), which makes survivor-conditioned point metrics (|log-ratio|) slope with depth even
    # when the model is fine. mean z ~ 0 and flat vs source depth => model depth-calibrated within
    # tissue and the readout fix is sufficient; a real slope => depth bias => exposure-model refit.
    # CAVEAT (why this is necessary but not strongly sufficient): in-sample the slope is partly
    # enforced by the fit -- mean z~0 is near-mechanical, and the depth tilt is absorbed wherever
    # depth is collinear with the design -- and at this n even a small slope is many SE from 0, so
    # "clean" is RELATIVE to the survivor-conditioned version. The decisive test is OUT-OF-SAMPLE
    # (scripts/score_holdouts.py: depth gate on held-out predictions), where the fit can't have
    # absorbed the residual. SE is printed here so the in-sample slope's resolution is visible.
    Xr_raw = obs.X.reshape(J, S, K).sum(2); Xr_shr = Xt.reshape(J, S, K).sum(2)
    si = Xr_shr.argmax(1)                                   # source tissue used for the metric
    d_src = Xr_raw[np.arange(J), si] / np.maximum(Xr_shr[np.arange(J), si], 1e-12)   # depth d_s
    prof = (D.reshape(J, S, K).sum(2) > 0)[np.arange(J), si]   # source tissue profiled at dest
    mean_src = mean_model.reshape(J, S, K)[np.arange(J), si, :]   # [J,K] predicted means in source tissue
    pred_s = mean_src.sum(1)
    obs_s = Yt.reshape(J, S, K).sum(2)[np.arange(J), si]
    # within-tissue total variance = sum of per-state NB2 variances (see decompose_prediction)
    z = (obs_s - pred_s) / np.sqrt(np.maximum(pred_s + (mean_src ** 2).sum(1) / phi, 1e-9))
    print("\n=== ACCEPTANCE TEST: standardized NB residual vs source depth, within tissue ===")
    print("    (unconditional; aggregated to source samples [patient x tp x tissue], slope across "
          "samples weighted by clone count; want mean z~0, slope within ~2 SE)")
    for ti, tname in enumerate(ss.tissues):
        sel = prof & (si == ti) & (d_src > 0) & np.isfinite(z)
        n = int(sel.sum())
        if n < 20:
            print(f"  {tname:5s}: n={n} (too few)"); continue
        key = np.array([f"{p}|{t}" for p, t in zip(patient[sel], src_tp[sel])])
        uk, inv = np.unique(key, return_inverse=True)
        nS = np.bincount(inv).astype(float)
        mz_s = np.bincount(inv, weights=z[sel]) / nS
        ld_s = np.bincount(inv, weights=np.log(d_src[sel])) / nS
        mz = float((nS * mz_s).sum() / nS.sum())
        slope, se = _wls_slope_se(ld_s, mz_s, nS)
        resolved = np.isfinite(se) and abs(slope) > 2 * se
        flag = "DEPTH-BIASED -> refit" if (abs(mz) >= 0.15 or (resolved and abs(slope) > 0.05)) else "calibrated (clean)"
        print(f"  {tname:5s}: nS={len(uk):3d} (n={n:6d})  mean z={mz:+.3f}  "
              f"slope={slope:+.3f} ± {se:.3f}  -> {flag}")


if __name__ == "__main__":
    main()
