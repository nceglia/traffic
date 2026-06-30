"""Where can we predict next-timepoint clonal behavior, and where can't we?

Scores the global fit's one-step-ahead prediction for every forward transition
and decomposes the destination prediction into three biologically distinct axes:

  expansion   the total destination mass  N_j = sum_z mu_j(z)  -- did the clone
              grow/shrink by the predicted amount (magnitude / net growth).
  migration   the destination *tissue marginal*  c^tis_j(s) = sum_k c_j(s,k)
              -- did cells redistribute across PBMC/CSF/TP as predicted.
  switching   the *within-tissue phenotype* composition  c_j(k|s), averaged over
              tissues weighted by observed tissue mass -- did phenotypes convert
              as predicted, holding tissue fixed.

Migration + switching are the exact KL chain-rule split of the destination
composition divergence  D(c_obs || c_pred) = D(tissue marginal) + E[D(pheno|tissue)]
(here computed with the bounded Jensen-Shannon analogue per piece); expansion is
the orthogonal magnitude axis. Each axis is scored against two nulls --
persistence (M = identity: "cells stay put") and climatology (population mean
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


def decompose_prediction(Xt, Y, D, mean, ss):
    """Per-clone expansion / migration / switching error for a predicted mean.

    Returns dict of length-J arrays (NaN where undefined: shape axes need a
    surviving clone; switching needs a tissue with both observed and predicted
    mass).  expansion = |log predicted/observed total| (survivors).
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

    # expansion: |log-ratio| of destination totals (survivors)
    expansion = np.full(J, np.nan)
    expansion[surv] = np.abs(np.log((N_pred[surv] + 0.5) / (N_obs[surv] + 0.5)))
    # signed log-growth (relative to source size) for direction accuracy
    obs_grow = np.full(J, np.nan); pred_grow = np.full(J, np.nan)
    nsrc = np.maximum(Xt.sum(1) * 0 + Yt.sum(1) * 0, 0)  # placeholder, set by caller

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

    return {"N_obs": N_obs, "N_pred": N_pred, "surv": surv,
            "expansion": expansion, "migration": migration, "switching": switching}


def stratum_summary(key, order, model_err, null_err, base_mask):
    """Per-stratum median model & null error and skill = 1 - med_model/med_null."""
    rows = []
    for lab in order:
        sel = base_mask & (key == lab) & np.isfinite(model_err)
        n = int(sel.sum())
        mm = float(np.median(model_err[sel])) if n else np.nan
        seln = base_mask & (key == lab) & np.isfinite(null_err)
        mn = float(np.median(null_err[seln])) if seln.sum() else np.nan
        # skill undefined when the null is ~perfect (mn≈0, e.g. persistence on the
        # static no-change majority) -- guard so it doesn't blow up.
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
    print(f"fit={fit_path}  J={J}  L={L}  tissues={ss.tissues}")

    # predicted destination means: model (M_hat), persistence (M=I), climatology (pi)
    obsv = D > 0
    Yt = np.where(obsv, Y, 0.0)
    pi = Yt.sum(0); pi = pi / max(pi.sum(), 1e-12)        # population destination comp [L]
    mean_model = np.where(obsv, D * (Xt @ fit.M_hat), 0.0)
    mean_persist = np.where(obsv, D * Xt, 0.0)
    mean_clim = np.where(obsv, D * (Xt.sum(1, keepdims=True) * pi[None, :]), 0.0)

    m = decompose_prediction(Xt, Y, D, mean_model, ss)
    p = decompose_prediction(Xt, Y, D, mean_persist, ss)
    c = decompose_prediction(Xt, Y, D, mean_clim, ss)

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
    # interest, not the static no-change majority (which persistence predicts for
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
    mover = {
        "expansion": surv & (np.abs(obs_lg) > np.log(2)),     # >2x net size change
        "migration": surv & (mig_change > 0.05),              # tissue distribution shifted
        "switching": surv & (swi_change > 0.05),              # phenotype distribution shifted
    }

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
                    "model_err", "persistence_err", "climatology_err",
                    "skill_vs_persistence_all", "skill_vs_climatology_all",
                    "skill_vs_persistence_movers", "skill_vs_climatology_movers"])
        for ax_name, (em, ep, ec) in axes.items():
            mv = mover[ax_name]
            for kind, (key, order) in strata.items():
                sp = stratum_summary(key, order, em, ep, surv)       # all survivors
                sc = stratum_summary(key, order, em, ec, surv)
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
        ax.plot(np.arange(len(order)), med_p, "rD", ms=6, label="persistence median")
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels([f"{o}\n(n={int((base&(key==o)).sum())})" for o in order],
                           fontsize=7)
        ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(fontsize=7)

    def by_group_bars(ax, key, order, em, ep, ec, base, ylabel, title):
        med = lambda arr, q: (np.nanmedian(arr[base & (key == q)])
                              if (base & (key == q)).any() else np.nan)
        xx = np.arange(len(order))
        ax.bar(xx - 0.25, [med(em, q) for q in order], 0.25, label="model", color="#4c72b0")
        ax.bar(xx, [med(ep, q) for q in order], 0.25, label="persistence", color="#c44e52", alpha=0.8)
        ax.bar(xx + 0.25, [med(ec, q) for q in order], 0.25, label="climatology", color="#dd8452", alpha=0.8)
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
                "|log(pred/obs total)|  (lower=better)",
                "B. Expansion error by clone abundance (expanders)", mover["expansion"])
    by_group_bars(ax[1, 0], patient, pat_order, m["expansion"], p["expansion"], c["expansion"],
                  mover["expansion"], "median |log-ratio|", "C. Expansion error by patient (expanders)")
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
    fig.suptitle("Expansion (net growth / magnitude) -- global fit one-step prediction", y=0.995)
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
                    "D. Switching skill vs persistence, SWITCHERS only\n(+ = model better, - = worse)",
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
                    "D. Migration skill vs persistence, MIGRATORS only\n(+ = model better, - = worse)",
                    mover["migration"])
    fig.colorbar(im, ax=ax[1, 1], fraction=0.046)
    fig.suptitle("Migration (cross-tissue redistribution) -- global fit one-step prediction", y=0.995)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "migration.png"), dpi=130); plt.close(fig)
    print(f"wrote {os.path.join(outdir, 'migration.png')}")

    # ===================== FIGURE 4: PREDICTABILITY SUMMARY =================== #
    # master heatmap: rows = strata (abundance bins + patients), cols = 3 axes,
    # color = skill vs persistence; a trailing column shows recapture rate.
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
    a.set_title("Predictability skill vs persistence, conditional on movers\n(+ = model adds value where the behavior occurs)")
    fig.colorbar(im, ax=a, fraction=0.046, label="skill = 1 - err_model/err_persist")
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
    print("\n=== axis medians: all survivors  vs  movers-only (the events of interest) ===")
    for ax_name, (em, ep, ec) in axes.items():
        mv = mover[ax_name]
        def med(arr, msk):
            s = msk & np.isfinite(arr)
            return np.median(arr[s]) if s.any() else np.nan
        sk_all = 1 - med(em, surv) / med(ep, surv) if med(ep, surv) > 1e-4 else np.nan
        sk_mv = 1 - med(em, mv) / med(ep, mv) if med(ep, mv) > 1e-4 else np.nan
        print(f"  {ax_name:10} ALL: model={med(em, surv):.3f} persist={med(ep, surv):.3f} "
              f"clim={med(ec, surv):.3f} skill_vs_persist={sk_all:+.2f} (n={int((surv&np.isfinite(em)).sum())})")
        print(f"  {'':10} MOVERS: model={med(em, mv):.3f} persist={med(ep, mv):.3f} "
              f"clim={med(ec, mv):.3f} skill_vs_persist={sk_mv:+.2f} (n_movers={int(mv.sum())})")
    print(f"  overall recapture rate = {surv.mean():.3f}")


if __name__ == "__main__":
    main()
