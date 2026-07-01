"""Validation figure: population vs individual recoverability across the three axes.

One frame, two questions:
  Q1 (population)  -- does the operator reproduce POOLED dynamics (aggregate over clones)?
  Q2 (individual)  -- can it predict a SINGLE clone's next step?
laid out as a 2x3 grid: rows = level (population / individual), cols = axis
(expansion / migration / switching), plus the recapture ceiling that gates the
individual row and a clone-size ribbon for the one stratification that matters.

Scoring (all cells 0 = pooled population-average null, higher = better):
  migration / switching : skill = 1 - JS_model / JS_pooled  (composition; JS in [0,1]).
  expansion             : DSS skill = (DSS_pooled - DSS_model) / (DSS_pooled - DSS_oracle),
                          the PROPER magnitude score (obs_s = within-source-tissue total,
                          Var_s = sum of per-state NB2 variances), normalized so 1 = the NB
                          oracle (best achievable given count noise). Coverage of the 90%
                          predictive interval is reported alongside as the calibration read.

The absolute clonal-expansion panel (fold-change in real cell numbers) is RESERVED --
it needs a per-sample cellularity anchor we do not yet have; left as a labeled blank.

Env: TRAFFIC_H5 (required), FIT (default nb_global_putative.npz), OUTDIR.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predictability_breakdown import decompose_prediction, abundance_bin, ABUND_LABELS  # noqa: E402

from traffic import data, io, statespace  # noqa: E402
from traffic.readouts import _js  # noqa: E402


def _wt_totals(mean, Y, D, src_tis, ss, phi):
    """within-source-tissue total: obs_s, pred_s, Var_s (sum of per-state NB2 vars)."""
    J = Y.shape[0]; S, K = ss.S, ss.K
    obsv = D > 0
    Yt = np.where(obsv, Y, 0.0).reshape(J, S, K)
    Mr = np.where(obsv, mean, 0.0).reshape(J, S, K)
    ar = np.arange(J)
    obs_s = Yt.sum(2)[ar, src_tis]
    pred_s = Mr.sum(2)[ar, src_tis]
    m2 = (Mr[ar, src_tis, :] ** 2).sum(1)
    var_s = np.maximum(pred_s + m2 / phi, 1e-9)
    return obs_s, pred_s, var_s


def _dss(obs, pred, var):
    return (obs - pred) ** 2 / var + np.log(var)


def _js_skill(model_err, pooled_err, base):
    """1 - median(model)/median(pooled) over `base` (composition axes)."""
    mm = model_err[base & np.isfinite(model_err)]
    pp = pooled_err[base & np.isfinite(pooled_err)]
    if not mm.size or not pp.size:
        return np.nan, 0
    med_p = np.median(pp)
    return (1 - np.median(mm) / med_p if med_p > 1e-6 else np.nan), int(mm.size)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    h5 = os.environ.get("TRAFFIC_H5")
    if not h5:
        raise SystemExit("set TRAFFIC_H5")
    fit_path = os.environ.get("FIT", "nb_global_putative.npz")
    outdir = os.environ.get("OUTDIR", "holdout_fits_putative/global/scoring")
    os.makedirs(outdir, exist_ok=True)

    ss = statespace.default()
    obs = data.build(data.load_obs_table(h5), ss)
    fit = io.load_fit(fit_path)
    S, K, L = ss.S, ss.K, ss.L
    Xt, Y, D = obs.Xtilde, obs.Y, obs.D
    J = Xt.shape[0]
    phi = float(np.exp(np.asarray(fit.dispersion["params"]["log_r"]).mean())) if fit.dispersion else np.inf
    obsv = D > 0
    Yt = np.where(obsv, Y, 0.0)
    pi = Yt.sum(0); pi = pi / max(pi.sum(), 1e-12)

    mean_model = np.where(obsv, D * (Xt @ fit.M_hat), 0.0)
    mean_pooled = np.where(obsv, D * (Xt.sum(1, keepdims=True) * pi[None, :]), 0.0)

    m = decompose_prediction(Xt, Y, D, mean_model, ss, phi)
    c = decompose_prediction(Xt, Y, D, mean_pooled, ss, phi)   # pooled population-average null

    surv = m["surv"]; prof = m["prof_src"]
    patient = np.asarray(obs.patient); pat_order = sorted(set(patient.tolist()))
    n_src = obs.n_src.astype(int); abin = np.array([abundance_bin(n) for n in n_src])
    src_tis = Xt.reshape(J, S, K).sum(2).argmax(1)

    # movers (composition axes score only where the behaviour occurs)
    src_t = obs.X.reshape(J, S, K).sum(2); src_t = src_t / np.maximum(src_t.sum(1, keepdims=True), 1e-12)
    obs_t = Yt.reshape(J, S, K).sum(2); obs_t = obs_t / np.maximum(obs_t.sum(1, keepdims=True), 1e-12)
    src_p = obs.X.reshape(J, S, K).sum(1); src_p = src_p / np.maximum(src_p.sum(1, keepdims=True), 1e-12)
    obs_p = Yt.reshape(J, S, K).sum(1); obs_p = obs_p / np.maximum(obs_p.sum(1, keepdims=True), 1e-12)
    from predictability_breakdown import _js_rows
    mig_ch = np.full(J, np.nan); swi_ch = np.full(J, np.nan)
    mig_ch[surv] = _js_rows(src_t[surv], obs_t[surv]); swi_ch[surv] = _js_rows(src_p[surv], obs_p[surv])
    mov_mig = surv & (mig_ch > 0.05); mov_swi = surv & (swi_ch > 0.05)

    # ---- expansion (magnitude) proper score: DSS skill toward the NB oracle ----
    obs_s, pred_s, var_s = _wt_totals(mean_model, Y, D, src_tis, ss, phi)
    _, pred_sp, var_sp = _wt_totals(mean_pooled, Y, D, src_tis, ss, phi)
    var_o = np.maximum(obs_s + obs_s ** 2 / phi, 1e-9)                 # var at perfect aim
    dss_m = _dss(obs_s, pred_s, var_s); dss_p = _dss(obs_s, pred_sp, var_sp); dss_o = np.log(var_o)

    def exp_skill(base):
        b = base & np.isfinite(dss_m) & np.isfinite(dss_p)
        if b.sum() < 5:
            return np.nan, int(b.sum())
        num = dss_p[b].mean() - dss_m[b].mean(); den = dss_p[b].mean() - dss_o[b].mean()
        return (num / den if abs(den) > 1e-9 else np.nan), int(b.sum())

    # sample grouping for the POPULATION expansion cell (aggregate within-tissue totals per sample)
    src_tp = obs.src_tp.astype(int)
    key = np.array([f"{p}|{t}|{s}" for p, t, s in zip(patient, src_tp, src_tis)])
    uk, inv = np.unique(key[prof], return_inverse=True)
    def agg(x): return np.bincount(inv, weights=x[prof])
    Os, Ps, Vs = agg(obs_s), agg(pred_s), agg(var_s)          # aggregate sums per sample
    _, Pp, Vp = agg(np.zeros(J)), agg(pred_sp), agg(var_sp)
    Vo = np.maximum(Os + Os ** 2 / phi, 1e-9)
    dss_M, dss_P, dss_O = _dss(Os, Ps, Vs), _dss(Os, Pp, Vp), np.log(Vo)
    pop_exp = ((dss_P.mean() - dss_M.mean()) / (dss_P.mean() - dss_O.mean())
               if abs(dss_P.mean() - dss_O.mean()) > 1e-9 else np.nan)

    # ---- assemble the 2x3 skill grid ----
    axes_lbl = ["expansion", "migration", "switching"]
    rows_lbl = ["population\n(aggregate)", "individual\n(per clone, movers)"]
    grid = np.full((2, 3), np.nan); ns = np.zeros((2, 3), int)

    # population row
    grid[0, 0] = pop_exp; ns[0, 0] = len(uk)
    # aggregate composition (pool all destination mass) for migration & switching
    def agg_comp_skill(kind):
        A_obs = Yt.sum(0); A_m = mean_model.sum(0); A_p = mean_pooled.sum(0)
        Ao = A_obs.reshape(S, K); Am = A_m.reshape(S, K); Ap = A_p.reshape(S, K)
        if kind == "migration":                                # tissue marginal
            o, mm, pp = Ao.sum(1), Am.sum(1), Ap.sum(1)
            jm, jp = _js(o, mm), _js(o, pp)
        else:                                                  # within-tissue phenotype, mass-weighted
            w = Ao.sum(1); w = w / w.sum()
            jm = sum(w[s] * _js(Ao[s], Am[s]) for s in range(S))
            jp = sum(w[s] * _js(Ao[s], Ap[s]) for s in range(S))
        return (1 - jm / jp) if jp > 1e-6 else np.nan
    grid[0, 1] = agg_comp_skill("migration"); ns[0, 1] = int(surv.sum())
    grid[0, 2] = agg_comp_skill("switching"); ns[0, 2] = int(surv.sum())

    # individual row
    grid[1, 0], ns[1, 0] = exp_skill(prof)
    grid[1, 1], ns[1, 1] = _js_skill(m["migration"], c["migration"], mov_mig)
    grid[1, 2], ns[1, 2] = _js_skill(m["switching"], c["switching"], mov_swi)

    # ---- recapture per patient (the ceiling; shown as-is, no flags) ----
    recap = {p: float((surv & (patient == p)).sum()) / max((patient == p).sum(), 1) for p in pat_order}

    # ---- clone-size ribbon: individual skill by abundance x axis ----
    ribbon = np.full((len(ABUND_LABELS), 3), np.nan)
    for i, lab in enumerate(ABUND_LABELS):
        sel = abin == lab
        ribbon[i, 0] = exp_skill(prof & sel)[0]
        ribbon[i, 1] = _js_skill(m["migration"], c["migration"], mov_mig & sel)[0]
        ribbon[i, 2] = _js_skill(m["switching"], c["switching"], mov_swi & sel)[0]

    # expansion calibration (coverage of the 90% interval) -- reported under the grid
    cov_pop = float(np.nanmean(m["exp_cov"][prof]))

    # ================================ FIGURE ================================ #
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.5, 1.0], width_ratios=[2.2, 1.1, 1.1],
                  hspace=0.32, wspace=0.32)

    # (A) the 2x3 core skill grid
    axg = fig.add_subplot(gs[0, 0])
    im = axg.imshow(grid, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    axg.set_xticks(range(3)); axg.set_xticklabels([a.capitalize() for a in axes_lbl], fontsize=11)
    axg.set_yticks(range(2)); axg.set_yticklabels(rows_lbl, fontsize=10)
    for r in range(2):
        for cc in range(3):
            v = grid[r, cc]
            axg.text(cc, r, ("n/a" if not np.isfinite(v) else f"{v:+.2f}\nn={ns[r, cc]}"),
                     ha="center", va="center", fontsize=10,
                     color="k" if (not np.isfinite(v) or abs(v) < 0.6) else "w")
    axg.set_title("A. Recoverability skill  (0 = population-average null, 1 = best achievable)\n"
                  f"expansion = magnitude calibration (90% coverage = {cov_pop:.2f})",
                  fontsize=11, loc="left")
    fig.colorbar(im, ax=axg, fraction=0.046, pad=0.03, label="skill")

    # (B) recapture per patient
    axr = fig.add_subplot(gs[0, 1])
    ys = np.arange(len(pat_order))
    axr.barh(ys, [recap[p] for p in pat_order], color="#8172b3", alpha=0.8)
    axr.set_yticks(ys); axr.set_yticklabels(pat_order, fontsize=9); axr.invert_yaxis()
    axr.set_xlim(0, 1); axr.axvline(surv.mean(), color="k", ls=":", lw=1)
    axr.set_xlabel("recapture rate"); axr.set_title(f"B. Recapture ceiling\n(overall {surv.mean():.2f})",
                                                    fontsize=11, loc="left")

    # (C) RESERVED -- absolute expansion, awaiting per-sample cellularity
    axp = fig.add_subplot(gs[0, 2])
    axp.axis("off")
    axp.add_patch(plt.Rectangle((0.03, 0.03), 0.94, 0.94, fill=False, ls="--", lw=1.5,
                                ec="#888", transform=axp.transAxes))
    axp.text(0.5, 0.5, "Absolute clonal\nexpansion\n\n(reserved:\nper-sample\ncellularity)",
             ha="center", va="center", fontsize=11, color="#666", transform=axp.transAxes)
    axp.set_title("C.", fontsize=11, loc="left")

    # (D) clone-size ribbon
    axb = fig.add_subplot(gs[1, 0])
    imb = axb.imshow(ribbon.T, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    axb.set_yticks(range(3)); axb.set_yticklabels([a.capitalize() for a in axes_lbl], fontsize=10)
    axb.set_xticks(range(len(ABUND_LABELS))); axb.set_xticklabels(ABUND_LABELS, fontsize=9)
    axb.set_xlabel("source clone size")
    for i in range(len(ABUND_LABELS)):
        for j in range(3):
            v = ribbon[i, j]
            if np.isfinite(v):
                axb.text(i, j, f"{v:+.2f}", ha="center", va="center", fontsize=8,
                         color="k" if abs(v) < 0.6 else "w")
    axb.set_title("D. Individual skill by clone size  (magnitude is legible only for large clones)",
                  fontsize=11, loc="left")
    fig.colorbar(imb, ax=axb, fraction=0.046, pad=0.03, label="skill")

    # (E) text takeaway
    axt = fig.add_subplot(gs[1, 1:]); axt.axis("off")
    axt.text(0.0, 0.95,
             "Reading:\n"
             "- Population row: pooled dynamics are recovered on all three axes.\n"
             "- Individual row: fate (migration, switching) is recoverable;\n"
             "  magnitude only in aggregate / for large clones.\n"
             "- All individual prediction is capped by recapture (panel B).\n"
             "- Expansion here is share-space magnitude; ABSOLUTE expansion\n"
             "  is panel C, reserved for the cellularity anchor.",
             ha="left", va="top", fontsize=9.5, transform=axt.transAxes)

    fig.suptitle("Model validation: population faithful, individual fate-but-not-size, recapture-limited",
                 fontsize=13, y=0.99)
    out = os.path.join(outdir, "validation_summary.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")
    print(f"grid (pop/ind x exp/mig/swi):\n{np.round(grid, 3)}")
    print(f"expansion 90% coverage = {cov_pop:.3f}  | overall recapture = {surv.mean():.3f}")


if __name__ == "__main__":
    main()
