"""Score a family of held-out fits and emit figures + a results table.

Loads the dataset once, scores every predict-mode holdout in a directory, runs
the influence comparison for the boundary (noT1) holdout against the full fit,
and writes:
    <outdir>/scores.csv               per-holdout summary table
    <outdir>/predictive.png           4-panel predictive read-out
    <outdir>/influence_<name>.png      influence panel(s)

Env:
  TRAFFIC_H5    dataset path (required)
  SCORE_DIR     directory of holdout npz (default holdout_fits/global)
  FULL_FIT      full-data fit for the influence baseline (default nb_global_logr_x64.npz)
  SCORE_OUT     output directory (default <SCORE_DIR>/scoring)
  SCORE_DRAWS   posterior draws to use (default 400)
"""
import csv
import glob
import os

import numpy as np

from traffic import data, scoring, statespace


def main():
    h5 = os.environ.get("TRAFFIC_H5")
    if not h5:
        raise SystemExit("set TRAFFIC_H5 to the dataset path")
    sdir = os.environ.get("SCORE_DIR", "holdout_fits/global")
    full = os.environ.get("FULL_FIT", "nb_global_logr_x64.npz")
    out = os.environ.get("SCORE_OUT", os.path.join(sdir, "scoring"))
    n_draws = int(os.environ.get("SCORE_DRAWS", "400"))
    os.makedirs(out, exist_ok=True)

    ss = statespace.default()
    obs = data.build(data.load_obs_table(h5), ss)
    print(f"dataset J={obs.Xtilde.shape[0]}  scoring dir={sdir}  draws={n_draws}")

    npzs = sorted(glob.glob(os.path.join(sdir, "*.npz")))
    scored, results, influence_targets = [], [], []
    for p in npzs:
        res = scoring.evaluate_holdout(p, obs, n_draws=n_draws)
        if res is None:
            influence_targets.append(p)
            print(f"  [influence] {scoring._short(p):16} (no test set -> compare to full)")
            continue
        sc, sk, agg = res["model"], res["skill"], res["aggregate"]
        scored.append(sc); results.append(res)
        s = sc.summary()
        print(f"  [predict ] {s['name']:16} n={s['n']:6d}  elpd={s['elpd_mean']:7.3f}  "
              f"JS(w)={s['jsd_wmean']:.3f}  | skill_shape(vs static/pooled)="
              f"{sk['skill_shape_vs_static']:+.2f}/{sk['skill_shape_vs_pooled']:+.2f}  "
              f"agg.JS model/static/floor={agg['js_model']:.3f}/{agg['js_static']:.3f}/"
              f"{agg['js_noise_floor']:.3f}")

    # results tables: per-clone summary + feasibility/skill
    csv_path = os.path.join(out, "scores.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(scored[0].summary().keys()))
        w.writeheader()
        for sc in scored:
            w.writerow(sc.summary())
    feas_path = os.path.join(out, "feasibility.csv")
    with open(feas_path, "w", newline="") as fh:
        rows = [{**r["skill"], **{f"agg_{k}": v for k, v in r["aggregate"].items() if k != "name"}}
                for r in results]
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"wrote {csv_path}\nwrote {feas_path}")

    # figures
    fig = scoring.figure_predictive(scored, os.path.join(out, "predictive.png"),
                                    title=f"Held-out predictive accuracy -- {os.path.basename(sdir)}",
                                    skills={r["skill"]["name"]: r["skill"] for r in results})
    print(f"wrote {fig}")
    fig = scoring.figure_feasibility(results, os.path.join(out, "feasibility.png"),
                                     title=f"Feasibility: model skill vs nulls -- {os.path.basename(sdir)}")
    print(f"wrote {fig}")

    # influence figures for boundary holdouts
    if os.path.exists(full):
        for p in influence_targets:
            inf = scoring.compare_fits(p, full, obs, n_draws=n_draws)
            name = scoring._short(p)
            fig = scoring.figure_influence(inf, ss, os.path.join(out, f"influence_{name}.png"))
            d = inf.delta
            print(f"  influence {name}: rel||dM||={inf.rel_frobenius:.3f}  "
                  f"d.elpd={d['elpd_mean']:+.3f}  d.JS(w)={d['jsd_wmean']:+.3f}  -> {fig}")
    else:
        print(f"full fit {full} not found -- skipping influence panels")

    # ---- out-of-sample depth-calibration gate (acceptance test E on held-out predictions) ----
    # Standardized within-source-tissue NB residual, aggregated to SOURCE SAMPLES (depth is a
    # per-sample quantity, so a clone-level slope is clustered/ill-conditioned). Per holdout we
    # report the sample-weighted mean residual; the depth SLOPE is fit only on the POOLED
    # leave-one-patient-out set, whose test sets are disjoint (each patient predicted by a fit
    # that never saw it) so pooling neither double-counts nor needs the thin per-holdout fits.
    # in-sample-flat + holdout-flat closes the depth branch; a resolved pooled slope (|slope| >
    # ~2 SE) with mean z off 0 reopens the exposure-model refit.
    print("\n=== out-of-sample depth gate: within-tissue NB residual vs source depth (sample-level) ===")
    pool = {t: {"logd": [], "z": [], "w": []} for t in ss.tissues}
    for r in results:
        g = r.get("depth_gate")
        if not g:
            continue
        is_pat = r["model"].meta.get("split") == "patient"      # leave-one-patient-out -> disjoint
        bt = g["by_tissue"]
        cells = "  ".join(
            (f"{t}:z={bt[t]['mean_z']:+.2f}(nS={bt[t]['n_samp']},nc={bt[t]['n_clones']})"
             if np.isfinite(bt[t]["mean_z"]) else f"{t}:nS={bt[t]['n_samp']}(thin)")
            for t in ss.tissues)
        print(f"  [{'PT' if is_pat else '  '}] {r['skill']['name']:16} {cells}")
        if is_pat:
            for c, t in enumerate(ss.tissues):
                sel = g["samp_tissue"] == c
                if sel.any():
                    pool[t]["logd"].append(g["samp_logd"][sel])
                    pool[t]["z"].append(g["samp_z"][sel])
                    pool[t]["w"].append(g["samp_w"][sel])
    print("  --- POOLED slope over leave-one-patient-out holdouts (disjoint; N = #samples) ---")
    for t in ss.tissues:
        if not pool[t]["logd"]:
            print(f"  {t:5s}: no out-of-sample samples"); continue
        x = np.concatenate(pool[t]["logd"]); y = np.concatenate(pool[t]["z"]); w = np.concatenate(pool[t]["w"])
        if x.size < 3:
            print(f"  {t:5s}: nS={x.size} (too few samples to fit)"); continue
        slope, se = scoring._wls_slope_se(x, y, w)
        mz = float((w * y).sum() / w.sum())
        resolved = np.isfinite(se) and abs(slope) > 2 * se
        flag = "depth-tilt -> inspect refit" if (abs(mz) >= 0.15 or (resolved and abs(slope) > 0.05)) else "calibrated"
        print(f"  {t:5s}: nS={x.size:3d}  nc={int(w.sum()):6d}  mean z={mz:+.3f}  "
              f"slope={slope:+.3f} ± {se:.3f}  -> {flag}")


if __name__ == "__main__":
    main()
