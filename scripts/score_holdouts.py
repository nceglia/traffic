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


if __name__ == "__main__":
    main()
