"""Per-transition operator fits M_r for the temporal-stabilization analysis.

Fits a SEPARATE factored NB2 operator for each consecutive step r->r+1 (source
timepoint r), on that transition's observations only. Comparing M_1->2, M_2->3, ...
answers whether the trafficking dynamics become time-homogeneous after the OVI
transient (T1 = pre-perturbation baseline; biopsy precedes injection in the same
surgery). This is the NUTS-path successor to the stale `scripts/stationarity.py`.

The fits are the expensive part, so each transition is saved incrementally to one npz as
it completes (a crash keeps finished transitions). This is a bespoke MULTI-transition
archive (not a single SavedFit): the per-transition factor draws (g/pi/Phi) and the global
log_r are stacked and `meta` is JSON, so a dedicated temporal reader can NB-score any
transition. Readouts/plots are a separate script.

Env:
  TRAFFIC_H5        dataset path (required)
  TRAFFIC_OUT       output npz (default temporal_Mr.npz)
  TRAFFIC_WARMUP    NUTS warmup   (default 500, exploratory)
  TRAFFIC_SAMPLES   NUTS samples  (default 500, exploratory)
  TRAFFIC_CHAINS    chains        (default 2)
  TRAFFIC_NSUB      posterior draws to keep per transition (default 200)
  TRAFFIC_TRANS     comma list of source timepoints to fit (default = all present)
"""
import json
import os

import jax
jax.config.update("jax_enable_x64", True)          # NB likelihood needs float64
import numpy as np

from traffic import FactoredPriorConfig, MCMCConfig, data, mcmc, statespace


def main():
    h5 = os.environ.get("TRAFFIC_H5")
    if not h5:
        raise SystemExit("set TRAFFIC_H5 to the dataset path")
    out = os.environ.get("TRAFFIC_OUT", "temporal_Mr.npz")
    warm = int(os.environ.get("TRAFFIC_WARMUP", "500"))
    samp = int(os.environ.get("TRAFFIC_SAMPLES", "500"))
    chains = int(os.environ.get("TRAFFIC_CHAINS", "2"))
    nsub = int(os.environ.get("TRAFFIC_NSUB", "200"))

    ss = statespace.default()
    obs_df = data.load_obs_table(h5)
    obs = data.build(obs_df, ss)
    rho = data.phenotype_dist_by_tissue(obs_df, ss)
    stp = obs.src_tp.astype(int)

    present = sorted(set(stp.tolist()))
    want = os.environ.get("TRAFFIC_TRANS")
    transitions = [int(x) for x in want.split(",")] if want else present
    transitions = [r for r in transitions if r in present]

    prior = FactoredPriorConfig()
    cfg = MCMCConfig(num_warmup=warm, num_samples=samp, num_chains=chains)
    print(f"backend={jax.default_backend()}  L={ss.L}  transitions={transitions}  "
          f"NUTS {warm}+{samp} x{chains}")

    M_hat, samples, r_hat, divs, ess, n_obs, src_tp = [], [], [], [], [], [], []
    g_s, pi_s, Phi_s, logr_s = [], [], [], []          # per-transition factor + dispersion draws
    rng = np.random.default_rng(0)
    for r in transitions:
        m = stp == r
        n = int(m.sum())
        res = mcmc.fit_nuts(obs.Xtilde[m], obs.Y[m], obs.D[m], prior, cfg)
        idx = np.sort(rng.choice(res.n_draws, min(nsub, res.n_draws), replace=False))
        M_hat.append(res.M_hat); samples.append(res.samples[idx])
        r_hat.append(res.r_hat_max); divs.append(res.num_divergences)
        ess.append(res.ess_min); n_obs.append(n); src_tp.append(r)
        g_s.append(res.factors["g"][idx]); pi_s.append(res.factors["pi"][idx])
        Phi_s.append(res.factors["Phi"][idx]); logr_s.append(res.dispersion["params"]["log_r"][idx])
        print(f"  M_{r}->{r+1}: n={n:6d}  rhat={res.r_hat_max:.3f}  div={res.num_divergences}  "
              f"ess={res.ess_min:.0f}")
        # incremental save after each transition -- keep the factor draws + global log_r so any
        # transition stays NB-scorable, and JSON `meta` so the archive is machine-readable.
        np.savez(out,
                 M_hat=np.array(M_hat), samples=np.array(samples),
                 factor_g=np.array(g_s), factor_pi=np.array(pi_s), factor_Phi=np.array(Phi_s),
                 log_r=np.array(logr_s),
                 r_hat=np.array(r_hat), divergences=np.array(divs), ess_min=np.array(ess),
                 n_obs=np.array(n_obs), src_tp=np.array(src_tp),
                 tissues=np.array(ss.tissues), phenotypes=np.array(ss.phenotypes), rho=rho,
                 meta=json.dumps({"warmup": warm, "samples": samp, "chains": chains,
                                  "parameterization": "factored", "dispersion_mode": "global"}))
        print(f"    saved {out}  ({len(M_hat)}/{len(transitions)} transitions)")


if __name__ == "__main__":
    main()
