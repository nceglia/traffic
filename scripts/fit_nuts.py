"""Fit the full Gamma-Poisson model by NUTS and save it for offline read-outs.

Intended for a GPU node: point TRAFFIC_H5 at the dataset, run, then copy the
saved .npz back and reload locally with ``traffic.io.load_fit`` (numpy only --
no JAX/GPU needed to read it).

The code is device-agnostic; on a CUDA node install a GPU jaxlib
(``pip install -U "jax[cuda12]"``) and it runs on the GPU with no changes.

Env:
  TRAFFIC_H5         dataset path (required)
  TRAFFIC_OUT        output file (default: nuts_fit.npz)
  TRAFFIC_X64        1=float64, 0=float32 (default 0; f32 is much faster on GPU)
  TRAFFIC_WARMUP     NUTS warmup iters   (default 1000)
  TRAFFIC_SAMPLES    NUTS samples/chain  (default 1000)
  TRAFFIC_CHAINS     chains              (default 2)
  TRAFFIC_SUBSAMPLE  observations to use (default 0 = full data)
  TRAFFIC_LIKELIHOOD poisson | nb                        (default poisson)
  TRAFFIC_DISPERSION none | global | tissue | patient    (NB only; default none)
"""
import os

import jax

jax.config.update("jax_enable_x64", os.environ.get("TRAFFIC_X64", "0") != "0")
import numpy as np

from traffic import LikelihoodConfig, MCMCConfig, PriorConfig, data, io, mcmc, statespace


def main():
    h5 = os.environ.get("TRAFFIC_H5")
    if not h5:
        raise SystemExit("set TRAFFIC_H5 to the dataset path")
    out = os.environ.get("TRAFFIC_OUT", "nuts_fit.npz")

    ss = statespace.default()
    obs_df = data.load_obs_table(h5)
    obs = data.build(obs_df, ss)
    rho = data.phenotype_dist_by_tissue(obs_df, ss)

    X, Y, D = obs.Xtilde, obs.Y, obs.D
    patient = np.asarray(obs.patient)
    sub = int(os.environ.get("TRAFFIC_SUBSAMPLE", "0"))
    if sub and sub < X.shape[0]:
        idx = np.sort(np.random.default_rng(0).choice(X.shape[0], sub, replace=False))
        X, Y, D, patient = X[idx], Y[idx], D[idx], patient[idx]

    lik = LikelihoodConfig(family=os.environ.get("TRAFFIC_LIKELIHOOD", "poisson"),
                           dispersion=os.environ.get("TRAFFIC_DISPERSION", "none"))
    patient_idx = None
    if lik.dispersion == "patient":
        pmap = {p: i for i, p in enumerate(sorted(set(patient.tolist())))}
        patient_idx = np.array([pmap[p] for p in patient])

    cfg = MCMCConfig(
        num_warmup=int(os.environ.get("TRAFFIC_WARMUP", "1000")),
        num_samples=int(os.environ.get("TRAFFIC_SAMPLES", "1000")),
        num_chains=int(os.environ.get("TRAFFIC_CHAINS", "2")),
    )
    print(f"backend={jax.default_backend()}  J={X.shape[0]}  "
          f"likelihood={lik.family}/{lik.dispersion}  "
          f"warmup={cfg.num_warmup} samples={cfg.num_samples} chains={cfg.num_chains}")

    res = mcmc.fit_nuts(X, Y, D, PriorConfig(), cfg, lik=lik, patient_idx=patient_idx)
    print(f"divergences={res.num_divergences}  r_hat_max={res.r_hat_max:.3f}  "
          f"ess_min={res.ess_min:.0f}")

    path = io.save_fit(out, res, ss=ss, rho=rho, meta={
        "J": int(X.shape[0]), "backend": jax.default_backend(),
        "warmup": cfg.num_warmup, "samples": cfg.num_samples, "chains": cfg.num_chains,
    })
    print(f"saved {path}  ({res.n_draws} draws of M[{ss.L},{ss.L}])")


if __name__ == "__main__":
    main()
