"""Fit the factored model  M = diag(g).(pi (x) Phi)  by NUTS and save it for offline read-outs.

Intended for a GPU node: point TRAFFIC_H5 at the dataset, run, then copy the saved .npz back and
reload locally with ``traffic.io.load_fit`` (numpy only -- no JAX/GPU needed to read it). On a CUDA
node install a GPU jaxlib (``pip install -U "jax[cuda12]"``) and it runs on the GPU unchanged.

Env:
  TRAFFIC_H5            dataset path (required)
  TRAFFIC_OUT          output file (default: factored_fit.npz)
  TRAFFIC_WARMUP       NUTS warmup iters   (default 1000)
  TRAFFIC_SAMPLES      NUTS samples/chain  (default 1000)
  TRAFFIC_CHAINS       chains              (default 2)
  TRAFFIC_SUBSAMPLE    observations to use (default 0 = full data)
  TRAFFIC_ALPHA_STAY   persistence stay concentration: a float, or 3 comma-separated per-tissue
                       values PBMC,CSF,TP  (default 4)
  TRAFFIC_TARGET_ACCEPT NUTS target accept (default 0.9)
"""
import os

import jax
jax.config.update("jax_enable_x64", True)   # factored NB2 needs float64
import numpy as np

from traffic import FactoredPriorConfig, MCMCConfig, data, io, mcmc, statespace


def _alpha_stay():
    parts = [float(x) for x in os.environ.get("TRAFFIC_ALPHA_STAY", "4").split(",")]
    return parts[0] if len(parts) == 1 else tuple(parts)


def main():
    h5 = os.environ.get("TRAFFIC_H5")
    if not h5:
        raise SystemExit("set TRAFFIC_H5 to the dataset path")
    out = os.environ.get("TRAFFIC_OUT", "factored_fit.npz")

    ss = statespace.default()
    obs_df = data.load_obs_table(h5)
    obs = data.build(obs_df, ss)
    rho = data.phenotype_dist_by_tissue(obs_df, ss)

    X, Y, D = obs.Xtilde, obs.Y, obs.D
    sub = int(os.environ.get("TRAFFIC_SUBSAMPLE", "0"))
    if sub and sub < X.shape[0]:
        idx = np.sort(np.random.default_rng(0).choice(X.shape[0], sub, replace=False))
        X, Y, D = X[idx], Y[idx], D[idx]

    prior = FactoredPriorConfig(alpha_stay=_alpha_stay())
    cfg = MCMCConfig(
        num_warmup=int(os.environ.get("TRAFFIC_WARMUP", "1000")),
        num_samples=int(os.environ.get("TRAFFIC_SAMPLES", "1000")),
        num_chains=int(os.environ.get("TRAFFIC_CHAINS", "2")),
        target_accept=float(os.environ.get("TRAFFIC_TARGET_ACCEPT", "0.9")),
    )
    print(f"backend={jax.default_backend()}  J={X.shape[0]}  factored NB2  "
          f"alpha_stay={prior.alpha_stay}  warmup={cfg.num_warmup} samples={cfg.num_samples} "
          f"chains={cfg.num_chains}")

    res = mcmc.fit_nuts(X, Y, D, prior, cfg)
    print(f"divergences={res.num_divergences}  r_hat_max={res.r_hat_max:.3f}  "
          f"ess_min(identified)={res.ess_min:.0f}")

    path = io.save_fit(out, res, ss=ss, rho=rho, meta={
        "J": int(X.shape[0]), "backend": jax.default_backend(),
        "warmup": cfg.num_warmup, "samples": cfg.num_samples, "chains": cfg.num_chains,
        "alpha_stay": str(prior.alpha_stay),
    })
    print(f"saved {path}  ({res.n_draws} draws of M[{ss.L},{ss.L}] + factors g,pi,Phi)")


if __name__ == "__main__":
    main()
