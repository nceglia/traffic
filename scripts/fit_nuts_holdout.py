"""Leave-one-timepoint-out NUTS fit for predictive validation.

Holds out a destination timepoint t (a "clean" holdout): every forward
transition that *touches* t is removed from training -- both the into-t step
(t-1 -> t, the one we will later test) and the out-of-t step (t -> t+1) -- so
timepoint-t cell counts never appear in training, even as a source feature.
The model is then fit on the remaining transitions and saved; the held-out
into-t transitions (src_tp == t-1) form the test set for offline read-out.

Env:
  TRAFFIC_H5         dataset path (required)
  TRAFFIC_HOLDOUT    destination timepoint to hold out (required; one of 2..6)
  TRAFFIC_OUT        output file (default: nuts_fit_holdout_t{t}.npz)
  TRAFFIC_X64        1=float64, 0=float32 (default 0)
  TRAFFIC_WARMUP     NUTS warmup iters   (default 1000)
  TRAFFIC_SAMPLES    NUTS samples/chain  (default 1000)
  TRAFFIC_CHAINS     chains              (default 2)
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
    t = os.environ.get("TRAFFIC_HOLDOUT")
    if not t:
        raise SystemExit("set TRAFFIC_HOLDOUT to the destination timepoint to hold out")
    t = int(t)
    out = os.environ.get("TRAFFIC_OUT", f"nuts_fit_holdout_t{t}.npz")

    ss = statespace.default()
    obs_df = data.load_obs_table(h5)
    obs = data.build(obs_df, ss)
    rho = data.phenotype_dist_by_tissue(obs_df, ss)

    # Clean leave-one-timepoint-out: drop every transition touching timepoint t.
    # into-t  has src_tp == t-1 ; out-of-t has src_tp == t.
    stp = obs.src_tp.astype(int)
    touches_t = np.isin(stp, [t - 1, t])
    train = ~touches_t
    test_into_t = stp == (t - 1)  # the held-out transitions we will predict

    X, Y, D = obs.Xtilde[train], obs.Y[train], obs.D[train]

    lik = LikelihoodConfig(family=os.environ.get("TRAFFIC_LIKELIHOOD", "poisson"),
                           dispersion=os.environ.get("TRAFFIC_DISPERSION", "none"))
    patient_idx = None
    if lik.dispersion == "patient":
        ptr = np.asarray(obs.patient)[train]
        pmap = {p: i for i, p in enumerate(sorted(set(ptr.tolist())))}
        patient_idx = np.array([pmap[p] for p in ptr])

    cfg = MCMCConfig(
        num_warmup=int(os.environ.get("TRAFFIC_WARMUP", "1000")),
        num_samples=int(os.environ.get("TRAFFIC_SAMPLES", "1000")),
        num_chains=int(os.environ.get("TRAFFIC_CHAINS", "2")),
    )
    print(f"holdout t={t}  backend={jax.default_backend()}  likelihood={lik.family}/{lik.dispersion}  "
          f"J_total={obs.Xtilde.shape[0]}  J_train={X.shape[0]}  "
          f"J_test(into_t)={int(test_into_t.sum())}  excluded_src_tp={sorted(set(stp[touches_t].tolist()))}  "
          f"warmup={cfg.num_warmup} samples={cfg.num_samples} chains={cfg.num_chains}")

    res = mcmc.fit_nuts(X, Y, D, PriorConfig(), cfg, lik=lik, patient_idx=patient_idx)
    print(f"divergences={res.num_divergences}  r_hat_max={res.r_hat_max:.3f}  "
          f"ess_min={res.ess_min:.0f}")

    path = io.save_fit(out, res, ss=ss, rho=rho, meta={
        "likelihood": lik.family, "dispersion": lik.dispersion,
        "holdout_t": t,
        "excluded_src_tp": sorted(set(int(s) for s in stp[touches_t].tolist())),
        "test_src_tp": t - 1,
        "J_total": int(obs.Xtilde.shape[0]),
        "J_train": int(X.shape[0]),
        "J_test": int(test_into_t.sum()),
        "backend": jax.default_backend(),
        "warmup": cfg.num_warmup, "samples": cfg.num_samples, "chains": cfg.num_chains,
    })
    print(f"saved {path}  ({res.n_draws} draws of M[{ss.L},{ss.L}])")


if __name__ == "__main__":
    main()
