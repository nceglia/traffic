"""Non-temporal train/test splits for predictive validation of M.

Unlike fit_nuts_holdout.py (which splits on timepoint), this fits M on a subset
of observations defined by patient, source-clone abundance, or a clone hash, so
the held-out test set is reconstructable downstream from the same rule.

Splits (TRAFFIC_SPLIT):
  patient     leave-one-patient-out. train = patient != TRAFFIC_PATIENT,
              test = patient == TRAFFIC_PATIENT. No timepoint exclusion.
  abundance   train on abundant source clones (n_src >= 5); test set downstream
              is the rare ones (n_src <= 2). (n_src in {3,4} is an intentional gap.)
  clone8020   split whole clones (trb) 80/20 by a stable hash; train = bucket<80,
              test = bucket>=80. A clone never spans both sides.

Env:
  TRAFFIC_H5         dataset path (required)
  TRAFFIC_SPLIT      patient | abundance | clone8020 (required)
  TRAFFIC_PATIENT    patient id to hold out (required for split=patient)
  TRAFFIC_OUT        output file (required)
  TRAFFIC_X64        1=float64, 0=float32 (default 0)
  TRAFFIC_WARMUP/SAMPLES/CHAINS   NUTS config (defaults 1000/1000/2)
"""
import hashlib
import os

import jax

jax.config.update("jax_enable_x64", os.environ.get("TRAFFIC_X64", "0") != "0")
import numpy as np

from traffic import LikelihoodConfig, MCMCConfig, PriorConfig, data, io, mcmc, statespace


def _clone_bucket(clones):
    """Stable per-clone bucket in [0,100) (hashlib, so reproducible across runs)."""
    return np.array(
        [int(hashlib.md5(str(c).encode()).hexdigest(), 16) % 100 for c in clones]
    )


def compute_masks(obs, split, patient=None):
    """Return (train_mask, test_mask, rule_dict) for the requested split."""
    if split == "patient":
        if not patient:
            raise SystemExit("split=patient needs TRAFFIC_PATIENT")
        train = obs.patient != patient
        test = obs.patient == patient
        rule = {"split": "patient", "held_out_patient": patient}
    elif split == "abundance":
        train = obs.n_src >= 5
        test = obs.n_src <= 2
        rule = {"split": "abundance", "train_rule": "n_src>=5", "test_rule": "n_src<=2"}
    elif split == "clone8020":
        b = _clone_bucket(obs.clone)
        train = b < 80
        test = b >= 80
        rule = {"split": "clone8020", "hash": "md5(trb)%100", "train_rule": "bucket<80",
                "test_rule": "bucket>=80"}
    else:
        raise SystemExit(f"unknown TRAFFIC_SPLIT={split}")
    return train, test, rule


def main():
    h5 = os.environ.get("TRAFFIC_H5")
    if not h5:
        raise SystemExit("set TRAFFIC_H5 to the dataset path")
    split = os.environ.get("TRAFFIC_SPLIT")
    if not split:
        raise SystemExit("set TRAFFIC_SPLIT (patient|abundance|clone8020)")
    out = os.environ.get("TRAFFIC_OUT")
    if not out:
        raise SystemExit("set TRAFFIC_OUT to the output path")
    patient = os.environ.get("TRAFFIC_PATIENT")

    ss = statespace.default()
    obs_df = data.load_obs_table(h5)
    obs = data.build(obs_df, ss)
    rho = data.phenotype_dist_by_tissue(obs_df, ss)

    train, test, rule = compute_masks(obs, split, patient)
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
    print(f"split={split} {rule}  backend={jax.default_backend()}  likelihood={lik.family}/{lik.dispersion}  "
          f"J_total={obs.Xtilde.shape[0]}  J_train={int(train.sum())}  J_test={int(test.sum())}  "
          f"warmup={cfg.num_warmup} samples={cfg.num_samples} chains={cfg.num_chains}")

    res = mcmc.fit_nuts(X, Y, D, PriorConfig(), cfg, lik=lik, patient_idx=patient_idx)
    print(f"divergences={res.num_divergences}  r_hat_max={res.r_hat_max:.3f}  "
          f"ess_min={res.ess_min:.0f}")

    meta = dict(rule)
    meta.update({
        "likelihood": lik.family, "dispersion": lik.dispersion,
        "J_total": int(obs.Xtilde.shape[0]),
        "J_train": int(train.sum()),
        "J_test": int(test.sum()),
        "backend": jax.default_backend(),
        "warmup": cfg.num_warmup, "samples": cfg.num_samples, "chains": cfg.num_chains,
    })
    path = io.save_fit(out, res, ss=ss, rho=rho, meta=meta)
    print(f"saved {path}  ({res.n_draws} draws of M[{ss.L},{ss.L}])")


if __name__ == "__main__":
    main()
