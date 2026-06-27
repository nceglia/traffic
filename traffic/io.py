"""Save / load a fitted NUTS posterior so a heavy GPU-node fit can be pulled down
and reloaded locally for read-outs and figures.

Portable single-file ``.npz`` -- loads with **numpy only**, no JAX or GPU needed.
Stores the M draws (the expensive artifact); ``load_fit`` returns a
:class:`SavedFit` with ``.draws()``.

    res = mcmc.fit_nuts(Xtilde, Y, D); io.save_fit("nuts_fit.npz", res, ss=ss, rho=rho)
    fit = io.load_fit("nuts_fit.npz"); draws = fit.draws(2000)   # [2000, L, L]
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .statespace import StateSpace


@dataclass
class SavedFit:
    M_hat: np.ndarray               # [L, L] posterior mean
    samples: np.ndarray             # [N, L, L] posterior draws of M
    ss: StateSpace
    rho: np.ndarray | None = None   # [S, K] phenotype dist by tissue (optional)
    meta: dict = field(default_factory=dict)
    dispersion: dict | None = None  # {"mode": str, "params": {site: [N,...]}} for NB fits

    def draws(self, n: int = 2000, seed: int = 0) -> np.ndarray:
        """[n, L, L] posterior draws of M (a random subset if n < n_draws). numpy only."""
        if n is None or n >= self.samples.shape[0]:
            return self.samples
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(self.samples.shape[0], n, replace=False))
        return self.samples[idx]


def save_fit(path, result, *, ss: StateSpace, rho=None, meta=None) -> str:
    """Write a NUTS ``MCMCResult`` (Poisson or NB) to a portable ``.npz``."""
    if not hasattr(result, "samples"):
        raise TypeError("save_fit expects an MCMCResult with .samples (a NUTS fit)")
    info = dict(meta or {})
    info.update(num_divergences=int(getattr(result, "num_divergences", -1)),
                r_hat_max=float(getattr(result, "r_hat_max", float("nan"))),
                ess_min=float(getattr(result, "ess_min", float("nan"))),
                n_draws=int(result.samples.shape[0]))
    payload = {
        "samples": np.asarray(result.samples, np.float32),
        "M_hat": np.asarray(result.M_hat, np.float64),
        "tissues": np.array(list(ss.tissues)),
        "phenotypes": np.array(list(ss.phenotypes)),
    }
    if rho is not None:
        payload["rho"] = np.asarray(rho, np.float64)
    disp = getattr(result, "dispersion", None)
    if disp:
        info["family"], info["dispersion_mode"] = "nb", disp["mode"]
        for k, arr in disp["params"].items():
            payload[f"disp_{k}"] = np.asarray(arr, np.float32)
    else:
        info["family"] = "poisson"
    payload["meta"] = np.array(json.dumps(info))
    path = str(path)
    if not path.endswith(".npz"):
        path += ".npz"
    np.savez_compressed(path, **payload)
    return path


def load_fit(path) -> SavedFit:
    """Reload a :class:`SavedFit` (numpy only -- no JAX/GPU)."""
    z = np.load(str(path), allow_pickle=False)
    files = set(z.files)
    ss = StateSpace(tissues=tuple(str(t) for t in z["tissues"]),
                    phenotypes=tuple(str(p) for p in z["phenotypes"]))
    meta = json.loads(str(z["meta"])) if "meta" in files else {}
    disp_keys = [f for f in files if f.startswith("disp_")]
    dispersion = ({"mode": meta.get("dispersion_mode"),
                   "params": {f[5:]: z[f] for f in disp_keys}} if disp_keys else None)
    return SavedFit(
        M_hat=z["M_hat"], samples=z["samples"], ss=ss,
        rho=z["rho"] if "rho" in files else None, meta=meta, dispersion=dispersion,
    )
