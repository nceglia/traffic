"""Save / load fitted posteriors so a heavy fit (e.g. full-data NUTS on a GPU
node) can be pulled down and reloaded locally for read-outs and figures.

Portable single-file ``.npz`` -- loads with **numpy only**, no JAX or GPU needed.
Works for both backends:
  * NUTS  -> stores the M draws (the expensive artifact);
  * CAVI  -> stores the Gamma shape/rate (draws regenerate locally on load).
``load_fit`` returns a :class:`SavedFit` with a uniform ``.draws()`` so the same
read-out code drives either backend.

    # GPU node
    res = mcmc.fit_nuts(Xtilde, Y, D)
    io.save_fit("nuts_fit.npz", res, ss=ss, rho=rho)
    # locally, after scp
    fit = io.load_fit("nuts_fit.npz")
    draws = fit.draws(2000)          # [2000, L, L]
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .statespace import StateSpace


@dataclass
class SavedFit:
    kind: str                       # 'nuts' | 'cavi'
    M_hat: np.ndarray               # [L, L] posterior mean
    ss: StateSpace
    samples: np.ndarray | None = None   # [N, L, L]  (nuts)
    shape: np.ndarray | None = None     # [L, L]     (cavi Gamma shape)
    rate: np.ndarray | None = None      # [L, L]     (cavi Gamma rate)
    rho: np.ndarray | None = None       # [S, K]     phenotype dist by tissue (optional)
    meta: dict = field(default_factory=dict)

    def draws(self, n: int = 2000, seed: int = 0) -> np.ndarray:
        """[n, L, L] posterior draws of M -- stored NUTS draws, or freshly sampled
        CAVI Gammas. numpy only (no JAX), so it runs on a plain local machine."""
        if self.samples is not None:
            if n is None or n >= self.samples.shape[0]:
                return self.samples
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(self.samples.shape[0], n, replace=False))
            return self.samples[idx]
        rng = np.random.default_rng(seed)
        # Gamma(shape, rate) == Gamma(shape, scale=1/rate)
        return rng.gamma(self.shape, 1.0 / self.rate, size=(n,) + self.shape.shape)


def save_fit(path, result, *, ss: StateSpace, rho=None, meta=None) -> str:
    """Write a CAVI ``FitResult`` or NUTS ``MCMCResult`` to a portable ``.npz``."""
    payload, info = {}, dict(meta or {})
    if hasattr(result, "samples"):                       # NUTS MCMCResult
        kind = "nuts"
        payload["samples"] = np.asarray(result.samples, np.float32)
        payload["M_hat"] = np.asarray(result.M_hat, np.float64)
        info.update(num_divergences=int(getattr(result, "num_divergences", -1)),
                    r_hat_max=float(getattr(result, "r_hat_max", float("nan"))),
                    ess_min=float(getattr(result, "ess_min", float("nan"))),
                    n_draws=int(result.samples.shape[0]))
    else:                                                # CAVI FitResult
        kind = "cavi"
        payload["shape"] = np.asarray(result.posterior.shape, np.float64)
        payload["rate"] = np.asarray(result.posterior.rate, np.float64)
        payload["M_hat"] = np.asarray(result.M_hat, np.float64)
    payload["kind"] = np.array(kind)
    payload["tissues"] = np.array(list(ss.tissues))
    payload["phenotypes"] = np.array(list(ss.phenotypes))
    if rho is not None:
        payload["rho"] = np.asarray(rho, np.float64)
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
    return SavedFit(
        kind=str(z["kind"]),
        M_hat=z["M_hat"],
        ss=ss,
        samples=z["samples"] if "samples" in files else None,
        shape=z["shape"] if "shape" in files else None,
        rate=z["rate"] if "rate" in files else None,
        rho=z["rho"] if "rho" in files else None,
        meta=json.loads(str(z["meta"])) if "meta" in files else {},
    )
