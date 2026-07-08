"""traffic -- Bayesian population-dynamics model for clonal T-cell trafficking across
tissue and phenotype, with a factored one-step operator M = diag(g).(pi (x) Phi) and a
Negative-Binomial likelihood (JAX + NumPyro / NUTS).

Executable form of docs/model_methods.tex.
"""
from . import model, recovery, sbc, simulate, statespace
from .config import FactoredPriorConfig, MCMCConfig
from .statespace import StateSpace
from .statespace import default as default_statespace

__all__ = [
    "StateSpace", "default_statespace",
    "FactoredPriorConfig", "MCMCConfig",
    "model", "recovery", "sbc", "simulate", "statespace",
]

# Optional modules (import lazily to keep the core light).
from . import dynamics, io, posterior  # noqa: E402
__all__ += ["dynamics", "io", "posterior"]

try:
    from . import readouts  # noqa: F401
    __all__.append("readouts")
except Exception:  # pragma: no cover
    pass
try:
    from . import data  # noqa: F401  (needs pandas/anndata)
    __all__.append("data")
except Exception:  # pragma: no cover
    pass
try:
    from . import mcmc  # noqa: F401  (needs numpyro)
    from .mcmc import MCMCResult, fit_nuts
    __all__ += ["mcmc", "fit_nuts", "MCMCResult"]
except Exception:  # pragma: no cover
    pass
