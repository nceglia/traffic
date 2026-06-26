"""traffic -- Bayesian Gamma-Poisson population-dynamics model for clonal
T-cell trafficking across tissue and phenotype (JAX + NumPyro / NUTS).

Executable form of gbm_trafficking/docs/model_methods.tex.
"""
from . import model, simulate, statespace
from .config import MCMCConfig, PriorConfig
from .statespace import StateSpace
from .statespace import default as default_statespace

__all__ = [
    "StateSpace", "default_statespace",
    "PriorConfig", "MCMCConfig",
    "model", "simulate", "statespace",
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
