"""traffic -- Bayesian Gamma-Poisson population-dynamics model for clonal
T-cell trafficking across tissue and phenotype (JAX, conjugate CAVI).

Executable form of gbm_trafficking/docs/model_methods.tex (v3).
"""
from . import model, simulate, statespace
from .config import CAVIConfig, MCMCConfig, PriorConfig
from .inference import FitResult, Posterior, fit
from .statespace import StateSpace
from .statespace import default as default_statespace

__all__ = [
    "StateSpace", "default_statespace",
    "PriorConfig", "CAVIConfig", "MCMCConfig",
    "fit", "Posterior", "FitResult",
    "model", "simulate", "statespace",
]

# Optional modules (added incrementally; import lazily to keep core light).
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
    __all__.append("mcmc")
except Exception:  # pragma: no cover
    pass
