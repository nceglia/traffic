"""Configuration dataclasses (leaf module)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PriorConfig:
    """Per-entry Gamma(a0, b0) prior on M_{zz'} (shape, rate).

    a0 = b0 = 1 puts a weak Exponential(1) prior on every growth-matrix entry
    (prior mean 1 = "one descendant per source cell", prior over-dispersed).
    """
    a0: float = 1.0
    b0: float = 1.0


@dataclass(frozen=True)
class CAVIConfig:
    """Coordinate-ascent settings for the conjugate Gamma-Poisson fit."""
    max_iter: int = 300
    tol: float = 1e-6      # stop when max relative change in M-hat < tol
    eps: float = 1e-12     # numerical floor


@dataclass(frozen=True)
class MCMCConfig:
    """NUTS settings for the marginal Gamma-Poisson posterior over M.

    Samples M directly from p(M | data): the per-source allocations marginalize
    exactly by Poisson superposition (model_methods.tex Section 5), so the target
    is the smooth L x L posterior with no auxiliary variables. Unlike CAVI, the
    sample spread is the full (non-factorized) posterior uncertainty.
    """
    num_warmup: int = 800
    num_samples: int = 800
    num_chains: int = 2
    target_accept: float = 0.9
    chain_method: str = "vectorized"  # parallel chains via vmap on one device
    seed: int = 0
    progress_bar: bool = False
