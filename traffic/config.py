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
class MCMCConfig:
    """NUTS settings for the marginal posterior over M.

    Samples M directly from p(M | data): the per-source allocations marginalize
    exactly by Poisson superposition, so the target is the smooth L x L posterior
    with no auxiliary variables.
    """
    num_warmup: int = 800
    num_samples: int = 800
    num_chains: int = 2
    target_accept: float = 0.9
    chain_method: str = "vectorized"  # parallel chains via vmap on one device
    seed: int = 0
    progress_bar: bool = False
