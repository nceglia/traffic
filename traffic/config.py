"""Configuration dataclasses (leaf module)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FactoredPriorConfig:
    """Priors for the factored operator  M = diag(g) . (pi (x) Phi).

    Each source state z=(a,u) draws three factors, assembled into the mean matrix
    M[z, b*K+v] = g[z] * pi[z,b] * Phi[z,b,v]:
      g_{a,u}       ~ LogNormal(mu_g, sigma_g^2)   per-state EXPANSION factor (centred ~1, heavy tail)
      pi_{(a,u)}    ~ Dirichlet(alpha_a)           destination-tissue distribution (trafficking)
      Phi_{(a,u),b} ~ Dirichlet(beta * 1)          destination-phenotype distribution (switching)
    plus a single global NB2 concentration  log phi ~ Normal(0, sigma_phi^2).

    Persistence prior: the tissue Dirichlet gives the stay column (b=a) concentration `alpha_stay`
    and the off-tissue columns `alpha_off`, so a priori a clone's descendants STAY rather than
    scatter (induced E[pi(stay)] = alpha_stay / (alpha_stay + (S-1) alpha_off)). `alpha_stay` is a
    scalar (global) or length-S (per source tissue, e.g. a lower stay for CSF as a transit
    compartment). `beta` is the symmetric switching concentration.
    """
    mu_g: float = 0.0            # LogNormal location for g (centred ~1)
    sigma_g: float = 0.5        # LogNormal scale (heavy right tail)
    alpha_off: float = 1.0      # Dirichlet conc. on off-tissue destinations of pi
    alpha_stay: "float | tuple" = 4.0   # stay (b=a) conc.; scalar=global, length-S=per source tissue
    beta: float = 1.0           # symmetric Dirichlet conc. for Phi (switching)
    sigma_phi: float = 1.0      # Normal-prior scale on log phi (NB2 concentration)


@dataclass(frozen=True)
class MCMCConfig:
    """NUTS settings for the factored posterior over {g, pi, Phi, log phi}."""
    num_warmup: int = 800
    num_samples: int = 800
    num_chains: int = 2
    target_accept: float = 0.9
    chain_method: str = "vectorized"  # parallel chains via vmap on one device
    seed: int = 0
    progress_bar: bool = False
