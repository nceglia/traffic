"""Forward dynamics + flux read-outs from the fitted growth matrix M-hat.

Two operators:
  - M-hat itself (growth: free row sums) -> mass grows/shrinks while it moves;
  - T = row-normalized M-hat (pure redistribution, mass-conserving) -> "where
    does mass go", the scale-free trafficking operator. Iterating T answers the
    drainage / unrolling question without the depth-driven growth-scale artifact.
"""
from __future__ import annotations

import numpy as np

from .readouts import decompose


def tissue_collapse(vec, ss):
    """Sum a length-L state vector into length-S tissue totals."""
    K = ss.K
    return np.array([np.asarray(vec)[a * K:(a + 1) * K].sum() for a in range(ss.S)])


def forward_iterate(p0, A, steps):
    """Iterate row-vector mass: p_{t+1} = p_t A. Returns [steps+1, L]."""
    p = np.asarray(p0, float)
    out = [p]
    for _ in range(steps):
        p = p @ A
        out.append(p)
    return np.array(out)


def steady_state(T):
    """Stationary distribution of a row-stochastic T (left eigvec, eigval 1)."""
    vals, vecs = np.linalg.eig(np.asarray(T).T)
    i = int(np.argmin(np.abs(vals - 1.0)))
    pi = np.real(vecs[:, i])
    pi = np.abs(pi)
    return pi / pi.sum()


def tissue_mass_flux(T, ss, p0_tissue=None):
    """Mass-weighted directed tissue->tissue flux F[a,b] (rows = source tissue).

    Weights source states within tissue a by the supplied per-tissue occupancy
    p0_tissue[a] x (uniform phenotypes) -> here we just use the row-stochastic T
    aggregated, i.e. average over source phenotypes. Returns F [S,S] of the
    fraction of tissue-a mass arriving in tissue b (rows sum to 1).
    """
    S, K = ss.S, ss.K
    F = np.zeros((S, S))
    for a in range(S):
        block_rows = np.asarray(T)[a * K:(a + 1) * K]            # [K, L]
        for b in range(S):
            F[a, b] = block_rows[:, b * K:(b + 1) * K].sum(1).mean()
    return F


def net_flux(F, occupancy):
    """Net tissue-tissue flux given current tissue occupancy [S].

    net[a,b] = occ[a] F[a,b] - occ[b] F[b,a]  (positive => net a->b).
    """
    occ = np.asarray(occupancy, float)
    gross = occ[:, None] * np.asarray(F)
    return gross - gross.T


def staging_scores(F, occupancy):
    """Per-tissue inflow, outflow, residence (for the 'staging' question).

    inflow[b]  = sum_{a!=b} occ[a] F[a,b]
    outflow[a] = occ[a] sum_{b!=a} F[a,b]
    residence[a] = F[a,a]
    """
    occ = np.asarray(occupancy, float)
    S = F.shape[0]
    off = F - np.diag(np.diag(F))
    inflow = (occ[:, None] * off).sum(0)
    outflow = occ * off.sum(1)
    residence = np.diag(F)
    return {"inflow": inflow, "outflow": outflow, "residence": residence}


def cycle_strength(F, occupancy):
    """Net circulation around the 3-tissue loop (A->B->C->A), a scalar.

    For 3 tissues, circulation = mean of (gross_ab + gross_bc + gross_ca) -
    (gross_ba + gross_cb + gross_ac); sign gives loop direction.
    """
    occ = np.asarray(occupancy, float)
    g = occ[:, None] * np.asarray(F)
    S = F.shape[0]
    fwd = sum(g[i, (i + 1) % S] for i in range(S))
    rev = sum(g[(i + 1) % S, i] for i in range(S))
    return float(fwd - rev)
