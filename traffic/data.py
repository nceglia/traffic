"""Build clone-level forward observations from the GBM T-cell AnnData.

The model only needs obs metadata (no expression matrix), so we read the
.h5ad backed and aggregate clone x timepoint x tissue x phenotype counts.

For each observation j = (patient i, source timepoint r, clone c) with the
clone present at r:
    x_irc(z)      = #cells of clone c in (i, r, tissue(z), phenotype(z))
    xtilde_irc(z) = x_irc(z) / d_{i,r,tissue(z)}        depth-normalized source
    y_irc(z')     = #cells of clone c in (i, r+1, ...)  destination counts
    d_{j,z'}      = d_{i,r+1,tissue(z')}                destination depth (broadcast)
    m_{j,z'}      = 1{ (i,r+1,tissue(z')) profiled }    missingness

Depth d_{i,r,s} = number of TCR+ cells in that (patient,timepoint,tissue) sample.
Steps pair consecutive integer timepoints r -> r+1 (both profiled for patient i).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .statespace import StateSpace, default as default_ss

OBS_COLS = ["patient", "timepoint", "tissue", "phenotype", "trb"]


@dataclass
class Observations:
    Xtilde: np.ndarray   # [J, L] depth-rescaled source
    Y: np.ndarray        # [J, L] destination counts (0 at missing)
    D: np.ndarray        # [J, L] destination depth (0 at missing)
    mask: np.ndarray     # [J, L] 1 where profiled
    X: np.ndarray        # [J, L] raw source counts
    n_src: np.ndarray    # [J] total source cells (clone size at source)
    patient: np.ndarray  # [J] patient id (str)
    src_tp: np.ndarray   # [J] source timepoint (int)
    clone: np.ndarray    # [J] clonotype id (str)
    ss: StateSpace
    meta: dict

    @property
    def theta(self):
        """Clone source composition (raw source normalized to a distribution), [J, L]."""
        return self.X / np.maximum(self.X.sum(1, keepdims=True), 1e-12)


def load_obs_table(h5ad_path, *, backed=True):
    """Return a DataFrame of the obs columns the model needs (TCR+ rows only)."""
    import anndata as ad
    import pandas as pd

    a = ad.read_h5ad(str(h5ad_path), backed="r" if backed else None)
    o = a.obs[OBS_COLS].copy()
    o = o.dropna(subset=["trb", "timepoint", "tissue", "phenotype"])
    o["patient"] = o["patient"].astype(str)
    o["tissue"] = o["tissue"].astype(str)
    o["phenotype"] = o["phenotype"].astype(str)
    o["trb"] = o["trb"].astype(str)
    o["timepoint"] = o["timepoint"].astype(str).astype(int)
    return o


def phenotype_dist_by_tissue(obs_df, ss: StateSpace) -> np.ndarray:
    """rho^a(u): pooled source-cell phenotype distribution per tissue, [S, K]."""
    rho = np.zeros((ss.S, ss.K))
    g = obs_df.groupby(["tissue", "phenotype"], observed=True).size()
    for (t, p), n in g.items():
        if t in ss.tissues and p in ss.phenotypes:
            rho[ss.tissues.index(t), ss.phenotypes.index(p)] += n
    rho = rho / np.maximum(rho.sum(1, keepdims=True), 1)
    return rho


def build(obs_df, ss: StateSpace = None, *, min_src_cells: int = 1) -> Observations:
    """Assemble forward observations into [J, L] arrays."""
    ss = ss or default_ss()
    L, K = ss.L, ss.K
    tissue_of = ss.tissue_of  # [L]

    df = obs_df[obs_df["tissue"].isin(ss.tissues) & obs_df["phenotype"].isin(ss.phenotypes)].copy()
    tix = df["tissue"].map({t: i for i, t in enumerate(ss.tissues)}).to_numpy()
    pix = df["phenotype"].map({p: i for i, p in enumerate(ss.phenotypes)}).to_numpy()
    df["z"] = tix * K + pix

    # (patient, clone, timepoint) x z  -> count matrix
    g = df.groupby(["patient", "trb", "timepoint", "z"], observed=True).size().reset_index(name="n")
    keys = g[["patient", "trb", "timepoint"]].drop_duplicates().reset_index(drop=True)
    key2row = {k: r for r, k in enumerate(map(tuple, keys.to_numpy()))}
    rows = g[["patient", "trb", "timepoint"]].apply(tuple, axis=1).map(key2row).to_numpy()
    src = np.zeros((len(keys), L))
    src[rows, g["z"].to_numpy()] = g["n"].to_numpy()

    # depth d_{i,r,s} and per-patient profiled timepoints
    dep = df.groupby(["patient", "timepoint", "tissue"], observed=True).size()
    depth = {(p, int(tp), t): int(n) for (p, tp, t), n in dep.items()}
    tps_present = df.groupby("patient")["timepoint"].apply(lambda s: set(int(x) for x in s.unique())).to_dict()
    tissues = list(ss.tissues)

    Xt, Y, D, M, Xraw, nsrc, pat, stp, cln = [], [], [], [], [], [], [], [], []
    for (p, c, tp), r in key2row.items():
        dtp = tp + 1
        if dtp not in tps_present.get(p, ()):  # patient must have a destination sample
            continue
        x = src[r]
        if x.sum() < min_src_cells:
            continue
        dsrc = np.array([depth.get((p, tp, t), 0) for t in tissues], float)[tissue_of]
        xt = np.where(dsrc > 0, x / np.maximum(dsrc, 1.0), 0.0)

        rdst = key2row.get((p, c, dtp))
        y = src[rdst].copy() if rdst is not None else np.zeros(L)
        ddst = np.array([depth.get((p, dtp, t), 0) for t in tissues], float)[tissue_of]
        m = (ddst > 0).astype(float)
        y = y * m
        ddst = ddst * m

        Xt.append(xt); Y.append(y); D.append(ddst); M.append(m)
        Xraw.append(x.copy()); nsrc.append(float(x.sum()))
        pat.append(p); stp.append(tp); cln.append(c)

    meta = {
        "n_obs": len(Xt),
        "n_clones": int(df["trb"].nunique()),
        "n_cells": int(len(df)),
        "samples_present": len(depth),
        "samples_possible": len(set(df["patient"])) * 6 * ss.S,
        "depth_summary": {
            "min": int(min(depth.values())), "median": int(np.median(list(depth.values()))),
            "max": int(max(depth.values())),
        },
    }
    return Observations(
        Xtilde=np.asarray(Xt), Y=np.asarray(Y), D=np.asarray(D), mask=np.asarray(M),
        X=np.asarray(Xraw), n_src=np.asarray(nsrc),
        patient=np.asarray(pat), src_tp=np.asarray(stp), clone=np.asarray(cln),
        ss=ss, meta=meta,
    )


def from_h5ad(h5ad_path, ss: StateSpace = None, *, min_src_cells: int = 1):
    """Convenience: load obs table + build observations + rho^a."""
    ss = ss or default_ss()
    obs_df = load_obs_table(h5ad_path)
    obs = build(obs_df, ss, min_src_cells=min_src_cells)
    rho = phenotype_dist_by_tissue(obs_df, ss)
    obs.meta["rho"] = rho
    return obs
