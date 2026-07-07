"""Posterior uncertainty helpers (parameterization-agnostic).

The factored model samples {g, pi, Phi, log phi} directly by NUTS, so per-draw uncertainty is read
straight off the draws (or off any deterministic read-out of them) -- there are no variational
shape/rate factors to summarize. This module keeps only the generic interval helper.
"""
from __future__ import annotations

import numpy as np


def credible_interval(samples_of_scalar, level=0.95):
    """mean + (lo, hi) percentile interval for a 1-D array of posterior draws."""
    x = np.asarray(samples_of_scalar)
    lo = (1 - level) / 2 * 100
    return float(x.mean()), float(np.percentile(x, lo)), float(np.percentile(x, 100 - lo))
