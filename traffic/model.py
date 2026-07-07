"""Mean-count spec for the factored operator (pure JAX helpers).

The one-step mean matrix M = diag(g).(pi (x) Phi) is assembled in mcmc.assemble; this
module holds the deterministic mean map used downstream by read-outs / scoring
(see gbm_trafficking/docs/model_methods.tex):

    mu_j       = xtilde_j @ M                     destination intensity vector
    E[y_j(z')] = d_{j,z'} * mu_j(z')              NB2 mean over non-missing states

Missingness is folded into the arrays: at a missing destination state the data builder
sets both Y[j,z'] = 0 and D[j,z'] = 0, so that term drops out of the likelihood.
"""
from __future__ import annotations


def intensity(xtilde, M):
    """mu_j = xtilde_j @ M, shape [J, L]."""
    return xtilde @ M


def expected_counts(xtilde, M, D):
    """E[y_j(z')] = d_{j,z'} * mu_j(z'), shape [J, L]."""
    return D * (xtilde @ M)
