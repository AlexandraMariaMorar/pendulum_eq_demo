"""The SINDy dictionary: 14 candidate terms, in the recorded order (Person B).

The names are the same strings as `sindy.terms` in configs/experiment.yaml,
and every name is also a valid SymPy expression in theta and v, so a fitted
coefficient vector turns straight into an equation string.
"""
from __future__ import annotations

import numpy as np

# name -> numpy function of (theta, v). Order matters: it is the column order.
TERMS = {
    "1":            lambda th, v: np.ones_like(th),
    "theta":        lambda th, v: th,
    "v":            lambda th, v: v,
    "theta**2":     lambda th, v: th**2,
    "theta*v":      lambda th, v: th * v,
    "v**2":         lambda th, v: v**2,
    "theta**3":     lambda th, v: th**3,
    "theta**2*v":   lambda th, v: th**2 * v,
    "theta*v**2":   lambda th, v: th * v**2,
    "v**3":         lambda th, v: v**3,
    "sin(theta)":   lambda th, v: np.sin(th),
    "cos(theta)":   lambda th, v: np.cos(th),
    "sin(2*theta)": lambda th, v: np.sin(2 * th),
    "cos(2*theta)": lambda th, v: np.cos(2 * th),
}
TERM_NAMES = list(TERMS)


def check_terms(config_terms):
    """The config and this file must list the same terms in the same order."""
    if list(config_terms) != TERM_NAMES:
        raise ValueError(f"config terms {config_terms} differ from features.TERMS")


def feature_matrix(X, names=TERM_NAMES) -> np.ndarray:
    """Phi with shape (N, len(names)); X has columns [theta, v]."""
    th, v = X[:, 0], X[:, 1]
    return np.column_stack([TERMS[n](th, v) for n in names]).astype(np.float64)


def expression_string(coefs, names=TERM_NAMES) -> str:
    """'c1*(term1) + c2*(term2) + ...' with full float precision.

    repr() of a Python float round-trips exactly, so no precision is lost
    between the fitted vector and the saved equation. Zero terms are dropped;
    an all-zero vector gives "0".
    """
    parts = [f"{float(c)!r}*({n})" for c, n in zip(coefs, names) if c != 0.0]
    return " + ".join(parts) if parts else "0"
