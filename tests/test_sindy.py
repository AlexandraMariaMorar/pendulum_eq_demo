"""Checks for features.py and the SINDy / known-form adapters (Person B)."""
import json

import numpy as np
import pytest
import sympy as sp

from pendulum_lab import data, features
from pendulum_lab.adapters import known_form_reference, sindy_adapter

CFG_PATH = "configs/experiment.yaml"
TH, V = sp.symbols("theta v")


@pytest.fixture(scope="module")
def cfg():
    return data.load_config(CFG_PATH)


@pytest.fixture(scope="module")
def pilot(tmp_path_factory):
    out = tmp_path_factory.mktemp("data")
    data.generate_all(CFG_PATH, out)
    return out / "pilot"


def eval_expr(expr, X):
    f = sp.lambdify((TH, V), sp.sympify(expr), "numpy")
    return np.broadcast_to(f(X[:, 0], X[:, 1]), (len(X),))


def test_config_terms_match_features(cfg):
    features.check_terms(cfg["sindy"]["terms"])


def test_known_vector_survives_scaling():
    """Scaled fit + back-transform returns a constructed coefficient vector."""
    X = np.random.default_rng(0).uniform(-2, 2, size=(400, 2))
    Phi = features.feature_matrix(X)
    c_true = np.zeros(14)
    c_true[[2, 6, 10]] = [-0.2, 0.05, -4.0]
    c, w, q = sindy_adapter.scale_fit(Phi, Phi @ c_true, s=2.1, threshold=0.0,
                                      alpha=0.0, max_iter=20)
    assert np.allclose(c, c_true, atol=1e-8)
    assert np.allclose(c, 2.1 * w / q)


def test_expression_matches_matrix_predictions():
    """The saved string predicts exactly what Phi @ c predicts."""
    X = np.random.default_rng(1).uniform(-2, 2, size=(50, 2))
    c = np.random.default_rng(2).normal(size=14)
    expr = features.expression_string(c)
    assert np.allclose(eval_expr(expr, X), features.feature_matrix(X) @ c,
                       rtol=1e-12, atol=1e-12)


def test_sindy_clean_pilot(pilot, cfg, tmp_path):
    st = sindy_adapter.fit(pilot / "c00" / "seed999" / "train.npz", tmp_path, 999,
                           cfg["sindy"])
    assert st["status"] == "ok"
    cands = json.loads((tmp_path / "candidates.json").read_text())
    assert len(cands) == len(cfg["sindy"]["thresholds"])
    sparse = cands[-1]["diagnostics"]["coefficients"]      # threshold 0.1
    names = features.TERM_NAMES
    assert np.isclose(sparse[names.index("sin(theta)")], -4.0, rtol=1e-9)
    assert np.isclose(sparse[names.index("v")], -0.2, rtol=1e-9)
    assert cands[-1]["diagnostics"]["n_terms"] == 2


def test_known_form_reference_clean(pilot, tmp_path):
    c1, c2 = known_form_reference.fit(pilot / "c00" / "seed999" / "train.npz", tmp_path)
    assert abs(c1 + 4.0) < 1e-9 and abs(c2 + 0.2) < 1e-9


def test_adapter_reports_errors_instead_of_crashing(tmp_path, cfg):
    bad = tmp_path / "train.npz"
    np.savez(bad, X=np.zeros((3, 3)), y=np.zeros(3))      # wrong schema
    st = sindy_adapter.fit(bad, tmp_path / "out", 0, cfg["sindy"])
    assert st["status"] == "error"
    assert json.loads((tmp_path / "out" / "candidates.json").read_text()) == []


def test_no_numerical_differentiation_used():
    import inspect
    src = inspect.getsource(sindy_adapter)
    # Labels are supplied; the high-level model (which differentiates) is never built.
    assert "differentiation_method" not in src and "ps.SINDy(" not in src
