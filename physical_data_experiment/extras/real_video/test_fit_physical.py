"""Checks for fit_physical.py (Person B, real-video extra).

    python -m pytest -q extras/real_video/test_fit_physical.py

Uses the synthetic mode: a simulated recording with the same frame times and
start states as the real one, so the right answer is known.
"""
import sys
from pathlib import Path

import numpy as np
import sympy as sp

sys.path.insert(0, str(Path(__file__).parent))
import fit_physical as F    # noqa: E402


def test_uses_the_core_sindy_code_and_settings():
    from pendulum_lab import features
    assert F.TERMS == features.TERM_NAMES
    assert F.VARIANTS["core"]["alpha"] == F.CORE["alpha"] == 0.0
    assert F.VARIANTS["core"]["thresholds"] == F.CORE["thresholds"]


def test_noise_free_synthetic_is_recovered(tmp_path, monkeypatch):
    """With exact angles the whole pipeline (smoothing, fits, selection) works."""
    monkeypatch.setattr(F, "OUT", tmp_path)          # don't touch results/
    rec = F.run("136g", "synthetic", "core", noise=0.0)
    a_true, b_true, _ = F.TRUTH["136g"]
    kf = sp.sympify(rec["chosen"]["known form"]["expr"])
    th, v = F.TH, F.V
    assert abs(float(kf.coeff(sp.sin(th))) + a_true) < 0.01 * a_true
    assert abs(float(kf.coeff(v)) + b_true) < 0.1 * b_true
    for m in ("SINDy", "known form"):
        assert max(rec["chosen"][m]["test_err"].values()) < 0.01


def test_sparsest_sindy_candidate_is_the_restoring_force(tmp_path, monkeypatch):
    """At the strictest core threshold, noise-free SINDy keeps only sin(theta)."""
    monkeypatch.setattr(F, "OUT", tmp_path)
    rec = F.run("090g", "synthetic", "core", noise=0.0)
    strict = [c for c in rec["all_candidates"] if c["setting"] == "threshold 0.1"][0]
    e = sp.sympify(strict["expr"])
    assert e.free_symbols == {F.TH} and e.has(sp.sin(F.TH))
    assert abs(float(e.coeff(sp.sin(F.TH))) + F.TRUTH["090g"][0]) < 0.05
