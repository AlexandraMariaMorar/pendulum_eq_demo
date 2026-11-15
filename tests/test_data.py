"""Checks for pendulum_lab.data (Person B). Run: python -m pytest -q tests/test_data.py"""
import json

import numpy as np
import pytest

from pendulum_lab import contracts, data

CFG_PATH = "configs/experiment.yaml"


@pytest.fixture(scope="module")
def cfg():
    return data.load_config(CFG_PATH)


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    out = tmp_path_factory.mktemp("data")
    manifest = data.generate_all(CFG_PATH, out)
    return out, manifest


# ---- reference physics ---------------------------------------------------
def test_zero_state_stays_zero(cfg):
    _, X = data.simulate((0.0, 0.0), cfg)
    assert np.all(X == 0.0)


def test_energy_never_increases(cfg):
    _, X = data.simulate((2.0, 0.0), cfg)
    a = cfg["physics"]["g_over_l"]
    E = 0.5 * X[:, 1] ** 2 + a * (1 - np.cos(X[:, 0]))
    assert np.max(np.diff(E)) < 1e-9          # dE/dtau = -0.2 v^2 <= 0


def test_tighter_tolerance_barely_changes_reference(cfg):
    for s in cfg["initial_states"]["train"]:
        _, A = data.simulate(s, cfg)
        _, B = data.simulate(s, cfg, rtol=1e-12, atol=1e-14)
        assert np.max(np.abs(A - B)) < 1e-8


def test_grid(cfg):
    t, X = data.simulate((1.0, 0.0), cfg)
    assert t.shape == (501,) and X.shape == (501, 2) and t[-1] == 10.0


# ---- splits and files ----------------------------------------------------
def test_row_counts_and_schema(generated, cfg):
    out, _ = generated
    for cid, cond in cfg["conditions"].items():
        for seed in cfg["seeds"]["repetitions"]:
            d = out / "benchmark" / cid / f"seed{seed}"
            contracts.check_train(d / "train.npz", expected_rows=cond["n"])
            contracts.check_train(d / "validation.npz", is_validation=True)
            contracts.check_training_file_has_no_test_labels(d / "train.npz")
    contracts.check_test(out / "benchmark" / "test.npz")


def test_disjoint_trajectory_ids(generated):
    out, _ = generated
    d = out / "benchmark" / "c00" / "seed101"
    tr = set(np.load(d / "train.npz")["trajectory_id"])
    va = set(np.load(d / "validation.npz")["trajectory_id"])
    assert tr == set(data.TRAIN_IDS) and va == set(data.VAL_IDS) and not tr & va


def test_subsets_are_nested(generated):
    out, _ = generated
    for seed in (101, 102, 103):
        rows = []
        for cid in ("c04", "c05", "c06", "c00"):          # N = 120, 300, 600, 1200
            z = np.load(out / "benchmark" / cid / f"seed{seed}" / "train.npz")
            rows.append(set(zip(z["trajectory_id"], z["sample_index"])))
        assert rows[0] < rows[1] < rows[2] < rows[3]


def test_every_bin_equally_covered(generated):
    out, _ = generated
    z = np.load(out / "benchmark" / "c04" / "seed101" / "train.npz")   # N = 120
    for tid in data.TRAIN_IDS:
        bins = z["sample_index"][z["trajectory_id"] == tid] // 50
        assert np.all(np.bincount(bins, minlength=10) == 2)


def test_same_noise_draws_across_levels(generated):
    out, _ = generated
    d = lambda c: np.load(out / "benchmark" / c / "seed101" / "train.npz")
    clean, r1, r3 = d("c00"), d("c01"), d("c03")
    assert np.array_equal(clean["X"], r3["X"])           # states stay exact
    e1 = (r1["y"] - clean["y"]) / 0.005
    e3 = (r3["y"] - clean["y"]) / 0.05
    assert np.allclose(e1, e3, rtol=1e-6, atol=1e-9)     # same epsilon, scaled


def test_seeds_differ_and_regeneration_is_identical(generated, tmp_path):
    out, manifest = generated
    h = manifest["datasets"]
    assert h["benchmark/c00/seed101"]["train"] != h["benchmark/c00/seed102"]["train"]
    again = data.generate_all(CFG_PATH, tmp_path)
    assert again["datasets"] == manifest["datasets"]
    assert again["test_hash"] == manifest["test_hash"]


def test_metadata(generated):
    out, manifest = generated
    m = json.loads((out / "benchmark" / "c03" / "seed103" / "metadata.json").read_text())
    assert m["n"] == 1200 and m["noise_r"] == 0.05 and m["validation_rows"] == 200
    assert m["scale_s"] == manifest["scale_s"]


def test_pilot_rollout_has_test_shape_and_separate_states(generated, cfg):
    """The pilot file passes the same check as test.npz, and never reuses a state."""
    from pendulum_lab import contracts
    out, _ = generated
    t, X, acc = contracts.check_test(out / "pilot" / "pilot_rollout.npz")
    assert X.shape == (4, 501, 2)
    ini = cfg["initial_states"]
    others = {tuple(s) for k in ("train", "validation", "test") for s in ini[k]}
    assert not others & {tuple(s) for s in ini["pilot_rollout"]}
