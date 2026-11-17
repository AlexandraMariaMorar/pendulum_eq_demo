"""Reference simulation and saved datasets (Person B).

Everything the two methods are trained and validated on is created here,
once, from configs/experiment.yaml, and saved to disk. Adapters only ever
read these files; they never simulate or resample anything themselves.

    python -m pendulum_lab.data --config configs/experiment.yaml --out data

(Person A's run.py can call generate_all() for its `generate` command.)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.integrate import solve_ivp

from pendulum_lab import contracts

GENERATOR_VERSION = 1

# Fixed stream numbers: each seed gets independent random streams for each job,
# so changing how many numbers one job draws can never shift another job's draws.
STREAMS = {"subsample": 0, "noise_train": 1, "noise_val": 2, "model": 3}

# Trajectory IDs are unique across splits, so "disjoint splits" is checkable.
TRAIN_IDS = range(0, 6)
VAL_IDS = range(6, 8)
TEST_IDS = range(8, 12)


# --------------------------------------------------------------------------
# Config and physics
# --------------------------------------------------------------------------
def load_config(path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def time_grid(cfg) -> np.ndarray:
    sv = cfg["solver"]
    t = np.linspace(0.0, sv["tau_max"], sv["n_states"])
    assert np.allclose(np.diff(t), sv["dt"])
    return t


def reference_acceleration(theta, v, cfg):
    """v_dot = -a*sin(theta) - b*v: the synthetic ground truth."""
    p = cfg["physics"]
    return -p["g_over_l"] * np.sin(theta) - p["damping"] * v


def simulate(initial_state, cfg, accel=None, rtol=None, atol=None):
    """Integrate [theta, v] on the config's output grid.

    accel(theta, v) defaults to the reference rule. Returns (time, states)
    with states of shape (n_states, 2), columns [theta, v].
    """
    sv = cfg["solver"]
    t = time_grid(cfg)
    if accel is None:
        def accel(th, v):
            return reference_acceleration(th, v, cfg)

    def rhs(_, y):
        return [y[1], accel(y[0], y[1])]

    sol = solve_ivp(rhs, (t[0], t[-1]), list(map(float, initial_state)),
                    method=sv["method"], t_eval=t,
                    rtol=sv["rtol"] if rtol is None else rtol,
                    atol=sv["atol"] if atol is None else atol)
    if sol.status != 0:
        raise RuntimeError(f"reference simulation failed: {sol.message}")
    return t, np.ascontiguousarray(sol.y.T, dtype=np.float64)


def simulate_set(states, cfg):
    return np.stack([simulate(s, cfg)[1] for s in states])   # (n_traj, 501, 2)


def rng(seed: int, stream: str) -> np.random.Generator:
    return np.random.default_rng([int(seed), STREAMS[stream]])


def model_seed(seed: int) -> int:
    """Seed for an algorithm's own randomness, from its own stream."""
    return int(rng(seed, "model").integers(0, 2**31 - 1))


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------
def training_pool(cfg):
    """Six training trajectories without tau = 10: 500 rows each."""
    t = time_grid(cfg)
    traj = simulate_set(cfg["initial_states"]["train"], cfg)[:, :-1, :]  # (6, 500, 2)
    acc = reference_acceleration(traj[..., 0], traj[..., 1], cfg)           # (6, 500)
    return t[:-1], traj, acc


def validation_indices(cfg) -> np.ndarray:
    """100 evenly spaced indices per validation trajectory (0, 5, ..., 495).

    Same tau < 10 range as training. The plan only says "evenly spaced";
    record this choice in metadata and in the PR so Person A can confirm.
    """
    k = cfg["sampling"]["validation_rows_per_trajectory"]
    pool = cfg["solver"]["n_states"] - 1
    return np.arange(0, pool, pool // k)


def bin_orders(cfg, seed):
    """For every (trajectory, bin): one random order of its 50 row indices."""
    sp = cfg["sampling"]
    per_bin = sp["rows_per_bin_total"]
    g = rng(seed, "subsample")
    n_traj = len(cfg["initial_states"]["train"])
    return np.array([[b * per_bin + g.permutation(per_bin)
                      for b in range(sp["bins_per_trajectory"])]
                     for _ in range(n_traj)])                  # (6, 10, 50)


def select_rows(order, k):
    """First k rows of every bin. Returns (trajectory_index, sample_index)."""
    n_traj, n_bins, _ = order.shape
    ti = np.repeat(np.arange(n_traj), n_bins * k)
    si = order[:, :, :k].reshape(-1)
    keep = np.lexsort((si, ti))           # sort by trajectory, then time
    return ti[keep], si[keep]


# --------------------------------------------------------------------------
# Writing one condition
# --------------------------------------------------------------------------
def _save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return contracts.hash_npz(path)


def write_condition(out_dir, cfg, cond_id, seed, pool, val, s, config_hash):
    t_pool, traj, acc = pool
    cond = cfg["conditions"][cond_id]
    n, r = int(cond["n"]), float(cond["r"])
    k = int(cfg["sampling"]["per_bin_counts"][n])

    order = bin_orders(cfg, seed)
    ti, si = select_rows(order, k)
    # One noise draw per pool row, made once per seed: the same row gets the
    # same epsilon at every r and every N, so conditions differ only by r and N.
    eps_train = rng(seed, "noise_train").standard_normal(acc.shape)
    X = traj[ti, si, :]
    y = acc[ti, si] + r * s * eps_train[ti, si]

    v_traj, v_idx, v_X, v_acc = val
    eps_val = rng(seed, "noise_val").standard_normal(v_acc.shape)
    v_y = v_acc + r * s * eps_val

    d = Path(out_dir) / cond_id / f"seed{seed}"
    h_train = _save_npz(d / "train.npz", X=X, y=y,
                        trajectory_id=np.asarray(TRAIN_IDS)[ti].astype(np.int64),
                        sample_index=si.astype(np.int64), time=t_pool[si])
    h_val = _save_npz(d / "validation.npz", X=v_X, y=v_y,
                      trajectory_id=v_traj.astype(np.int64),
                      sample_index=v_idx.astype(np.int64), time=t_pool[v_idx])
    contracts.check_train(d / "train.npz", expected_rows=n)
    contracts.check_train(d / "validation.npz", is_validation=True)
    contracts.check_training_file_has_no_test_labels(d / "train.npz")
    meta = {
        "schema_version": contracts.SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "experiment_id": cfg["experiment_id"],
        "condition_id": cond_id, "seed": int(seed), "n": n, "noise_r": r,
        "noise_model": "y = clean + r * s * eps, eps ~ N(0,1) per row, fixed per seed",
        "scale_s": s, "scale_ddof": cfg["noise"]["scale_ddof"],
        "validation_rows": int(len(v_y)),
        "validation_index_rule": "arange(0, 500, 5) per validation trajectory",
        "random_streams": STREAMS,
        "model_seed": model_seed(seed),
        "config_hash": config_hash,
        "hashes": {"train": h_train, "validation": h_val},
    }
    contracts.write_json_atomic(d / "metadata.json", meta)
    return meta


def generate_all(config_path, out_dir="data"):
    cfg = load_config(config_path)
    config_hash = contracts.hash_config(config_path)
    out = Path(out_dir)
    t = time_grid(cfg)

    pool = training_pool(cfg)
    s = float(np.std(pool[2], ddof=cfg["noise"]["scale_ddof"]))

    idx = validation_indices(cfg)
    vtraj = simulate_set(cfg["initial_states"]["validation"], cfg)[:, idx, :]   # (2, 100, 2)
    v_X = vtraj.reshape(-1, 2)
    val = (np.repeat(np.asarray(VAL_IDS), len(idx)), np.tile(idx, len(VAL_IDS)), v_X,
           reference_acceleration(v_X[:, 0], v_X[:, 1], cfg))

    # Clean test set: only the evaluator reads it.
    test = simulate_set(cfg["initial_states"]["test"], cfg)
    h_test = _save_npz(out / "benchmark" / "test.npz", time=t, X=test,
                       acceleration=reference_acceleration(test[..., 0], test[..., 1], cfg))
    contracts.check_test(out / "benchmark" / "test.npz")
    # Pilot rollout states: used while tuning, so the final test stays unseen.
    # Four of them, so the file passes the same contract check as test.npz.
    pil = simulate_set(cfg["initial_states"]["pilot_rollout"], cfg)
    h_pilot = _save_npz(out / "pilot" / "pilot_rollout.npz", time=t, X=pil,
                        acceleration=reference_acceleration(pil[..., 0], pil[..., 1], cfg))
    contracts.check_test(out / "pilot" / "pilot_rollout.npz")   # same shape as test.npz

    manifest = {"config_hash": config_hash, "scale_s": s, "test_hash": h_test,
                "pilot_rollout_hash": h_pilot, "datasets": {}}
    for group, seeds in (("benchmark", cfg["seeds"]["repetitions"]),
                         ("pilot", [cfg["seeds"]["pilot"]])):
        for cond_id in cfg["conditions"]:
            for seed in seeds:
                m = write_condition(out / group, cfg, cond_id, seed, pool, val, s, config_hash)
                manifest["datasets"][f"{group}/{cond_id}/seed{seed}"] = m["hashes"]
    contracts.write_json_atomic(out / "manifest.json", manifest)
    print(f"s = {s:.6f}; wrote {len(manifest['datasets'])} datasets, "
          f"test.npz and pilot_rollout.npz under {out}/")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate all saved datasets.")
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    generate_all(a.config, a.out)
