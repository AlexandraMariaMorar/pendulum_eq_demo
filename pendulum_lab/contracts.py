"""Shared contracts: schema checks, run identifiers and canonical array hashes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
VALIDATION_ROWS = 200
TEST_STATES = 501
TEST_TRAJECTORIES = 4
METHODS = ("aifeynman", "sindy")

TRAIN_KEYS = ("X", "y", "trajectory_id", "sample_index", "time")
TEST_KEYS = ("time", "X", "acceleration")
CANDIDATE_KEYS = ("candidate_id", "raw_expression", "expression", "method_settings", "diagnostics")
FIT_STATUS_VALUES = ("ok", "timeout", "error", "empty_output", "parse_error")
RUN_RECORD_KEYS = (
    "schema_version", "experiment_id", "condition_id", "run_id", "attempt_id",
    "method", "method_version", "git_commit", "config_hash", "data_hash",
    "data_seed", "model_seed", "n", "noise", "hardware",
    "fit_time", "selection_time", "evaluation_time", "selected_candidate_id",
    "fit_status", "evaluation_status",
)


class ContractError(ValueError):
    """Raised when data or records violate the agreed schema."""


# ---------- hashing ----------

def hash_array(a: np.ndarray) -> str:
    """SHA-256 of canonical contents: dtype, shape and C-ordered bytes."""
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(json.dumps(list(a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def hash_arrays(arrays: dict) -> str:
    """Order-independent hash of named arrays."""
    h = hashlib.sha256()
    for key in sorted(arrays):
        h.update(key.encode())
        h.update(hash_array(arrays[key]).encode())
    return h.hexdigest()


def hash_npz(path) -> str:
    """Hash array contents of an .npz file, ignoring container timestamps."""
    with np.load(path) as z:
        return hash_arrays({k: z[k] for k in z.files})


def hash_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hash_config(path) -> str:
    return hash_file(path)


# ---------- identifiers ----------

def run_id(condition_id: str, method: str, seed: int) -> str:
    if method not in METHODS:
        raise ContractError(f"unknown method {method!r}")
    return f"{condition_id}_{method}_seed{int(seed)}"


def attempt_id(rid: str, attempt: int) -> str:
    if attempt < 1:
        raise ContractError("attempt numbers start at 1")
    return f"{rid}_a{int(attempt)}"


# ---------- dataset checks ----------

def _check_float64(name, a, ndim, shape_tail=None):
    if a.dtype != np.float64:
        raise ContractError(f"{name} must be float64, got {a.dtype}")
    if a.ndim != ndim:
        raise ContractError(f"{name} must have {ndim} dims, got shape {a.shape}")
    if shape_tail is not None and a.shape[1:] != shape_tail:
        raise ContractError(f"{name} has shape {a.shape}, expected (*, {shape_tail})")
    if not np.all(np.isfinite(a)):
        raise ContractError(f"{name} contains nonfinite values")


def check_train(path, expected_rows=None, is_validation=False):
    """Validate a train.npz / validation.npz file. Returns (X, y)."""
    with np.load(path) as z:
        missing = [k for k in TRAIN_KEYS if k not in z.files]
        if missing:
            raise ContractError(f"{path}: missing keys {missing}")
        d = {k: z[k] for k in TRAIN_KEYS}
    _check_float64("X", d["X"], 2, (2,))
    _check_float64("y", d["y"], 1)
    n = d["X"].shape[0]
    if d["y"].shape[0] != n:
        raise ContractError("y length does not match X")
    for k in ("trajectory_id", "sample_index", "time"):
        if d[k].ndim != 1 or d[k].shape[0] != n:
            raise ContractError(f"{k} must be 1-D with {n} rows")
    if is_validation and n != VALIDATION_ROWS:
        raise ContractError(f"validation must have {VALIDATION_ROWS} rows, got {n}")
    if expected_rows is not None and n != expected_rows:
        raise ContractError(f"expected {expected_rows} rows, got {n}")
    return d["X"], d["y"]


def check_test(path):
    """Validate test.npz. Only the evaluator should call this."""
    with np.load(path) as z:
        missing = [k for k in TEST_KEYS if k not in z.files]
        if missing:
            raise ContractError(f"{path}: missing keys {missing}")
        t, X, acc = z["time"], z["X"], z["acceleration"]
    _check_float64("time", t, 1)
    _check_float64("X", X, 3, (TEST_STATES, 2))
    _check_float64("acceleration", acc, 2, (TEST_STATES,))
    if t.shape != (TEST_STATES,) or X.shape[0] != TEST_TRAJECTORIES or acc.shape[0] != TEST_TRAJECTORIES:
        raise ContractError("test arrays have wrong shapes")
    return t, X, acc


def check_training_file_has_no_test_labels(path):
    """Adapter-side guard: training files must not carry test arrays."""
    with np.load(path) as z:
        extra = [k for k in z.files if k in ("acceleration", "clean_acceleration")]
    if extra:
        raise ContractError(f"{path}: contains clean-label keys {extra}")


# ---------- JSON records ----------

def check_candidates(obj):
    """Validate the content of candidates.json (a list of candidate dicts)."""
    if not isinstance(obj, list):
        raise ContractError("candidates.json must hold a list")
    seen = set()
    for c in obj:
        missing = [k for k in CANDIDATE_KEYS if k not in c]
        if missing:
            raise ContractError(f"candidate missing keys {missing}")
        if c["candidate_id"] in seen:
            raise ContractError(f"duplicate candidate_id {c['candidate_id']}")
        seen.add(c["candidate_id"])
        if not isinstance(c["expression"], str):
            raise ContractError("expression must be a string")
    return obj


def check_fit_status(obj):
    for k in ("status", "fit_time", "message"):
        if k not in obj:
            raise ContractError(f"fit_status missing {k}")
    if obj["status"] not in FIT_STATUS_VALUES:
        raise ContractError(f"bad fit status {obj['status']!r}")
    return obj


def check_run_record(rec):
    missing = [k for k in RUN_RECORD_KEYS if k not in rec]
    if missing:
        raise ContractError(f"run record missing keys {missing}")
    if rec["schema_version"] != SCHEMA_VERSION:
        raise ContractError("schema_version mismatch")
    _no_nan(rec)
    return rec


def _no_nan(x, path="record"):
    if isinstance(x, float) and not np.isfinite(x):
        raise ContractError(f"{path} is NaN/inf; use null with an explanation")
    if isinstance(x, dict):
        for k, v in x.items():
            _no_nan(v, f"{path}.{k}")
    if isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            _no_nan(v, f"{path}[{i}]")


def write_json_atomic(path, obj):
    """Write JSON via temp file + rename so interrupted runs leave no partial record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)
