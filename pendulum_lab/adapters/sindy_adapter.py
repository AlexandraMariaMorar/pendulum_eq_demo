"""SINDy adapter (Person B): PySINDy's STLSQ on an explicit 14-term matrix.

Adapter boundary shared with the AI Feynman adapter:
    fit(train_path, output_dir, seed, settings)
        -> output_dir/candidates.json and output_dir/fit_status.json

One candidate per threshold. Choosing between them is NOT done here: the
shared selector (Person A, evaluate.py) does that on the validation file.

    python -m pendulum_lab.adapters.sindy_adapter data/pilot/c00/seed999/train.npz runs/pilot_sindy
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

from pendulum_lab import contracts, features
from pendulum_lab.data import load_config


def scale_fit(Phi, y, s, threshold, alpha, max_iter, unbias=True):
    """One STLSQ fit with the plan's manual scaling.

    Z[:, j] = Phi[:, j] / q[j],  target y / s,  then c[j] = s * w[j] / q[j].
    Returns (c, w, q).
    """
    import pysindy as ps

    q = np.sqrt(np.mean(Phi**2, axis=0))          # training RMS of each column
    if not np.all(q > 0):
        raise ValueError("a dictionary column is identically zero on this data")
    Z = Phi / q
    opt = ps.STLSQ(threshold=threshold, alpha=alpha, max_iter=max_iter,
                   normalize_columns=False,       # manual RMS scaling instead
                   unbias=unbias)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")           # threshold=0 ill-conditioning notices
        opt.fit(Z, (y / s).reshape(-1, 1))
    w = opt.coef_.ravel().astype(np.float64)
    return s * w / q, w, q


def fit(train_path, output_dir, seed, settings):
    """settings: the config's `sindy` block, plus `scale_s` if not in metadata."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    status = {"status": "ok", "fit_time": None, "message": "",
              "method": "sindy", "seed": int(seed)}
    candidates = []
    try:
        import pysindy as ps
        status["method_version"] = f"pysindy {ps.__version__}"
        contracts.check_training_file_has_no_test_labels(train_path)
        X, y = contracts.check_train(train_path)
        meta_path = Path(train_path).with_name("metadata.json")
        s = settings.get("scale_s") or json.loads(meta_path.read_text())["scale_s"]
        features.check_terms(settings["terms"])
        Phi = features.feature_matrix(X)

        for k, lam in enumerate(settings["thresholds"]):
            c, w, q = scale_fit(Phi, y, s, lam, settings["alpha"],
                                settings["max_iter"], settings["unbias"])
            pred = Phi @ c
            candidates.append({
                "candidate_id": f"sindy_t{k}",
                "raw_expression": json.dumps({"scaled_weights": w.tolist()}),
                "expression": features.expression_string(c),
                "method_settings": {"threshold": lam, "alpha": settings["alpha"],
                                    "max_iter": settings["max_iter"],
                                    "normalize_columns": False,
                                    "unbias": settings["unbias"]},
                "diagnostics": {"terms": features.TERM_NAMES,
                                "coefficients": c.tolist(),
                                "column_rms_q": q.tolist(), "scale_s": s,
                                "n_terms": int(np.count_nonzero(c)),
                                "train_nrmse": float(np.sqrt(np.mean((pred - y) ** 2)) / s)},
            })
            if time.perf_counter() - t0 > settings.get("fit_cap_s", 900):
                status["status"] = "timeout"
                status["message"] = f"stopped after threshold index {k}"
                break
        if not candidates:
            status["status"] = "empty_output"
    except Exception as err:
        status["status"] = "error"
        status["message"] = f"{err!r}\n{traceback.format_exc()}"
    status["fit_time"] = time.perf_counter() - t0

    contracts.check_candidates(candidates)
    contracts.check_fit_status(status)
    contracts.write_json_atomic(out / "candidates.json", candidates)
    contracts.write_json_atomic(out / "fit_status.json", status)
    return status


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fit SINDy candidates to one train.npz")
    ap.add_argument("train_path")
    ap.add_argument("output_dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="configs/experiment.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    st = fit(a.train_path, a.output_dir, a.seed,
             {**cfg["sindy"], "fit_cap_s": cfg["budgets_seconds"]["fit"]})
    print(json.dumps({k: st[k] for k in ("status", "fit_time", "method_version")
                      if k in st}))
