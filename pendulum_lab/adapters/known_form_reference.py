"""Known-form reference (Person B): least squares on sin(theta) and v only.

Told the answer's shape, so it is a pipeline diagnostic, not a discovery
method. Same boundary and output schema as the real adapters, labelled
known_form_reference everywhere.

On clean data it must recover -4 and -0.2 to numerical precision. If it
doesn't, suspect column order, units or scaling before blaming SINDy or
AI Feynman.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pendulum_lab import contracts

LABEL = "known_form_reference"


def fit(train_path, output_dir, seed=0, settings=None):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    X, y = contracts.check_train(train_path)
    A = np.column_stack([np.sin(X[:, 0]), X[:, 1]])     # no intercept
    (c1, c2), *_ = np.linalg.lstsq(A, y, rcond=None)
    cand = [{
        "candidate_id": LABEL,
        "raw_expression": json.dumps({"sin(theta)": float(c1), "v": float(c2)}),
        "expression": f"{float(c1)!r}*(sin(theta)) + {float(c2)!r}*(v)",
        "method_settings": {"label": LABEL, "terms": ["sin(theta)", "v"],
                            "intercept": False},
        "diagnostics": {"coefficients": [float(c1), float(c2)],
                        "train_predictions_rms": float(np.sqrt(np.mean((A @ [c1, c2]) ** 2)))},
    }]
    status = {"status": "ok", "fit_time": time.perf_counter() - t0, "message": "",
              "method": LABEL, "method_version": f"numpy {np.__version__}",
              "seed": int(seed)}
    contracts.write_json_atomic(out / "candidates.json", contracts.check_candidates(cand))
    contracts.write_json_atomic(out / "fit_status.json", contracts.check_fit_status(status))
    return float(c1), float(c2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("train_path")
    ap.add_argument("output_dir")
    a = ap.parse_args()
    c1, c2 = fit(a.train_path, a.output_dir)
    print(json.dumps({"label": LABEL, "sin(theta)": c1, "v": c2}))
