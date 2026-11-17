"""AI Feynman adapter: saved dataset -> AI Feynman subprocess -> candidates.json + fit_status.json."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path
import re


import numpy as np

from pendulum_lab.contracts import (
    ContractError, check_train, check_training_file_has_no_test_labels,
    hash_npz, write_json_atomic,
)

METHOD = "aifeynman"
INPUT_NAME = "data.txt"
VARIABLE_MAP = {"x0": "theta", "x1": "v"}


def prepare_input(train_path, work_dir):
    """Check the training file and write AI Feynman's input table: theta, v, y."""
    check_training_file_has_no_test_labels(train_path)
    X, y = check_train(train_path)
    work_dir = Path(work_dir)
    (work_dir / "input").mkdir(parents=True, exist_ok=True)
    table = np.column_stack([X[:, 0], X[:, 1], y])
    np.savetxt(work_dir / "input" / INPUT_NAME, table, fmt="%.17g")
    return {
        "input_file": f"input/{INPUT_NAME}",
        "columns": ["theta", "v", "y"],
        "variable_map": VARIABLE_MAP,
        "rows": int(X.shape[0]),
        "train_hash": hash_npz(train_path),
    }


def run_runner(work_dir, settings, python_path, budget_seconds):
    """Start the runner in its own process group; stop the whole group at the budget."""
    work_dir = Path(work_dir)
    write_json_atomic(work_dir / "run_settings.json", settings)
    runner = Path(__file__).with_name("aifeynman_runner.py")
    python_path = os.path.abspath(python_path)
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(python_path) + os.pathsep + env.get("PATH", "")
    t0 = time.time()
    with open(work_dir / "stdout.log", "w") as out, open(work_dir / "stderr.log", "w") as err:
        proc = subprocess.Popen(
            [python_path, "-u", str(runner), "run_settings.json"],
            cwd=work_dir, stdout=out, stderr=err, env=env, start_new_session=True,
        )
        try:
            proc.wait(timeout=budget_seconds)
            timed_out = False
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            timed_out = True
    return {
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "fit_time": round(time.time() - t0, 3),
    }

SOLUTION_COLUMNS = ["test_error", "mean_log2_error", "total_log2_error", "complexity_bits", "error"]


def to_theta_v(raw):
    """Rename AI Feynman's x0, x1 to theta, v (whole names only)."""
    expr = raw
    for name, ours in VARIABLE_MAP.items():
        expr = re.sub(rf"\b{name}\b", ours, expr)
    return expr

def _is_number(text):
    try:
        float(text)
        return True
    except ValueError:
        return False

def parse_solution(solution_path, test_percentage, settings):
    """Turn every row of results/solution_<file> into one candidate dict."""
    cols = SOLUTION_COLUMNS if test_percentage > 0 else SOLUTION_COLUMNS[1:]
    candidates = []
    for i, line in enumerate(Path(solution_path).read_text().splitlines()):
        if not line.strip():
            continue
        parts = line.split(None, len(cols))
        if len(parts) != len(cols) + 1:
            raise ContractError(f"solution row {i}: expected {len(cols) + 1} fields, got {len(parts)}")
        if not all(_is_number(p) for p in parts[:-1]):
            raise ContractError(f"solution row {i}: a numeric column is not a number")
        raw = parts[-1].strip()
        candidates.append({
            "candidate_id": f"aifeynman_row{i}",
            "raw_expression": raw,
            "expression": to_theta_v(raw),
            "method_settings": settings,
            "diagnostics": {
                "solution_row": i,
                "solution_columns": dict(zip(cols, parts[:-1])),
                "variable_map": VARIABLE_MAP,
            },
        })
    return candidates

def fit(train_path, output_dir, seed, settings, python_path, budget_seconds):
    """Plan's adapter boundary: writes candidates.json and fit_status.json, even on failure."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_settings = dict(settings, model_seed=int(seed), input_name=INPUT_NAME)
    status = {"method": METHOD, "status": "error", "fit_time": None, "message": "",
              "settings": run_settings, "python_path": str(python_path),
              "budget_seconds": budget_seconds}
    candidates = []
    try:
        status["input"] = prepare_input(train_path, output_dir)
        proc = run_runner(output_dir, run_settings, python_path, budget_seconds)
        status.update(proc)
        solution = output_dir / "results" / f"solution_{INPUT_NAME}"
        if solution.exists():
            candidates = parse_solution(solution, run_settings["test_percentage"], run_settings)
        if proc["timed_out"]:
            status["status"] = "timeout"
            status["message"] = f"stopped at the {budget_seconds} s budget; {len(candidates)} candidates found"
        elif proc["returncode"] != 0:
            status["status"] = "error"
            status["message"] = f"runner exited with code {proc['returncode']}; see stderr.log"
        elif not candidates:
            status["status"] = "empty_output"
            status["message"] = "runner finished but wrote no candidates"
        else:
            status["status"] = "ok"
            status["message"] = f"{len(candidates)} candidates"
    except ContractError as e:
        if status.get("timed_out"):
            status["status"] = "timeout"
            status["message"] = f"stopped at the {budget_seconds} s budget; partial output unreadable: {e}"
        else:
            status["status"] = "parse_error"
            status["message"] = str(e)

    except Exception as e:
        status["status"] = "error"
        status["message"] = f"{type(e).__name__}: {e}"
    finally:
        write_json_atomic(output_dir / "candidates.json", candidates)
        write_json_atomic(output_dir / "fit_status.json", status)
    return status
