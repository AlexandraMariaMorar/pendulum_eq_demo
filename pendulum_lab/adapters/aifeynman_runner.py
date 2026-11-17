"""Runs inside .venv-aifeynman: read run_settings.json, seed, call AI Feynman.

Started by aifeynman_adapter.py as a subprocess with the run folder as the
working directory. Imports nothing from pendulum_lab on purpose.
"""
import json
import random
import sys
import time

import numpy as np
import torch
import aifeynman


def main(settings_path):
    s = json.load(open(settings_path))
    seed = int(s["model_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(s["threads"]))
    t0 = time.time()
    aifeynman.run_aifeynman(
        "./input/", s["input_name"], s["bf_try_time"], s["ops_file"],
        polyfit_deg=s["polyfit_deg"], NN_epochs=s["nn_epochs"],
        test_percentage=s["test_percentage"],
    )
    print("RUNNER_DONE", round(time.time() - t0, 3), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
