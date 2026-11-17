"""Fit the real pendulum recordings (after-the-core extra, Person B).

    python extras/real_video/fit_physical.py                     # real recordings, core SINDy settings
    python extras/real_video/fit_physical.py --synthetic         # same pipeline on simulated "video"
    python extras/real_video/fit_physical.py --synthetic --noise 0   # pipeline check: noise-free
    python extras/real_video/fit_physical.py --sindy-variant ridge   # sensitivity check (see below)
    python extras/real_video/fit_physical.py --mass 136g         # one mass
Run from the repository root (it imports pendulum_lab).

Design, mirroring the simulated benchmark as closely as real data allows:
  * TRAIN on one long release, TEST on the other releases of the same bob.
    A different release is a genuinely new starting state, like test.npz.
  * VALIDATE on the last 25% (in time) of the training release. Never random
    rows: neighbouring frames are near copies, so a random split would leak.
  * Labels: the camera gives only angles, so velocity and acceleration come
    from a Savitzky-Golay filter (a polynomial fitted in a sliding window).
    This is the "noisy derivatives" problem the core deliberately avoids.
  * Methods: SINDy through the core's own tested code (pendulum_lab.features
    and sindy_adapter.scale_fit, i.e. pysindy STLSQ with the manual RMS
    scaling) and, by default, the core's settings from configs/experiment.yaml.
    Also the known form a*sin(theta) + b*v and the known form plus drag v*|v|.
  * --sindy-variant ridge adds a ridge penalty and a threshold of 0.3. These
    were chosen AFTER seeing first results on these recordings, so they are a
    labelled sensitivity check, never the headline result.
  * --synthetic replaces each recording by a simulated one: same start state,
    same frame times, true equation from the known-form fit, plus Gaussian
    angle noise at the level measured in the video. It runs the identical
    pipeline, so it separates "the code/settings fail" from "real video fails".
  * Selection, per method: simulate each candidate across the validation
    stretch; keep the simplest within max(0.005, 10% of that method's best). Frozen before
    looking at the test releases.
Writes extras/real_video/results/<mass>_<source>_<variant>.json and .png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import sympy as sp                       # noqa: E402
from scipy.integrate import solve_ivp    # noqa: E402
from scipy.signal import savgol_filter   # noqa: E402

from pendulum_lab import data as core_data, features          # noqa: E402
from pendulum_lab.adapters.sindy_adapter import scale_fit      # noqa: E402

HERE = Path(__file__).parent
DATA = HERE / "physical_data"
OUT = HERE / "results"
G = 9.81
SETUPS = {"090g": {"train": "exp2", "test": ["exp3", "exp4"]},     # exp1 is one swing only
          "136g": {"train": "exp1", "test": ["exp2"]}}
WINDOW_S, ORDER = 0.3, 5           # smoothing window and polynomial order
VAL_FRACTION = 0.25
CORE = core_data.load_config("configs/experiment.yaml")["sindy"]
VARIANTS = {   # alpha is pysindy's ridge penalty on the RMS-scaled problem
    "core":  {"alpha": CORE["alpha"], "thresholds": CORE["thresholds"]},
    "ridge": {"alpha": None, "thresholds": CORE["thresholds"] + [0.3]},   # alpha = 0.05 * rows
}
# Synthetic control: true equation per bob = the known-form fit on the real
# training release (rounded), noise = measured raw-minus-smoothed angle spread.
TRUTH = {"090g": (29.83, 0.0208, 0.0022), "136g": (24.31, 0.0239, 0.0009)}

TH, V = sp.symbols("theta v")
features.check_terms(CORE["terms"])
TERMS = features.TERM_NAMES
SYM = [sp.sympify(n, locals={"theta": TH, "v": V}) for n in TERMS]
FUN = [sp.lambdify((TH, V), s, "numpy") for s in SYM]


def library(th, v, syms=SYM, funs=FUN):
    return np.column_stack([np.broadcast_to(f(th, v), th.shape) for f in funs]).astype(float)


# ---------- data ----------
def load(mass, exp):
    d = np.loadtxt(DATA / f"mass{mass}_{exp}.csv", delimiter=",", comments="#", skiprows=4)
    return d[:, 0], np.unwrap(d[:, 3])                  # t_s, theta_rad


def synthetic(mass, exp, rng, noise=None):
    """Simulated recording with the real one's frame times and start state."""
    t, theta = load(mass, exp)
    k, th, v, _ = smooth(t, theta)
    a_, b_, measured = TRUTH[mass]
    noise = measured if noise is None else noise
    f = lambda _, y: [y[1], -a_ * np.sin(y[0]) - b_ * y[1]]
    i0 = k.start
    fwd = solve_ivp(f, (t[i0], t[-1]), [th[i0], v[i0]], t_eval=t[i0:], method="DOP853",
                    rtol=1e-10, atol=1e-12).y[0]
    back = solve_ivp(f, (t[i0], t[0]), [th[i0], v[i0]], t_eval=t[i0::-1], method="DOP853",
                     rtol=1e-10, atol=1e-12).y[0][::-1]
    clean = np.r_[back[:-1], fwd]
    return t, clean + noise * rng.standard_normal(len(t))


def smooth(t, theta):
    dt = float(np.median(np.diff(t)))
    w = max(int(round(WINDOW_S / dt)) | 1, ORDER + 2)
    th = savgol_filter(theta, w, ORDER, deriv=0, delta=dt)
    v = savgol_filter(theta, w, ORDER, deriv=1, delta=dt)
    a = savgol_filter(theta, w, ORDER, deriv=2, delta=dt)
    k = slice(w // 2, len(t) - w // 2)                  # filter edges are unreliable
    return k, th, v, a


# ---------- fitting ----------
def to_expr(c, syms):
    return sum((sp.Float(float(ci), 6) * s for ci, s in zip(c, syms) if ci != 0), sp.Integer(0))


def candidates(th, v, a, variant):
    out = []
    Phi = features.feature_matrix(np.column_stack([th, v]))
    s = float(np.std(a))
    cfg = VARIANTS[variant]
    alpha = 0.05 * len(a) if cfg["alpha"] is None else cfg["alpha"]
    for thr in cfg["thresholds"]:
        c, _, _ = scale_fit(Phi, a, s, thr, alpha, CORE["max_iter"], CORE["unbias"])
        out.append({"method": "SINDy", "setting": f"threshold {thr}",
                    "expr": to_expr(c, SYM)})
    for name, syms in (("known form", [sp.sin(TH), V]),
                       ("known form + drag", [sp.sin(TH), V, V * sp.Abs(V)])):
        X = library(th, v, syms, [sp.lambdify((TH, V), s, "numpy") for s in syms])
        c = np.linalg.lstsq(X, a, rcond=None)[0]
        out.append({"method": name, "setting": "least squares", "expr": to_expr(c, syms)})
    return out


# ---------- prediction ----------
def simulate(expr, th0, v0, t):
    f = sp.lambdify((TH, V), expr, "numpy")
    def rhs(_, y):
        return [y[1], float(f(y[0], y[1]))]
    def blowup(_, y):
        return 50 - abs(y[0])
    blowup.terminal = True
    sol = solve_ivp(rhs, (t[0], t[-1]), [th0, v0], t_eval=t, method="DOP853",
                    rtol=1e-9, atol=1e-11, events=blowup)
    if sol.status != 0 or sol.y.shape[1] != len(t):
        return None
    return sol.y[0]


def angle_error(pred, meas):
    if pred is None:
        return np.inf
    return float(np.sqrt(np.mean((pred - meas) ** 2)) / np.sqrt(np.mean(meas**2)))


def run(mass, source="real", variant="core", noise=None):
    OUT.mkdir(exist_ok=True)
    setup = SETUPS[mass]
    rng = np.random.default_rng(0)
    get = (lambda e: load(mass, e)) if source == "real" else (lambda e: synthetic(mass, e, rng, noise))
    t, theta = get(setup["train"])
    k, th, v, a = smooth(t, theta)
    idx = np.arange(len(t))[k]
    cut = int((1 - VAL_FRACTION) * len(t))
    tr, va = idx[idx < cut], idx[idx >= cut]
    cands = candidates(th[tr], v[tr], a[tr], variant)

    # Validation: simulate from the first validation state, compare with measured angle
    for c in cands:
        pred = simulate(c["expr"], th[va[0]], v[va[0]], t[va])
        c["val_err"] = angle_error(pred, theta[va])
        c["ops"] = int(sp.count_ops(c["expr"]))
    chosen = {}
    for m in ("SINDy", "known form", "known form + drag"):     # each method on its own
        ok = [c for c in cands if c["method"] == m and np.isfinite(c["val_err"])]
        if not ok:
            continue
        best = min(c["val_err"] for c in ok)
        near = [c for c in ok if c["val_err"] <= best + max(0.005, 0.1 * best)]
        chosen[m] = min(near, key=lambda c: (c["ops"], c["val_err"], str(c["expr"])))

    # Test: other releases, simulated from their own (smoothed) first state
    tests = {}
    for exp in setup["test"]:
        tt, thm = get(exp)
        kk, ths, vs, _ = smooth(tt, thm)
        i0 = kk.start
        tests[exp] = (tt[i0:], thm[i0:], ths[i0], vs[i0],
                      angle_error(ths[i0:], thm[i0:]))     # smoothing floor

    L = None
    if source == "synthetic":
        source = f"synthetic_noise{TRUTH[mass][2] if noise is None else noise:g}"
    rec = {"mass": mass, "source": source, "sindy_variant": variant,
           "sindy_settings": {"alpha": VARIANTS[variant]["alpha"] if VARIANTS[variant]["alpha"] is not None
                              else "0.05*rows", "thresholds": VARIANTS[variant]["thresholds"],
                              "max_iter": CORE["max_iter"], "unbias": CORE["unbias"]},
           "train": setup["train"], "test": setup["test"],
           "rows_train": int(len(tr)), "rows_validation": int(len(va)),
           "all_candidates": [{k_: (str(v_) if k_ == "expr" else v_) for k_, v_ in c.items()}
                              for c in cands], "chosen": {}}
    print(f"\n=== {source} data, SINDy settings: {variant} === {mass} bob: train {setup['train']} ({len(tr)} rows), "
          f"validate last {VAL_FRACTION:.0%} ({len(va)} rows), test {', '.join(setup['test'])}")
    for c in cands:
        print(f"  {c['method']:18s} {c['setting']:16s} val err {c['val_err']:.3f}  {c['expr']}")
    fig, axes = plt.subplots(len(tests), 1, figsize=(9, 2.6 * len(tests)), squeeze=False)
    colors = {"SINDy": "#2a78d6", "known form": "#1baf7a", "known form + drag": "#6250d6"}
    for m, c in chosen.items():
        errs = {}
        for (exp, (tt, thm, th0, v0, floor)), ax in zip(tests.items(), axes[:, 0]):
            pred = simulate(c["expr"], th0, v0, tt)
            errs[exp] = angle_error(pred, thm)
            if pred is not None:
                ax.plot(tt, np.degrees(pred), color=colors[m], lw=1.2, label=m)
        rec["chosen"][m] = {"expr": str(c["expr"]), "setting": c["setting"],
                            "val_err": c["val_err"], "test_err": errs}
        print(f"  CHOSEN {m:18s} test errors {', '.join(f'{e}: {v_:.3f}' for e, v_ in errs.items())}"
              f"  ->  {c['expr']}")
    for (exp, (tt, thm, *_ , floor)), ax in zip(tests.items(), axes[:, 0]):
        ax.plot(tt, np.degrees(thm), ".", ms=2, color="#8a8985", label="measured")
        ax.set_title(f"{source}, {mass} bob, {exp} (never seen in fitting); "
                     f"SINDy {variant}; smoothing floor {floor:.3f}",
                     loc="left", fontsize=9)
        ax.set_ylabel("angle (deg)"); ax.spines[["top", "right"]].set_visible(False)
        rec.setdefault("smoothing_floor", {})[exp] = floor
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left", ncol=4)
    axes[-1, 0].set_xlabel("time since release start (s)")
    tag = f"{mass}_{source}_{variant}"
    fig.tight_layout(); fig.savefig(OUT / f"{tag}.png", dpi=130); plt.close(fig)
    (OUT / f"{tag}.json").write_text(json.dumps(rec, indent=2, default=float))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mass", choices=list(SETUPS), nargs="+", default=list(SETUPS))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--noise", type=float, default=None,
                    help="synthetic angle noise in rad (default: the level measured in the video)")
    ap.add_argument("--sindy-variant", choices=list(VARIANTS), default="core")
    a = ap.parse_args()
    for m in a.mass:
        run(m, "synthetic" if a.synthetic else "real", a.sindy_variant, a.noise)
    print(f"\nwrote results to {OUT}/")


if __name__ == "__main__":
    main()
