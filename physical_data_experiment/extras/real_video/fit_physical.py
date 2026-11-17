"""Fit the real pendulum recordings (after-the-core extra, Person B).

    python extras/real_video/fit_physical.py                 # both masses
    python extras/real_video/fit_physical.py --mass 136g     # one mass

Design, mirroring the simulated benchmark as closely as real data allows:
  * TRAIN on one long release, TEST on the other releases of the same bob.
    A different release is a genuinely new starting state, like test.npz.
  * VALIDATE on the last 25% (in time) of the training release. Never random
    rows: neighbouring frames are near copies, so a random split would leak.
  * Labels: the camera gives only angles, so velocity and acceleration come
    from a Savitzky-Golay filter (a polynomial fitted in a sliding window).
    This is the "noisy derivatives" problem the core deliberately avoids.
  * Methods: SINDy with the core's 14 terms (sequential thresholding, the same
    manual RMS scaling, plus a small ridge penalty: see ALPHA), the known form a*sin(theta) + b*v, and the known form
    plus air drag v*|v|. Self-contained: needs numpy, scipy, sympy, matplotlib.
  * Selection, per method: simulate each candidate across the validation
    stretch; keep the simplest within max(0.005, 10% of that method's best). Frozen before
    looking at the test releases.
Writes extras/real_video/results/<mass>_results.json and <mass>_fit.png.
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

HERE = Path(__file__).parent
DATA = HERE / "physical_data"
OUT = HERE / "results"
G = 9.81
SETUPS = {"090g": {"train": "exp2", "test": ["exp3", "exp4"]},     # exp1 is one swing only
          "136g": {"train": "exp1", "test": ["exp2"]}}
WINDOW_S, ORDER = 0.3, 5           # smoothing window and polynomial order
VAL_FRACTION = 0.25
THRESHOLDS = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]   # core grid + 0.3 for noisy labels
# Ridge penalty on the scaled problem. The core uses 0 (exact labels). Here the
# labels come from differentiating video, and theta, sin(theta), theta**3 and
# sin(2*theta) are almost identical over a 40-60 degree swing; with no penalty
# least squares "explains" the noise with huge, cancelling coefficients.
ALPHA = 0.05

TH, V = sp.symbols("theta v")
TERMS = ["1", "theta", "v", "theta**2", "theta*v", "v**2", "theta**3", "theta**2*v",
         "theta*v**2", "v**3", "sin(theta)", "cos(theta)", "sin(2*theta)", "cos(2*theta)"]
SYM = [sp.sympify(n, locals={"theta": TH, "v": V}) for n in TERMS]
FUN = [sp.lambdify((TH, V), s, "numpy") for s in SYM]


def library(th, v, syms=SYM, funs=FUN):
    return np.column_stack([np.broadcast_to(f(th, v), th.shape) for f in funs]).astype(float)


# ---------- data ----------
def load(mass, exp):
    d = np.loadtxt(DATA / f"mass{mass}_{exp}.csv", delimiter=",", comments="#", skiprows=4)
    return d[:, 0], np.unwrap(d[:, 3])                  # t_s, theta_rad


def smooth(t, theta):
    dt = float(np.median(np.diff(t)))
    w = max(int(round(WINDOW_S / dt)) | 1, ORDER + 2)
    th = savgol_filter(theta, w, ORDER, deriv=0, delta=dt)
    v = savgol_filter(theta, w, ORDER, deriv=1, delta=dt)
    a = savgol_filter(theta, w, ORDER, deriv=2, delta=dt)
    k = slice(w // 2, len(t) - w // 2)                  # filter edges are unreliable
    return k, th, v, a


# ---------- fitting ----------
def ridge(Z, y, alpha):
    if alpha == 0:
        return np.linalg.lstsq(Z, y, rcond=None)[0]
    n = Z.shape[1]
    return np.linalg.solve(Z.T @ Z / len(y) + alpha * np.eye(n), Z.T @ y / len(y))


def stlsq(Phi, y, thr, alpha=ALPHA, max_iter=20):
    """Sequential thresholded ridge regression on RMS-scaled columns.

    Same scaling and thresholding as the core's STLSQ; the final pass refits
    the surviving terms without the penalty ("unbias", as in the core).
    """
    q = np.sqrt(np.mean(Phi**2, axis=0)); q[q == 0] = 1
    s = float(np.std(y))
    Z, yz = Phi / q, y / s
    keep = np.ones(Phi.shape[1], bool)
    w = np.zeros(Phi.shape[1])
    for _ in range(max_iter):
        w[:] = 0
        w[keep] = ridge(Z[:, keep], yz, alpha)
        new = keep & (np.abs(w) >= thr)
        if (new == keep).all():
            break
        keep = new
        if not keep.any():
            break
    w[:] = 0
    if keep.any():
        w[keep] = np.linalg.lstsq(Z[:, keep], yz, rcond=None)[0]   # unbias
    return s * w / q


def to_expr(c, syms):
    return sum((sp.Float(float(ci), 6) * s for ci, s in zip(c, syms) if ci != 0), sp.Integer(0))


def candidates(th, v, a):
    out = []
    Phi = library(th, v)
    for thr in THRESHOLDS:
        out.append({"method": "SINDy", "setting": f"threshold {thr}",
                    "expr": to_expr(stlsq(Phi, a, thr), SYM)})
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


def run(mass):
    OUT.mkdir(exist_ok=True)
    setup = SETUPS[mass]
    t, theta = load(mass, setup["train"])
    k, th, v, a = smooth(t, theta)
    idx = np.arange(len(t))[k]
    cut = int((1 - VAL_FRACTION) * len(t))
    tr, va = idx[idx < cut], idx[idx >= cut]
    cands = candidates(th[tr], v[tr], a[tr])

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
        tt, thm = load(mass, exp)
        kk, ths, vs, _ = smooth(tt, thm)
        i0 = kk.start
        tests[exp] = (tt[i0:], thm[i0:], ths[i0], vs[i0],
                      angle_error(ths[i0:], thm[i0:]))     # smoothing floor

    L = None
    rec = {"mass": mass, "train": setup["train"], "test": setup["test"],
           "rows_train": int(len(tr)), "rows_validation": int(len(va)),
           "all_candidates": [{k_: (str(v_) if k_ == "expr" else v_) for k_, v_ in c.items()}
                              for c in cands], "chosen": {}}
    print(f"\n=== {mass} bob: train {setup['train']} ({len(tr)} rows), "
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
        ax.set_title(f"{mass} bob, {exp} (never seen in fitting); smoothing floor {floor:.3f}",
                     loc="left", fontsize=9)
        ax.set_ylabel("angle (deg)"); ax.spines[["top", "right"]].set_visible(False)
        rec.setdefault("smoothing_floor", {})[exp] = floor
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left", ncol=4)
    axes[-1, 0].set_xlabel("time since release start (s)")
    fig.tight_layout(); fig.savefig(OUT / f"{mass}_fit.png", dpi=130)
    (OUT / f"{mass}_results.json").write_text(json.dumps(rec, indent=2, default=float))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mass", choices=list(SETUPS), nargs="+", default=list(SETUPS))
    for m in ap.parse_args().mass:
        run(m)
    print(f"\nwrote {OUT}/<mass>_results.json and <mass>_fit.png")


if __name__ == "__main__":
    main()
