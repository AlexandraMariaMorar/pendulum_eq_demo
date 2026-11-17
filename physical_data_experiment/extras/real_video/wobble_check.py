"""How much does the real pendulum leave its plane? (after-the-core extra)

The camera sees a 2-D projection. If the bob swings partly toward or away
from the camera by a distance z, its projected distance from the pivot is
    r = sqrt(L**2 - z**2)  <=  L.
So out-of-plane wobble shows up as the apparent string length getting
SHORTER, and only shorter. Tracking noise, by contrast, makes r jitter both
ways. That one-sidedness is what this check uses.

    python extras/real_video/wobble_check.py my_tracker_export.txt [--pivot X Y]

Prints z/L percentiles and a verdict, and saves wobble_check.png.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402


def load_tracker(path):
    rows = []
    for line in Path(path).read_text(errors="ignore").splitlines():
        cells = [c.strip() for c in re.split(r"[\t,;]", line)]
        if len(cells) < 3:
            cells = line.split()
        try:
            rows.append(tuple(float(c) for c in cells[:3]))
        except ValueError:
            continue
    d = np.array([r for r in rows if np.isfinite(r).all()])
    return d[:, 0], d[:, 1], d[:, 2]


def fit_circle(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    (D, E, F), *_ = np.linalg.lstsq(A, -(x**2 + y**2), rcond=None)
    return -D / 2, -E / 2


def wobble(x, y, t, pivot=None, window_s=0.25):
    """Returns r/L (smoothed), implied |z|/L, L, and the z/L detection limit.

    Wobble changes slowly (about once per swing); tracking noise changes every
    frame. A running median over ~0.25 s removes most of the noise first,
    which matters because z = L*sqrt(1 - (r/L)**2) blows small errors up:
    a 0.5% shortening already means z = 10% of L.
    """
    xc, yc = fit_circle(x, y) if pivot is None else pivot
    r = np.hypot(x - xc, y - yc)
    dt = np.median(np.diff(t))
    w = max(3, int(round(window_s / dt)) | 1)
    pad = np.pad(r, w // 2, mode="edge")
    r_s = np.array([np.median(pad[i:i + w]) for i in range(len(r))])
    sigma = np.std(r - r_s) / np.sqrt(w)          # noise left after smoothing
    # True length: the longest projections are when the bob is back in plane.
    L = np.percentile(r_s, 98)
    ratio = r_s / L
    z_over_L = np.sqrt(np.clip(1 - np.minimum(ratio, 1) ** 2, 0, None))
    # Shortenings smaller than ~3 sigma, plus the spread caused by a slightly
    # wrong pivot or L, are invisible. The 1.5 safety factor comes from fake
    # planar data, where noise and pivot error alone reached about 1.1x this.
    z_limit = 1.5 * float(np.sqrt(2 * 3 * sigma / L + 2 * (np.percentile(r_s, 98)
                                                         - np.percentile(r_s, 90)) / L))
    return ratio, z_over_L, L, z_limit, (xc, yc)


def verdict(z95, z_limit):
    if z95 <= z_limit:
        return (f"no wobble detectable above the noise (anything below about "
                f"{z_limit:.2f} L is invisible to this check)")
    if z95 < 0.05:
        return "negligible: treat the motion as planar"
    if z95 < 0.15:
        return "noticeable: usable, but say so, and prefer the most planar stretch"
    return "serious: the motion is not planar; re-film or trim to a planar stretch"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--pivot", nargs=2, type=float)
    a = ap.parse_args()
    t, x, y = load_tracker(a.file)
    ratio, z, L, z_limit, piv = wobble(x, y, t, a.pivot)
    z50, z95 = np.percentile(z, [50, 95])
    print(f"pivot = ({piv[0]:.4f}, {piv[1]:.4f}), length L ≈ {L:.4f}")
    print(f"apparent length r/L (smoothed): 5th pct {np.percentile(ratio, 5):.4f}, "
          f"median {np.median(ratio):.4f}")
    print(f"out-of-plane |z|/L: median {z50:.3f}, 95th pct {z95:.3f}, "
          f"detection limit {z_limit:.3f}")
    print("verdict:", verdict(z95, z_limit))

    fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax[0].plot(t, ratio, ".", ms=2, color="#2a78d6")
    ax[0].axhline(1, color="#52514e", lw=0.8)
    ax[0].set_ylabel("apparent length r / L")
    ax[1].plot(t, z, ".", ms=2, color="#eb6834")
    ax[1].set_ylabel("implied |z| / L"); ax[1].set_xlabel("time (s)")
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig("wobble_check.png", dpi=150)
    print("saved wobble_check.png")


if __name__ == "__main__":
    main()
