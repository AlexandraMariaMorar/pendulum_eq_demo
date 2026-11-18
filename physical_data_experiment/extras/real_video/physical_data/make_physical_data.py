"""Clean the two Tracker exports into one file per experiment (release).

    python make_physical_data.py data_dirty_1.txt data_dirty_2.1.txt --out physical_data

What it does, for each recording:
  1. Reads t, x, y. File 2 has two empty columns first; the real ones are
     t_{1}, x, y_{1}, so columns are found by name, not position.
  2. Rebuilds the time axis from the frame count: t = frame / TRUE_FPS.
     Tracker's own times assumed the wrong frame rate (60 and 30 fps); the
     swing periods only match g = 9.81 m/s^2 at about 24-25 fps.
     CHANGE TRUE_FPS BELOW once you have checked the video's real frame rate.
  3. Removes frames Tracker filled in itself. Those lie on perfectly smooth
     lines: the second difference of x and y is below 1e-4 m for every frame.
     Real tracked frames never are (pixel noise alone is bigger).
  4. Splits at those fills and at gaps in time. Each piece is one release of
     the pendulum, saved with time restarting at 0.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Recording -> (mass, labelled fps in Tracker, assumed true fps)
RECORDINGS = {
    "data_dirty_1": ("mass090g", 60.0, 24.0),
    "data_dirty_2": ("mass136g", 30.0, 25.0),
}
FILL_TOL = 1e-4        # metres; second difference below this = filled in
MIN_FRAMES = 30        # shorter pieces are dropped


def read_export(path):
    lines = Path(path).read_text(errors="ignore").splitlines()
    head = next(i for i, l in enumerate(lines) if l.split("\t")[0].strip() in ("t", "")
                and "x" in [c.strip() for c in l.split("\t")])
    names = [c.strip() for c in lines[head].split("\t")]
    ti = names.index("t_{1}") if "t_{1}" in names else names.index("t")
    xi = names.index("x")
    yi = names.index("y_{1}") if "y_{1}" in names else names.index("y")
    rows = []
    for l in lines[head + 1:]:
        c = l.split("\t")
        try:
            rows.append((float(c[ti]), float(c[xi]), float(c[yi])))
        except (ValueError, IndexError):
            continue
    d = np.array(rows)
    return d[:, 0], d[:, 1], d[:, 2]


def pivot(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    (D, E, _), *_ = np.linalg.lstsq(A, -(x**2 + y**2), rcond=None)
    return -D / 2, -E / 2


def pieces(t, x, y, labelled_fps):
    dt = 1 / labelled_fps
    d2 = np.abs(np.diff(x, 2)) + np.abs(np.diff(y, 2))
    filled = np.r_[False, d2 < FILL_TOL, False]
    jump = np.r_[True, np.diff(t) > 1.5 * dt]
    out, cur = [], []
    for i in range(len(t)):
        if filled[i]:
            if len(cur) >= MIN_FRAMES: out.append(cur)
            cur = []
            continue
        if jump[i] and cur:
            if len(cur) >= MIN_FRAMES: out.append(cur)
            cur = []
        cur.append(i)
    if len(cur) >= MIN_FRAMES: out.append(cur)
    return [np.array(p) for p in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="physical_data")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    for f in a.files:
        key = next(k for k in RECORDINGS if Path(f).stem.startswith(k))
        mass, lab_fps, fps = RECORDINGS[key]
        t, x, y = read_export(f)
        frame = np.rint((t - t[0]) * lab_fps).astype(int)     # frame number in the video
        xc, yc = pivot(x, y)
        for n, p in enumerate(pieces(t, x, y, lab_fps), 1):
            fr = frame[p]
            tt = (fr - fr[0]) / fps
            th = np.arctan2(x[p] - xc, -(y[p] - yc))
            name = out / f"{mass}_exp{n}.csv"
            with open(name, "w", newline="\n") as fh:
                fh.write(f"# {mass[4:]} bob, release {n} of recording {Path(f).name}\n")
                fh.write(f"# time rebuilt from frame number at {fps:g} fps (Tracker said {lab_fps:g}); "
                         f"x, y in metres; pivot fitted at ({xc:.4f}, {yc:.4f}) m\n")
                fh.write("# theta measured from straight down, positive to the right\n")
                fh.write("t_s,x_m,y_m,theta_rad,frame,t_tracker_s\n")
                for i, k in enumerate(p):
                    fh.write(f"{tt[i]:.6f},{x[k]:.6f},{y[k]:.6f},{th[i]:.6f},{fr[i]},{t[k]:.6f}\n")
            print(f"{name.name}: {len(p)} frames, {tt[-1]:.2f} s")


if __name__ == "__main__":
    main()
