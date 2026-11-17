# Real pendulum video (after-the-core extra, Person B)

Not part of the benchmark. The plan keeps the four-day core to simulated data
("Postpone real pendulum video, noisy derivative…"), so nothing in
`pendulum_lab/` reads these files.

| Path | What it is |
|---|---|
| `physical_data/` | Six cleaned Tracker recordings, one CSV per release (90 g ×4, 136 g ×2). See its `README.txt` for the frame-rate correction and what was removed. |
| `wobble_check.py` | Does the bob leave its swing plane? Uses the apparent string length. |
| `fit_physical.py` | Fits SINDy, the known form and known form + drag; tests on releases never used in fitting. |
| `results/` | Output of `fit_physical.py`: one JSON record and one figure per mass. |

## Run

```
python extras/real_video/fit_physical.py
python extras/real_video/wobble_check.py extras/real_video/physical_data/mass136g_exp1.csv
```
Needs only numpy, scipy, sympy and matplotlib (all in `requirements-core.txt`).

## Design

- **Train** on the longest release of a bob, **validate** on its last 25 % in time,
  **test** on the bob's other releases (new starting states, like `test.npz`).
- Velocity and acceleration come from a Savitzky–Golay filter (order 5, 0.3 s window),
  because the camera only gives angles.
- SINDy uses the core's 14 terms and manual RMS scaling, plus a small ridge penalty
  (`ALPHA = 0.05`) and one larger threshold (0.3). The core uses neither; both are
  real-data settings, recorded here.
- Each method picks its own candidate: simplest within max(0.005, 10 %) of its best
  validation angle error. The score is the angle error of a simulated swing, divided
  by the RMS of the measured angle (0 = perfect, about 1 = no better than zero).

## First results (15 Nov 2026)

Test angle error on releases never used in fitting:

| Bob | Test release | SINDy | Known form | Known form + drag |
|---|---|---|---|---|
| 90 g | exp3 | 0.91 | 0.20 | **0.15** |
| 90 g | exp4 | failed (blew up) | 0.65 | **0.09** |
| 136 g | exp2 | 1.94 | **0.04** | 0.41 |

- **The known form recovers g/L.** 90 g: θ'' = −29.83 sin θ − 0.021 θ', giving
  L = 9.81/29.83 = 0.329 m against 0.32 m measured from the video.
  136 g: θ'' = −24.31 sin θ − 0.024 θ', giving L = 0.404 m against 0.41 m. Both within
  about 3 %, which also supports the 24 / 25 fps time correction.
- **SINDy does not find a usable equation here.** Over swings of 40–65°, θ, sin θ,
  θ³ and sin 2θ are almost the same curve, and the derivative noise is enough for
  SINDy to spread the restoring force over several of them with large, cancelling
  coefficients. These fit the training stretch but drift or blow up on new releases.
  This is the "noisy derivatives" problem the core avoids by supplying exact labels.
- **Damping is barely measurable.** The linear damping is about 0.02 s⁻¹. Adding
  air drag helps on the 90 g releases and hurts on the 136 g one, so neither damping
  model is clearly right, and the two masses cannot be compared on damping yet.

## Limitations

- Frame rate inferred (24 fps for 90 g, 25 fps for 136 g), not yet checked against the video file.
- `wobble_check.py` flags possible out-of-plane motion in 5 of 6 releases, just above its
  detection limit; some of that is tracking slips at the turning points.
- Releases are 2–15 s long; one training release per bob.
