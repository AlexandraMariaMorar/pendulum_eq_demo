# Real pendulum video (after-the-core extra, Person B)

Not part of the benchmark. The plan keeps the four-day core to simulated data
("Postpone real pendulum video, noisy derivative…"), so nothing in
`pendulum_lab/` reads these files, and **results here are not set side by side
with the benchmark's.** The benchmark gives SINDy exact acceleration labels;
here they come from differentiating a video. Data, label source and pipeline
all differ, so a gap between the two would not say which one caused it.

| Path | What it is |
|---|---|
| `physical_data/` | Six cleaned Tracker recordings, one CSV per release (90 g ×4, 136 g ×2). See its `README.txt`. |
| `fit_physical.py` | Fits SINDy, the known form and known form + drag; tests on releases never used in fitting. |
| `test_fit_physical.py` | Checks: core code and settings are used; a noise-free simulated recording is recovered. |
| `wobble_check.py` | Does the bob leave its swing plane? Uses the apparent string length. |
| `results/` | One JSON record and one figure per run of `fit_physical.py`. |

## Run (from the repository root)

```
python -m pytest -q extras/real_video/test_fit_physical.py
python extras/real_video/fit_physical.py                          # real recordings
python extras/real_video/fit_physical.py --synthetic              # control: simulated, measured noise
python extras/real_video/fit_physical.py --synthetic --noise 0    # control: simulated, no noise
python extras/real_video/fit_physical.py --sindy-variant ridge    # sensitivity check only
```

## Design

- **Train** on the longest release of a bob, **validate** on its last 25 % in time,
  **test** on the bob's other releases (new starting states).
- Velocity and acceleration come from a Savitzky–Golay filter (order 5, 0.3 s window),
  because the camera only gives angles. Fixed before fitting; not tuned.
- **SINDy uses the core's own tested code and settings**: `pendulum_lab.features`
  (the 14 terms) and `sindy_adapter.scale_fit` (pysindy STLSQ with the manual RMS
  scaling), with `alpha`, `thresholds`, `max_iter` and `unbias` read from
  `configs/experiment.yaml`. No real-data tuning in the headline result.
- Each method picks its own candidate on validation only: simplest within
  max(0.005, 10 %) of its best simulated-swing angle error.
- Score: angle error of a simulated swing on the test release, divided by the RMS
  of the measured angle (0 = perfect, about 1 = no better than zero).
- **Synthetic control** (`--synthetic`): every recording is replaced by a simulated
  one with the same frame times and start state, the equation from the known-form
  fit, and Gaussian angle noise at the level measured in the video (0.0022 rad for
  90 g, 0.0009 rad for 136 g). Same pipeline, known answer.

## Results (17 Nov 2026)

### 1. Real recordings, core SINDy settings

| Bob | Test release | SINDy | Known form | Known form + drag |
|---|---|---|---|---|
| 90 g | exp3 | 1.03 | 0.20 | **0.15** |
| 90 g | exp4 | 1.50 | 0.65 | **0.09** |
| 136 g | exp2 | blew up | **0.04** | 0.41 |

The known form gives θ'' = −29.83 sin θ − 0.021 θ' (90 g) and −24.31 sin θ − 0.024 θ'
(136 g), implying lengths of 0.329 m and 0.404 m against 0.32 m and 0.41 m measured
in the video.

### 2. Control: the same pipeline on simulated recordings

| Angle noise (rad) | 90 g SINDy (exp3 / exp4) | 90 g known form | 136 g SINDy | 136 g known form |
|---|---|---|---|---|
| 0 | 0.002 / 0.004 | 0.002 / 0.003 | 0.003 | 0.002 |
| 0.0002 | 0.07 / 1.56 | 0.008 / 0.006 | 0.39 | 0.003 |
| 0.0005 | 0.21 / blew up | 0.016 / 0.011 | 1.06 | 0.010 |
| 0.001 | 1.59 / blew up | 0.031 / 0.018 | blew up | 0.021 |
| measured (0.0022 / 0.0009) | blew up / blew up | 0.065 / 0.036 | blew up | 0.019 |

- With no noise the whole pipeline works, so the code and selection are not the cause.
- SINDy already breaks at a tenth of the measured noise, on simulated data where the
  true equation is exactly in its dictionary. So the real-data failure does **not**
  need anything specific to the real pendulum (friction, wobble, camera) to explain it.
- Why: over one 15 s release with 40–65° swings, θ, sin θ, θ³ and sin 2θ are nearly the
  same curve (condition number of the scaled dictionary about 2·10⁶). Small derivative
  errors then move weight between them in large, cancelling amounts. The fits match the
  training stretch but extrapolate badly to a new release.

### 3. Sensitivity check, labelled: ridge penalty

`--sindy-variant ridge` adds a ridge penalty (0.05 × rows) and a threshold of 0.3.
**These were chosen after seeing the first results on these recordings**, so they are
not a fair headline. They do not change the conclusion: SINDy test errors 0.91 /
blew up (90 g) and 1.94 (136 g).

## What this experiment does and does not show

- It shows that with differentiated video angles from one short release per bob, SINDy
  with the benchmark's 14-term dictionary and settings does not give a usable
  equation, while the two-term known form does.
- It does not compare SINDy with AI Feynman: **AI Feynman was not run on these
  recordings.** If it is, it should use its config settings without real-data tuning,
  the same releases for training, validation and testing, and the same score.
- It does not show that SINDy "fails on real data" in general: longer recordings, wider
  swings, several training releases or a smaller dictionary may change the result.

## Limitations

- Frame rate inferred (24 fps for 90 g, 25 fps for 136 g), not yet checked against the video files.
- `wobble_check.py` flags possible out-of-plane motion in 5 of 6 releases, just above its
  detection limit; some of that is tracking slips at the turning points.
- Releases are 2–15 s long; one training release per bob; damping is too small to compare the masses.
