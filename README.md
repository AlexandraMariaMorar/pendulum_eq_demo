# pendulum-discovery-demo

A reproducible comparison of **AI Feynman** and **SINDy** at recovering damped-pendulum dynamics from simulated data, with an interactive viewer showing the recovered equations and the predicted motion.

Reference dynamics (dimensionless time, angles in radians):

```
theta_dot = v
v_dot     = -4 * sin(theta) - 0.2 * v
```

Both methods learn only the acceleration `v_dot` from `(theta, v)`. The experiment varies acceleration-label noise and training sample count. It is a controlled rediscovery demonstration. It does not discover new physics or rank symbolic-regression methods in general.

## Status

Work in progress . Setup, pipeline and viewer instructions will be added here as the components are merged.

## Ownership

| Area | Owner |
|---|---|
| `configs/`, `contracts.py`, `expressions.py`, `evaluate.py`, `run.py`, AI Feynman adapter | Alex |
| `data.py`, `features.py`, SINDy adapter, `report.py`, `app/viewer.py` | Stephan |

Work on short task branches (`a/...`, `b/...`) from an up-to-date `main` and merge through reviewed pull requests. Never force-push `main`.
