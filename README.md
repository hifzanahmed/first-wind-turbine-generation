# Wind-turbine DDQN prototype

This repository is an **offline research prototype**, not a turbine controller.
It currently uses a CSV-driven mock environment and a simplified reward/dynamics
proxy. Do not connect it to an operating turbine or treat its outputs as safe
actuator commands.

## Input data

Pass a time-ordered CSV with the following required columns:

| Column | Expected unit |
| --- | --- |
| `wind_speed` | m/s |
| `yaw_error` | degrees |
| `rotor_speed` | Same unit as the configured rated rotor speed (2.0 in the mock) |
| `blade_pitch` | degrees |
| `rotor_yaw` | degrees |

The loader rejects missing columns, non-numeric or non-finite values, negative
wind/rotor speeds, and blade pitch outside 0–90 degrees. It preserves row order;
sort and clean measured SCADA records before training. The bundled `data/Input.csv`
is a small example input, not a validated field dataset.

The example assumes wind speed in m/s and angular fields in degrees. Convert
measured rotor speed to the unit expected by the configured rated reference before
training; the example CSV does not record measurement units.

The environment selects discrete yaw, pitch, and generator-torque increments.
Pitch and torque influence rotor speed through a deliberately simple proxy; this
is not a calibrated turbine model. The reward is a power proxy with an overspeed
penalty, not measured electrical power.

## Run

```powershell
python main.py --csv data/Input.csv --episodes 300 `
  --validation-fraction 0.2 --checkpoint artifacts/ddqn.pt `
  --metrics artifacts/training_metrics.json
```

The latest contiguous fraction of the CSV is held out for greedy validation to
avoid mixing future rows into training. Use `--validation-fraction 0` to disable
the holdout. Training runs the requested episode count by default (300 here);
early stopping is opt-in with `--patience 30`. Evaluate a saved checkpoint with:

```powershell
python main.py --csv data/Input.csv --eval-only --load artifacts/ddqn.pt
```

Prioritized replay uses proportional absolute-TD-error priorities and
importance-sampling weights. Checkpoints retain model dimensions and optimizer
state; legacy weights-only checkpoints can also be loaded when dimensions match.

For real-turbine research, validate units and sensor semantics, calibrate the
dynamics/reward against a trusted simulator or measured outcomes, and complete
independent safety review before considering any hardware integration.
