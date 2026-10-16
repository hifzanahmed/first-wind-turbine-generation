# Wind-turbine DDQN prototype

This repository is an **offline research prototype inspired by the cited paper**,
not a faithful reproduction or a turbine controller. Its BEM implementation,
turbine geometry, airfoil polar, controller dynamics, and wind file are generic
illustrative assumptions. Do not connect it to an operating turbine or treat its
outputs as safe actuator commands.

Reference: Soler et al., “Reinforcement learning to maximize wind turbine energy
generation,” *Expert Systems with Applications*, 249 (2024), 123502,
[doi:10.1016/j.eswa.2024.123502](https://doi.org/10.1016/j.eswa.2024.123502).

## Two input files

`data/Input.csv` retains the original CSV mock environment and requires:

| Column | Expected unit |
| --- | --- |
| `wind_speed` | m/s |
| `yaw_error` | degrees |
| `rotor_speed` | Same unit as the configured mock rated rotor speed |
| `blade_pitch` | degrees |
| `rotor_yaw` | degrees |

`data/TurbulentWind.csv` is the second input for the BEM environment:

| Column | Expected unit |
| --- | --- |
| `wind_speed` | m/s |
| `wind_direction_error` | degrees (wind direction relative to the initial nacelle heading) |

Both included files are **synthetic examples**, not measured/validated turbine
data. Rows are consumed in time order. Replace the BEM model's default
`TurbineSpec` geometry, airfoil polar, drivetrain, and operating limits with
verified turbine-specific values before interpreting results.
The wind file assumes one-second row spacing; the BEM environment has no timestamp
column, so change `TurbineSpec.time_step_s` if the profile cadence differs.

The `bem` environment (default) uses a simplified blade-element/momentum
calculation, turbulent wind input, and discrete rotor-speed/yaw/pitch commands.
`csv-mock` preserves the original yaw/pitch/torque mock for comparison. The BEM
airfoil polar is analytic and the dynamic rotor/generator response is approximate;
this implementation does not reproduce the paper's calibrated turbine model.

## Run

```powershell
python main.py --environment bem --wind-csv data/TurbulentWind.csv --episodes 300 `
  --validation-fraction 0.2 --checkpoint artifacts/ddqn.pt `
  --metrics artifacts/training_metrics.json
```

The latest contiguous wind segment is held out for validation. Training runs the
requested episode count by default; early stopping is opt-in with `--patience 30`.
Compare with PID and empirical tabular value iteration on the held-out wind with:

```powershell
python main.py --environment bem --compare-baselines --vi-sample-steps 2000
```

The printed MWh figure is an annualized **mean-power extrapolation from the
short synthetic profile**, not a validated annual energy production estimate.
Evaluate a saved checkpoint with:

```powershell
python main.py --environment bem --wind-csv data/TurbulentWind.csv `
  --eval-only --load artifacts/ddqn.pt
```

Prioritized replay uses proportional absolute-TD-error priorities and
importance-sampling weights. Checkpoints retain model dimensions and optimizer
state; legacy weights-only checkpoints can also be loaded when dimensions match.

Use `--environment csv-mock --csv data/Input.csv` to run the original mock. For
research-grade reproduction, enter the exact paper turbine geometry, airfoil
polars, state/action/reward definitions, wind fields, and controller parameters,
then validate against the authors' setup and measured/benchmark results. The
included baselines and annual-energy estimate are scaffolding, not paper-result
replications or safety certification.
