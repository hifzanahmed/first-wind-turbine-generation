import math
import unittest
from pathlib import Path

from turbine_rl.bem import (
    BEMWindEnvironment,
    BladeElementMomentumModel,
    TurbineSpec,
    annual_energy_mwh,
    generate_turbulent_wind,
)
from turbine_rl.benchmarks import (
    PIDController,
    ValueIterationController,
    evaluate_policy,
)


ROOT = Path(__file__).resolve().parents[1]


class BEMEnvironmentTests(unittest.TestCase):
    def test_bem_solver_returns_finite_positive_loads(self) -> None:
        # Verify illustrative geometry produces finite aerodynamic loads.
        result = BladeElementMomentumModel(TurbineSpec()).calculate(
            wind_speed_m_s=8.0,
            rotor_speed_rad_s=1.2,
            pitch_deg=2.0,
            yaw_error_deg=0.0,
        )

        self.assertTrue(math.isfinite(result.power_w))
        self.assertTrue(math.isfinite(result.torque_nm))
        self.assertTrue(math.isfinite(result.thrust_n))
        self.assertGreater(result.power_w, 0.0)
        self.assertGreater(result.torque_nm, 0.0)

    def test_synthetic_turbulence_is_reproducible(self) -> None:
        # Confirm the synthetic turbulent wind generator honors its seed.
        first = generate_turbulent_wind(20, seed=42)
        second = generate_turbulent_wind(20, seed=42)

        self.assertEqual(first, second)
        self.assertGreater(len({item.speed_m_s for item in first}), 1)

    def test_wind_csv_drives_controls_and_energy_accounting(self) -> None:
        # Load the second sample input, apply a paper-like actuator action, and measure energy.
        wind_path = ROOT / "data" / "TurbulentWind.csv"
        environment = BEMWindEnvironment(wind_csv_path=wind_path, end_index=8)
        state = environment.reset()
        action = environment.ACTIONS.index(
            next(
                item
                for item in environment.ACTIONS
                if item.rotor_speed_delta_rad_s == 0.1
                and item.yaw_delta_deg == 5.0
                and item.pitch_delta_deg == 1.0
            )
        )
        next_state, reward, done = environment.step(action)

        self.assertEqual(len(state), environment.observation_size)
        self.assertEqual(len(next_state), environment.observation_size)
        self.assertEqual(environment.action_size, 27)
        self.assertFalse(done)
        self.assertTrue(math.isfinite(reward))
        self.assertGreaterEqual(environment.episode_energy_j, 0.0)

        for _ in range(7):
            _, _, done = environment.step(action)
        self.assertTrue(done)
        self.assertGreater(environment.episode_energy_j, 0.0)
        self.assertGreater(
            annual_energy_mwh(
                environment.episode_energy_j, environment.duration_seconds
            ),
            0.0,
        )

    def test_pid_and_value_iteration_policies_run_on_same_profile(self) -> None:
        # Smoke-test both baselines and the empirical tabular planner on fixed wind.
        path = ROOT / "data" / "TurbulentWind.csv"
        training_environment = BEMWindEnvironment(wind_csv_path=path, end_index=8)
        evaluation_environment = BEMWindEnvironment(wind_csv_path=path, start_index=8, end_index=16)
        pid_result = evaluate_policy(
            evaluation_environment, PIDController(evaluation_environment)
        )
        value_iteration = ValueIterationController(training_environment)
        value_iteration.fit(sample_steps=24, iterations=5, seed=4)
        vi_result = evaluate_policy(evaluation_environment, value_iteration)

        self.assertTrue(math.isfinite(pid_result.reward))
        self.assertTrue(math.isfinite(vi_result.reward))
        self.assertGreaterEqual(pid_result.annual_energy_mwh, 0.0)
        self.assertGreaterEqual(vi_result.annual_energy_mwh, 0.0)


if __name__ == "__main__":
    # Allow the BEM tests to run as a module or through unittest discovery.
    unittest.main()
