import csv
import math
import tempfile
import unittest
from pathlib import Path

from turbine_rl.environment import ControlAction, CsvWindEnvironment


class CsvWindEnvironmentTests(unittest.TestCase):
    def make_environment(self, rows: list[dict[str, float]]) -> CsvWindEnvironment:
        # Write a temporary canonical CSV fixture for environment behavior tests.
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        csv_path = Path(temporary_directory.name) / "wind.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=CsvWindEnvironment.REQUIRED_COLUMNS,
            )
            writer.writeheader()
            writer.writerows(rows)
        return CsvWindEnvironment(csv_path)

    @staticmethod
    def action_index(yaw_delta: float, pitch_delta: float, torque_delta: float) -> int:
        # Resolve a named control combination to the agent's discrete action ID.
        return CsvWindEnvironment.CONTROL_ACTIONS.index(
            ControlAction(yaw_delta, pitch_delta, torque_delta)
        )

    def test_action_space_and_observation_include_control_state(self) -> None:
        # Confirm multi-actuator controls update the emitted state dimensions.
        environment = self.make_environment(
            [
                {
                    "wind_speed": 12.0,
                    "yaw_error": 0.0,
                    "rotor_speed": 1.4,
                    "blade_pitch": 2.0,
                    "rotor_yaw": 180.0,
                }
            ]
        )

        observation = environment.reset()
        action = self.action_index(2.0, 1.0, 0.1)
        next_observation, _, done = environment.step(action)

        self.assertEqual(environment.action_size, 27)
        self.assertEqual(environment.observation_size, 8)
        self.assertEqual(len(observation), 8)
        self.assertTrue(done)
        self.assertAlmostEqual(next_observation[1], -2.0 / 30.0)
        self.assertAlmostEqual(next_observation[2], 1.34 / 2.0)
        self.assertAlmostEqual(next_observation[3], 3.0 / 25.0)
        self.assertAlmostEqual(next_observation[7], 0.6)

    def test_controls_are_bounded_and_overspeed_is_penalized(self) -> None:
        # Confirm repeated actions respect limits and the reward penalizes overspeed.
        row = {
            "wind_speed": 12.0,
            "yaw_error": 0.0,
            "rotor_speed": 2.5,
            "blade_pitch": 24.0,
            "rotor_yaw": 180.0,
        }
        environment = self.make_environment([row] * 20)
        action = self.action_index(2.0, 1.0, 0.1)

        for _ in range(20):
            observation, reward, done = environment.step(action)

        self.assertTrue(done)
        self.assertAlmostEqual(observation[2], 2.24 / 2.0)
        self.assertEqual(observation[3], 1.0)
        self.assertEqual(observation[6], 1.0)
        self.assertEqual(observation[7], 1.0)
        self.assertAlmostEqual(reward, math.cos(math.radians(30.0)) ** 3 - 0.072)

        overspeed_environment = self.make_environment([row])
        _, baseline_reward, _ = overspeed_environment.step(
            self.action_index(0.0, 0.0, 0.0)
        )
        self.assertAlmostEqual(baseline_reward, 0.1875)

    def test_out_of_range_action_is_rejected(self) -> None:
        # Reject action IDs that are not part of the discrete action table.
        environment = self.make_environment(
            [
                {
                    "wind_speed": 12.0,
                    "yaw_error": 0.0,
                    "rotor_speed": 1.0,
                    "blade_pitch": 0.0,
                    "rotor_yaw": 0.0,
                }
            ]
        )

        with self.assertRaises(ValueError):
            environment.step(environment.action_size)

    def test_csv_rejects_non_finite_sensor_values(self) -> None:
        # Fail fast when a sensor row contains NaN rather than silently training on it.
        with self.assertRaisesRegex(ValueError, "finite"):
            self.make_environment(
                [
                    {
                        "wind_speed": math.nan,
                        "yaw_error": 0.0,
                        "rotor_speed": 1.0,
                        "blade_pitch": 0.0,
                        "rotor_yaw": 0.0,
                    }
                ]
            )

    def test_csv_supports_contiguous_time_slices(self) -> None:
        # Verify start/end indices create non-overlapping chronological segments.
        rows = [
            {
                "wind_speed": float(index + 1),
                "yaw_error": 0.0,
                "rotor_speed": 1.0,
                "blade_pitch": 0.0,
                "rotor_yaw": 0.0,
            }
            for index in range(4)
        ]
        whole = self.make_environment(rows)
        with tempfile.TemporaryDirectory() as temporary_directory:
            csv_path = Path(temporary_directory) / "wind.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=CsvWindEnvironment.REQUIRED_COLUMNS,
                )
                writer.writeheader()
                writer.writerows(rows)
            training = CsvWindEnvironment(csv_path, end_index=2)
            validation = CsvWindEnvironment(csv_path, start_index=2)

        self.assertEqual(len(whole.samples), 4)
        self.assertEqual([item.wind_speed for item in training.samples], [1.0, 2.0])
        self.assertEqual([item.wind_speed for item in validation.samples], [3.0, 4.0])


if __name__ == "__main__":
    # Allow running this test module directly as well as through discovery.
    unittest.main()
