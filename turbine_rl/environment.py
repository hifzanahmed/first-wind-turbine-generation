"""Environment interfaces and the CSV-backed mock wind simulator."""

from __future__ import annotations

import csv
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class Environment(Protocol):
    """Minimal interface required by the training loop."""

    # Number of normalized state features consumed by the Q-network.
    @property
    def observation_size(self) -> int: ...

    # Number of discrete actions available to the agent.
    @property
    def action_size(self) -> int: ...

    # Reset an episode and return its initial state vector.
    def reset(self) -> list[float]: ...

    # Apply one action and return next state, scalar reward, and terminal flag.
    def step(self, action: int) -> tuple[list[float], float, bool]: ...


@dataclass(frozen=True)
class WindSample:
    wind_speed: float
    yaw_error: float
    rotor_speed: float
    blade_pitch: float
    rotor_yaw: float


@dataclass(frozen=True)
class ControlAction:
    yaw_delta: float
    pitch_delta: float
    torque_delta: float



class CsvWindEnvironment:
    """CSV-backed mock with discrete yaw, blade-pitch, and torque controls.

    Rotor speed is an observed state, not a direct actuator. Its response to
    pitch and torque is a simple proxy for this mock environment, not a turbine
    dynamics model.
    """

    REQUIRED_COLUMNS = (
        "wind_speed",
        "yaw_error",
        "rotor_speed",
        "blade_pitch",
        "rotor_yaw",
    )
    YAW_ACTIONS = (-2.0, 0.0, 2.0)
    PITCH_ACTIONS = (-1.0, 0.0, 1.0)
    TORQUE_ACTIONS = (-0.1, 0.0, 0.1)
    CONTROL_ACTIONS = tuple(
        ControlAction(*values)
        for values in itertools.product(YAW_ACTIONS, PITCH_ACTIONS, TORQUE_ACTIONS)
    )
    MAX_YAW_OFFSET = 30.0
    MIN_BLADE_PITCH = 0.0
    MAX_BLADE_PITCH = 25.0
    MIN_TORQUE_COMMAND = 0.0
    MAX_TORQUE_COMMAND = 1.0
    BASE_TORQUE_COMMAND = 0.5
    RATED_WIND_SPEED = 12.0
    RATED_ROTOR_SPEED = 2.0
    MAX_ROTOR_SPEED = 2.0
    ROTOR_SPEED_PER_TORQUE = 0.5
    ROTOR_SPEED_PER_PITCH_DEGREE = 0.01
    OVERSPEED_PENALTY = 5.0

    def __init__(
        self, csv_path: Path, start_index: int = 0, end_index: int | None = None
    ) -> None:
        # Load only the requested contiguous time range for leakage-free splits.
        samples = self._load_samples(csv_path)
        stop_index = len(samples) if end_index is None else end_index
        if not 0 <= start_index < stop_index <= len(samples):
            raise ValueError("CSV sample range must be non-empty and within the file")
        self.samples = samples[start_index:stop_index]
        self._index = 0
        self._yaw_offset = 0.0
        self._pitch_offset = 0.0
        self._torque_command = self.BASE_TORQUE_COMMAND

    @classmethod
    def _load_samples(cls, csv_path: Path) -> list[WindSample]:
        # Parse and validate every required SCADA-like field before training.
        samples: list[WindSample] = []
        with csv_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {csv_path}")

            missing = set(cls.REQUIRED_COLUMNS).difference(reader.fieldnames)
            if missing:
                raise ValueError(
                    f"CSV is missing required columns: {', '.join(sorted(missing))}"
                )
            for line_number, row in enumerate(reader, start=2):
                try:
                    values = {
                        name: float(row[name])
                        for name in cls.REQUIRED_COLUMNS
                    }
                    if not all(math.isfinite(value) for value in values.values()):
                        raise ValueError("values must be finite")
                    if values["wind_speed"] < 0 or values["rotor_speed"] < 0:
                        raise ValueError("wind and rotor speeds must be non-negative")
                    if not 0.0 <= values["blade_pitch"] <= 90.0:
                        raise ValueError("blade_pitch must be between 0 and 90 degrees")
                    samples.append(WindSample(**values))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid turbine reading on CSV line {line_number}: {exc}"
                    ) from exc

        if not samples:
            raise ValueError(f"CSV contains no data rows: {csv_path}")
        return samples

    @property
    def observation_size(self) -> int:
        # Expose the fixed feature count expected by the Q-network.
        return 8

    @property
    def action_size(self) -> int:
        # Expose the number of discrete actuator-command combinations.
        return len(self.CONTROL_ACTIONS)

    def reset(self) -> list[float]:
        # Start a fresh episode with neutral controls over the selected time range.
        self._index = 0
        self._yaw_offset = 0.0
        self._pitch_offset = 0.0
        self._torque_command = self.BASE_TORQUE_COMMAND
        return self._observation()

    def step(self, action: int) -> tuple[list[float], float, bool]:
        # Apply bounded controls, score the mock power/safety objective, advance time.
        if not 0 <= action < self.action_size:
            raise ValueError(f"Action must be in [0, {self.action_size - 1}]")

        sample = self.samples[self._index]
        control = self.CONTROL_ACTIONS[action]
        self._yaw_offset = max(
            -self.MAX_YAW_OFFSET,
            min(
                self.MAX_YAW_OFFSET,
                self._yaw_offset + control.yaw_delta,
            ),
        )
        current_pitch = self._blade_pitch(sample)
        next_pitch = max(
            self.MIN_BLADE_PITCH,
            min(self.MAX_BLADE_PITCH, current_pitch + control.pitch_delta),
        )
        self._pitch_offset = next_pitch - sample.blade_pitch
        self._torque_command = max(
            self.MIN_TORQUE_COMMAND,
            min(
                self.MAX_TORQUE_COMMAND,
                self._torque_command + control.torque_delta,
            ),
        )

        residual_yaw_error = sample.yaw_error - self._yaw_offset
        normalized_wind = min(sample.wind_speed / self.RATED_WIND_SPEED, 1.0)
        yaw_efficiency = max(math.cos(math.radians(residual_yaw_error)), 0.0) ** 3
        rotor_speed = self._rotor_speed(sample)
        power_proxy = (
            normalized_wind**3
            * yaw_efficiency
            * self._torque_command
            * min(rotor_speed / self.RATED_ROTOR_SPEED, 1.0)
        )
        overspeed = max(rotor_speed - self.MAX_ROTOR_SPEED, 0.0)
        reward = power_proxy - self.OVERSPEED_PENALTY * (
            overspeed / self.MAX_ROTOR_SPEED
        ) ** 2

        self._index += 1
        terminated = self._index >= len(self.samples)
        next_observation = (
            self._observation()
            if not terminated
            else self._observation(self.samples[-1])
        )
        return next_observation, reward, terminated

    def _observation(self, sample: WindSample | None = None) -> list[float]:
        # Normalize sensor channels and append current controller settings.
        current = sample if sample is not None else self.samples[self._index]
        rotor_yaw_radians = math.radians(current.rotor_yaw)
        return [
            current.wind_speed / 25.0,
            (current.yaw_error - self._yaw_offset) / 30.0,
            self._rotor_speed(current) / self.RATED_ROTOR_SPEED,
            self._blade_pitch(current) / self.MAX_BLADE_PITCH,
            math.sin(rotor_yaw_radians),
            math.cos(rotor_yaw_radians),
            self._yaw_offset / self.MAX_YAW_OFFSET,
            self._torque_command,
        ]

    def _blade_pitch(self, sample: WindSample) -> float:
        # Return the bounded pitch command represented by this mock controller.
        return max(
            self.MIN_BLADE_PITCH,
            min(
                self.MAX_BLADE_PITCH,
                sample.blade_pitch + self._pitch_offset,
            ),
        )

    def _rotor_speed(self, sample: WindSample) -> float:
        # Approximate rotor response to pitch and torque without claiming physics fidelity.
        torque_response = (
            self._torque_command - self.BASE_TORQUE_COMMAND
        ) * self.ROTOR_SPEED_PER_TORQUE
        pitch_response = (
            self._blade_pitch(sample) - sample.blade_pitch
        ) * self.ROTOR_SPEED_PER_PITCH_DEGREE
        return max(sample.rotor_speed - torque_response - pitch_response, 0.0)
