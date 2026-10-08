"""Environment interfaces and the CSV-backed mock wind simulator."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class Environment(Protocol):
    """Minimal interface required by the training loop."""

    @property
    def observation_size(self) -> int: ...

    @property
    def action_size(self) -> int: ...

    def reset(self) -> list[float]: ...

    def step(self, action: int) -> tuple[list[float], float, bool]: ...


@dataclass(frozen=True)
class WindSample:
    wind_speed: float
    yaw_error: float
    rotor_speed: float
    blade_pitch: float
    rotor_yaw: float



class CsvWindEnvironment:
    """ CSV-backed wind environment.
        The environment stores the loaded wind samples, the current sample index,
        and the persistent yaw-offset controller state. Each CSV row represents one
        timestep of external conditions, and the agent updates the yaw offset
        through actions while the original samples remain unchanged.
    """

    REQUIRED_COLUMNS = (
        "wind_speed",
        "yaw_error",
        "rotor_speed",
        "blade_pitch",
        "rotor_yaw",
    )
    YAW_ACTIONS = (-2.0, 0.0, 2.0)
    MAX_YAW_OFFSET = 30.0
    RATED_WIND_SPEED = 12.0

    #This will run first and call load_samples to load the CSV file and store the samples in self.samples. 
    # It will also initialize the index and yaw_offset to 0.
    def __init__(self, csv_path: Path) -> None:
        self.samples = self._load_samples(csv_path)
        self._index = 0
        self._yaw_offset = 0.0

    @classmethod
    def _load_samples(cls, csv_path: Path) -> list[WindSample]:
        samples: list[WindSample] = []
        with csv_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
            # DictReader is help to read the CSV file into a dictionary format
            reader = csv.DictReader(csv_file)
            #Validate the CSV file has the required columns
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {csv_path}")

            missing = set(cls.REQUIRED_COLUMNS).difference(reader.fieldnames)
            if missing:
                raise ValueError(
                    f"CSV is missing required columns: {', '.join(sorted(missing))}"
                )
            # Read each row from the CSV, start from line 2 and convert to WindSample, handling errors
            for line_number, row in enumerate(reader, start=2):
                try:
                    samples.append(
                        WindSample(
                            wind_speed=float(row["wind_speed"]),
                            yaw_error=float(row["yaw_error"]),
                            rotor_speed=float(row["rotor_speed"]),
                            blade_pitch=float(row["blade_pitch"]),
                            rotor_yaw=float(row["rotor_yaw"]),
                        )
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid numeric value on CSV line {line_number}"
                    ) from exc

        if not samples:
            raise ValueError(f"CSV contains no data rows: {csv_path}")
        return samples

    @property
    def observation_size(self) -> int:
        return 7

    @property
    def action_size(self) -> int:
        return len(self.YAW_ACTIONS)

    def reset(self) -> list[float]:
        self._index = 0
        self._yaw_offset = 0.0
        return self._observation()

    def step(self, action: int) -> tuple[list[float], float, bool]:
        if not 0 <= action < self.action_size:
            raise ValueError(f"Action must be in [0, {self.action_size - 1}]")

        sample = self.samples[self._index]
        self._yaw_offset = max(
            -self.MAX_YAW_OFFSET,
            min(
                self.MAX_YAW_OFFSET,
                self._yaw_offset + self.YAW_ACTIONS[action],
            ),
        )
        residual_yaw_error = sample.yaw_error - self._yaw_offset
        normalized_wind = min(sample.wind_speed / self.RATED_WIND_SPEED, 1.0)
        yaw_efficiency = max(math.cos(math.radians(residual_yaw_error)), 0.0) ** 3
        reward = normalized_wind**3 * yaw_efficiency

        self._index += 1
        terminated = self._index >= len(self.samples)
        next_observation = (
            self._observation()
            if not terminated
            else self._observation(self.samples[-1])
        )
        return next_observation, reward, terminated

    def _observation(self, sample: WindSample | None = None) -> list[float]:
        current = sample if sample is not None else self.samples[self._index]
        rotor_yaw_radians = math.radians(current.rotor_yaw)
        return [
            current.wind_speed / 25.0,
            (current.yaw_error - self._yaw_offset) / 30.0,
            current.rotor_speed / 2.0,
            current.blade_pitch / 20.0,
            math.sin(rotor_yaw_radians),
            math.cos(rotor_yaw_radians),
            self._yaw_offset / self.MAX_YAW_OFFSET,
        ]
