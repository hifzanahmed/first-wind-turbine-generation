"""Illustrative blade-element/momentum turbine environment and wind generator."""

from __future__ import annotations

import csv
import itertools
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class TurbineSpec:
    """Generic turbine inputs; replace every default with the studied turbine data."""

    rotor_radius_m: float = 30.0
    hub_radius_m: float = 2.4
    blade_count: int = 3
    radial_fractions: tuple[float, ...] = (0.2, 0.35, 0.5, 0.65, 0.8, 0.95)
    chord_m: tuple[float, ...] = (3.2, 3.0, 2.6, 2.1, 1.6, 1.0)
    twist_deg: tuple[float, ...] = (14.0, 9.0, 5.5, 3.0, 1.2, 0.0)
    air_density_kg_m3: float = 1.225
    rated_wind_speed_m_s: float = 12.0
    rated_power_w: float = 2_000_000.0
    min_rotor_speed_rad_s: float = 0.3
    max_rotor_speed_rad_s: float = 2.5
    rotor_inertia_kg_m2: float = 4_000_000.0
    generator_gain_nm_per_rad_s: float = 100_000.0
    time_step_s: float = 1.0
    max_pitch_deg: float = 25.0
    max_yaw_error_deg: float = 30.0
    tip_speed_ratio: float = 7.0

    def __post_init__(self) -> None:
        # Reject geometry or operating data that would make the BEM calculation invalid.
        if len(self.radial_fractions) != len(self.chord_m) or len(self.chord_m) != len(self.twist_deg):
            raise ValueError("Radial fractions, chord, and twist arrays must match")
        if not 2 <= self.blade_count or not 0 < self.hub_radius_m < self.rotor_radius_m:
            raise ValueError("Blade count and hub/rotor radii are invalid")
        if any(
            not self.hub_radius_m / self.rotor_radius_m < radius < 1
            for radius in self.radial_fractions
        ):
            raise ValueError("Radial fractions must fall outside the hub and inside the tip")
        if tuple(sorted(self.radial_fractions)) != self.radial_fractions:
            raise ValueError("Radial fractions must be in increasing order")
        if any(chord <= 0 for chord in self.chord_m):
            raise ValueError("Blade chord lengths must be positive")
        if (
            self.rated_power_w <= 0
            or self.rated_wind_speed_m_s <= 0
            or self.rotor_inertia_kg_m2 <= 0
            or self.air_density_kg_m3 <= 0
            or self.time_step_s <= 0
            or self.max_rotor_speed_rad_s <= self.min_rotor_speed_rad_s
            or self.min_rotor_speed_rad_s <= 0
            or self.max_pitch_deg <= 0
            or self.generator_gain_nm_per_rad_s < 0
            or self.tip_speed_ratio <= 0
        ):
            raise ValueError("Turbine operating parameters are outside valid ranges")


@dataclass(frozen=True)
class WindCondition:
    """One time-aligned ambient wind-speed and direction observation."""

    speed_m_s: float
    direction_error_deg: float


@dataclass(frozen=True)
class BEMResult:
    """Integrated aerodynamic loads computed over all blade elements."""

    power_w: float
    torque_nm: float
    thrust_n: float


def generate_turbulent_wind(
    steps: int,
    mean_speed_m_s: float = 8.0,
    turbulence_intensity: float = 0.12,
    seed: int = 7,
    time_step_s: float = 1.0,
) -> tuple[WindCondition, ...]:
    # Generate reproducible correlated speed/direction fluctuations as a synthetic profile.
    if steps < 1 or mean_speed_m_s <= 0 or time_step_s <= 0:
        raise ValueError("Steps, mean wind speed, and time step must be positive")
    if not 0.0 <= turbulence_intensity <= 0.5:
        raise ValueError("Turbulence intensity must be between zero and 0.5")
    rng = random.Random(seed)
    speed_noise = 0.0
    direction_noise = 0.0
    profile: list[WindCondition] = []
    correlation = math.exp(-time_step_s / 10.0)
    for _ in range(steps):
        # Use a correlated Gaussian process rather than independent random gusts.
        speed_noise = (
            correlation * speed_noise
            + math.sqrt(1.0 - correlation**2)
            * rng.gauss(0.0, mean_speed_m_s * turbulence_intensity)
        )
        direction_noise = (
            correlation * direction_noise
            + math.sqrt(1.0 - correlation**2) * rng.gauss(0.0, 5.0)
        )
        profile.append(
            WindCondition(max(0.1, mean_speed_m_s + speed_noise), direction_noise)
        )
    return tuple(profile)


class BladeElementMomentumModel:
    """Steady BEM solver using an explicitly approximate generic airfoil polar."""

    def __init__(self, spec: TurbineSpec = TurbineSpec()) -> None:
        # Keep geometry and solver settings together for repeatable simulations.
        self.spec = spec

    def calculate(
        self,
        wind_speed_m_s: float,
        rotor_speed_rad_s: float,
        pitch_deg: float,
        yaw_error_deg: float,
        max_iterations: int = 100,
        tolerance: float = 1e-5,
    ) -> BEMResult:
        # Iterate axial/tangential induction at each blade station and integrate loads.
        if wind_speed_m_s < 0 or rotor_speed_rad_s <= 0 or pitch_deg < 0:
            raise ValueError("Wind must be non-negative; rotor speed positive; pitch non-negative")
        spec = self.spec
        axial_wind = wind_speed_m_s * max(
            math.cos(math.radians(yaw_error_deg)), 0.0
        )
        if axial_wind <= 1e-6:
            return BEMResult(0.0, 0.0, 0.0)
        radii = [fraction * spec.rotor_radius_m for fraction in spec.radial_fractions]
        edges = [spec.hub_radius_m]
        edges.extend((left + right) / 2.0 for left, right in zip(radii, radii[1:]))
        edges.append(spec.rotor_radius_m)
        total_torque = 0.0
        total_thrust = 0.0
        for index, radius in enumerate(radii):
            # Solve one annular blade element with Prandtl hub and tip-loss factors.
            annulus_width = edges[index + 1] - edges[index]
            axial_induction = 0.25
            tangential_induction = 0.0
            for _ in range(max_iterations):
                axial_velocity = axial_wind * (1.0 - axial_induction)
                tangential_velocity = rotor_speed_rad_s * radius * (
                    1.0 + tangential_induction
                )
                inflow_angle = math.atan2(axial_velocity, tangential_velocity)
                sin_phi = max(abs(math.sin(inflow_angle)), 1e-4)
                angle_of_attack = inflow_angle - math.radians(
                    spec.twist_deg[index] + pitch_deg
                )
                lift_coefficient = max(
                    -1.5, min(1.5, 2.0 * math.pi * angle_of_attack)
                )
                drag_coefficient = 0.008 + 0.01 * lift_coefficient**2
                normal_coefficient = (
                    lift_coefficient * math.cos(inflow_angle)
                    + drag_coefficient * math.sin(inflow_angle)
                )
                tangential_coefficient = (
                    lift_coefficient * math.sin(inflow_angle)
                    - drag_coefficient * math.cos(inflow_angle)
                )
                solidity = (
                    spec.blade_count * spec.chord_m[index] / (2.0 * math.pi * radius)
                )
                tip_loss = self._loss_factor(
                    spec.blade_count,
                    (spec.rotor_radius_m - radius) / (radius * sin_phi),
                )
                hub_loss = self._loss_factor(
                    spec.blade_count,
                    (radius - spec.hub_radius_m) / (spec.hub_radius_m * sin_phi),
                )
                loss = max(tip_loss * hub_loss, 0.05)
                # Clamp unstable/high-induction solutions in this simplified solver.
                cn = max(normal_coefficient, 0.05)
                axial_denominator = (
                    4.0 * loss * math.sin(inflow_angle) ** 2 / (solidity * cn) + 1.0
                )
                next_axial = max(0.0, min(0.45, 1.0 / axial_denominator))
                ct = tangential_coefficient
                tangential_denominator = (
                    4.0
                    * loss
                    * math.sin(inflow_angle)
                    * math.cos(inflow_angle)
                    / (solidity * ct)
                    - 1.0
                    if abs(ct) > 1e-6
                    else math.inf
                )
                next_tangential = (
                    max(-0.5, min(1.5, 1.0 / tangential_denominator))
                    if math.isfinite(tangential_denominator)
                    and abs(tangential_denominator) > 1e-6
                    else tangential_induction
                )
                if max(
                    abs(next_axial - axial_induction),
                    abs(next_tangential - tangential_induction),
                ) < tolerance:
                    axial_induction = next_axial
                    tangential_induction = next_tangential
                    break
                axial_induction = 0.75 * axial_induction + 0.25 * next_axial
                tangential_induction = (
                    0.75 * tangential_induction + 0.25 * next_tangential
                )

            # Integrate final normal/tangential blade loads over this annulus.
            axial_velocity = axial_wind * (1.0 - axial_induction)
            tangential_velocity = rotor_speed_rad_s * radius * (
                1.0 + tangential_induction
            )
            relative_speed_squared = axial_velocity**2 + tangential_velocity**2
            dynamic_pressure_area = (
                0.5
                * spec.air_density_kg_m3
                * relative_speed_squared
                * spec.chord_m[index]
                * spec.blade_count
                * annulus_width
            )
            inflow_angle = math.atan2(axial_velocity, tangential_velocity)
            angle_of_attack = inflow_angle - math.radians(
                spec.twist_deg[index] + pitch_deg
            )
            lift_coefficient = max(
                -1.5, min(1.5, 2.0 * math.pi * angle_of_attack)
            )
            drag_coefficient = 0.008 + 0.01 * lift_coefficient**2
            normal = lift_coefficient * math.cos(inflow_angle) + drag_coefficient * math.sin(inflow_angle)
            tangential = lift_coefficient * math.sin(inflow_angle) - drag_coefficient * math.cos(inflow_angle)
            total_thrust += dynamic_pressure_area * normal
            total_torque += dynamic_pressure_area * tangential * radius

        # Convert integrated aerodynamic torque to non-negative shaft power.
        torque = max(total_torque, 0.0)
        return BEMResult(max(torque * rotor_speed_rad_s, 0.0), torque, max(total_thrust, 0.0))

    @staticmethod
    def _loss_factor(blade_count: int, exponent_argument: float) -> float:
        # Calculate a bounded Prandtl loss factor near blade tips or the hub.
        exponent = math.exp(-max(blade_count * exponent_argument / 2.0, 0.0))
        return 2.0 / math.pi * math.acos(max(-1.0, min(1.0, exponent)))


@dataclass(frozen=True)
class BEMControlAction:
    """Discrete commanded changes for rotor-speed setpoint, yaw, and pitch."""

    rotor_speed_delta_rad_s: float
    yaw_delta_deg: float
    pitch_delta_deg: float


class BEMWindEnvironment:
    """Turbulent-wind environment driven by a generic BEM turbine model."""

    ACTIONS = tuple(
        BEMControlAction(*values)
        for values in itertools.product((-0.1, 0.0, 0.1), (-5.0, 0.0, 5.0), (-1.0, 0.0, 1.0))
    )

    def __init__(
        self,
        wind_profile: Sequence[WindCondition] | None = None,
        spec: TurbineSpec = TurbineSpec(),
        wind_seed: int = 7,
        episode_steps: int = 300,
        mean_wind_speed_m_s: float = 8.0,
        turbulence_intensity: float = 0.12,
        wind_csv_path: Path | None = None,
        start_index: int = 0,
        end_index: int | None = None,
    ) -> None:
        # Prepare a deterministic wind episode from a profile, CSV, or synthetic process.
        self.spec = spec
        self.model = BladeElementMomentumModel(spec)
        if wind_profile is not None and wind_csv_path is not None:
            raise ValueError("Specify either wind_profile or wind_csv_path, not both")
        if wind_csv_path is not None:
            samples = self._load_wind_csv(wind_csv_path)
            stop_index = len(samples) if end_index is None else end_index
            if not 0 <= start_index < stop_index <= len(samples):
                raise ValueError("CSV sample range must be non-empty and within the file")
            self.wind_profile = tuple(samples[start_index:stop_index])
        elif wind_profile is not None:
            self.wind_profile = tuple(wind_profile)
        else:
            self.wind_profile = generate_turbulent_wind(
                episode_steps,
                mean_wind_speed_m_s,
                turbulence_intensity,
                wind_seed,
                spec.time_step_s,
            )
        if not self.wind_profile:
            raise ValueError("Wind profile must contain at least one time step")
        if any(
            not math.isfinite(sample.speed_m_s)
            or not math.isfinite(sample.direction_error_deg)
            or sample.speed_m_s < 0
            for sample in self.wind_profile
        ):
            raise ValueError("Wind profile readings must be finite with non-negative speed")
        self._index = 0
        self._episode_energy_j = 0.0
        self._last_power_w = 0.0
        self._rotor_speed_rad_s = spec.min_rotor_speed_rad_s
        self._rotor_speed_target_rad_s = spec.min_rotor_speed_rad_s
        self._yaw_heading_deg = 0.0
        self._pitch_deg = 2.0

    @staticmethod
    def _load_wind_csv(csv_path: Path) -> list[WindCondition]:
        # Validate and load the second input format for BEM wind time histories.
        profile: list[WindCondition] = []
        with csv_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
            reader = csv.DictReader(csv_file)
            required = {"wind_speed", "wind_direction_error"}
            if reader.fieldnames is None:
                raise ValueError(f"Wind CSV has no header: {csv_path}")
            missing = required.difference(reader.fieldnames)
            if missing:
                raise ValueError(
                    f"Wind CSV is missing columns: {', '.join(sorted(missing))}"
                )
            for line_number, row in enumerate(reader, start=2):
                try:
                    speed = float(row["wind_speed"])
                    direction = float(row["wind_direction_error"])
                    if not math.isfinite(speed) or not math.isfinite(direction):
                        raise ValueError("values must be finite")
                    if speed < 0:
                        raise ValueError("wind_speed must be non-negative")
                    profile.append(WindCondition(speed, direction))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid wind reading on CSV line {line_number}: {exc}"
                    ) from exc
        if not profile:
            raise ValueError(f"Wind CSV contains no data rows: {csv_path}")
        return profile

    @property
    def observation_size(self) -> int:
        # Return the feature count consumed by the DDQN network.
        return 9

    @property
    def action_size(self) -> int:
        # Return the number of rotor-speed/yaw/pitch command combinations.
        return len(self.ACTIONS)

    @property
    def episode_energy_j(self) -> float:
        # Expose electrical energy accumulated during the current episode.
        return self._episode_energy_j

    @property
    def duration_seconds(self) -> float:
        # Convert profile length to elapsed simulated seconds.
        return len(self.wind_profile) * self.spec.time_step_s

    def reset(self) -> list[float]:
        # Reset actuators and rotor state to a repeatable initial operating point.
        self._index = 0
        self._episode_energy_j = 0.0
        self._last_power_w = 0.0
        self._rotor_speed_rad_s = self.spec.min_rotor_speed_rad_s
        self._rotor_speed_target_rad_s = self._optimal_speed(
            self.wind_profile[0].speed_m_s
        )
        self._yaw_heading_deg = 0.0
        self._pitch_deg = 2.0
        return self._observation(self.wind_profile[0])

    def step(self, action: int) -> tuple[list[float], float, bool]:
        # Apply controls, run BEM loads, advance rotor dynamics, and accumulate energy.
        if not 0 <= action < self.action_size:
            raise ValueError(f"Action must be in [0, {self.action_size - 1}]")
        wind = self.wind_profile[self._index]
        control = self.ACTIONS[action]
        spec = self.spec
        self._rotor_speed_target_rad_s = self._clip(
            self._rotor_speed_target_rad_s + control.rotor_speed_delta_rad_s,
            spec.min_rotor_speed_rad_s,
            spec.max_rotor_speed_rad_s,
        )
        self._yaw_heading_deg = self._clip(
            self._yaw_heading_deg + control.yaw_delta_deg, -180.0, 180.0
        )
        self._pitch_deg = self._clip(
            self._pitch_deg + control.pitch_delta_deg, 0.0, spec.max_pitch_deg
        )
        yaw_error = self._wrap_degrees(wind.direction_error_deg - self._yaw_heading_deg)
        aerodynamic = self.model.calculate(
            wind.speed_m_s,
            self._rotor_speed_rad_s,
            self._pitch_deg,
            yaw_error,
        )
        generator_torque = self._clip(
            aerodynamic.torque_nm
            + spec.generator_gain_nm_per_rad_s
            * (self._rotor_speed_rad_s - self._rotor_speed_target_rad_s),
            0.0,
            spec.rated_power_w / spec.max_rotor_speed_rad_s,
        )
        acceleration = (
            aerodynamic.torque_nm - generator_torque
        ) / spec.rotor_inertia_kg_m2
        self._rotor_speed_rad_s = self._clip(
            self._rotor_speed_rad_s + acceleration * spec.time_step_s,
            spec.min_rotor_speed_rad_s,
            spec.max_rotor_speed_rad_s,
        )
        self._last_power_w = min(
            max(generator_torque * self._rotor_speed_rad_s * 0.94, 0.0),
            spec.rated_power_w,
        )
        self._episode_energy_j += self._last_power_w * spec.time_step_s
        overspeed = max(
            self._rotor_speed_rad_s - spec.max_rotor_speed_rad_s * 0.95, 0.0
        )
        reward = self._last_power_w / spec.rated_power_w - 5.0 * (
            overspeed / spec.max_rotor_speed_rad_s
        ) ** 2
        self._index += 1
        terminated = self._index >= len(self.wind_profile)
        next_wind = self.wind_profile[min(self._index, len(self.wind_profile) - 1)]
        return self._observation(next_wind), reward, terminated

    def _observation(self, wind: WindCondition) -> list[float]:
        # Normalize wind, yaw, rotor, pitch, setpoint, and recent generation state.
        yaw_error = self._wrap_degrees(wind.direction_error_deg - self._yaw_heading_deg)
        return [
            wind.speed_m_s / self.spec.rated_wind_speed_m_s,
            math.sin(math.radians(yaw_error)),
            math.cos(math.radians(yaw_error)),
            self._rotor_speed_rad_s / self.spec.max_rotor_speed_rad_s,
            self._pitch_deg / self.spec.max_pitch_deg,
            self._yaw_heading_deg / 180.0,
            self._rotor_speed_target_rad_s / self.spec.max_rotor_speed_rad_s,
            self._last_power_w / self.spec.rated_power_w,
            wind.direction_error_deg / 180.0,
        ]

    def _optimal_speed(self, wind_speed_m_s: float) -> float:
        # Set the nominal rotor speed from a tip-speed-ratio operating point.
        return self._clip(
            self.spec.tip_speed_ratio * wind_speed_m_s / self.spec.rotor_radius_m,
            self.spec.min_rotor_speed_rad_s,
            self.spec.max_rotor_speed_rad_s,
        )

    @staticmethod
    def _clip(value: float, lower: float, upper: float) -> float:
        # Keep a continuous command within its actuator or operating bounds.
        return max(lower, min(upper, value))

    @staticmethod
    def _wrap_degrees(angle: float) -> float:
        # Normalize angular error to the conventional minus/plus 180-degree range.
        return (angle + 180.0) % 360.0 - 180.0


def annual_energy_mwh(energy_j: float, duration_seconds: float) -> float:
    # Extrapolate mean simulated power to annual hours while keeping the unit explicit.
    if energy_j < 0 or duration_seconds <= 0:
        raise ValueError("Energy must be non-negative and duration must be positive")
    mean_power_w = energy_j / duration_seconds
    return mean_power_w * 8760.0 / 1_000_000.0
