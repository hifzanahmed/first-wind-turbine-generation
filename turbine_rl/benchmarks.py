"""Illustrative PID and empirical value-iteration baselines for BEM runs."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol

from turbine_rl.bem import (
    BEMControlAction,
    BEMWindEnvironment,
    annual_energy_mwh,
)


class DiscretePolicy(Protocol):
    """A policy that selects one discrete action from a normalized state."""

    def select_action(self, observation: list[float]) -> int: ...


@dataclass(frozen=True)
class BenchmarkResult:
    """Reward and annualized mean-power estimate for one controller run."""

    reward: float
    annual_energy_mwh: float


class PIDController:
    """Simple independent proportional control loops mapped onto discrete actions."""

    def __init__(self, environment: BEMWindEnvironment) -> None:
        # Retain the actuator table and initialize independent PID controller state.
        self.environment = environment
        self.reset()

    def reset(self) -> None:
        # Clear integral and derivative history at the start of each evaluation.
        self._integrals = {"yaw": 0.0, "rotor": 0.0, "pitch": 0.0}
        self._previous_errors: dict[str, float] = {}

    def select_action(self, observation: list[float]) -> int:
        # Track nominal rotor speed, wind direction, and pitch with discrete PID outputs.
        spec = self.environment.spec
        wind_speed = observation[0] * spec.rated_wind_speed_m_s
        yaw_error = math.degrees(math.atan2(observation[1], observation[2]))
        pitch = observation[4] * spec.max_pitch_deg
        rotor_target = observation[6] * spec.max_rotor_speed_rad_s
        desired_speed = self.environment._optimal_speed(wind_speed)
        desired_pitch = (
            min(spec.max_pitch_deg, max(2.0, (wind_speed - spec.rated_wind_speed_m_s) * 2.0 + 2.0))
            if wind_speed > spec.rated_wind_speed_m_s
            else 2.0
        )
        rotor_signal = self._pid_output(
            "rotor", desired_speed - rotor_target, 1.0, 0.05, 0.05, 2.0
        )
        yaw_signal = self._pid_output("yaw", yaw_error, 0.8, 0.01, 0.05, 45.0)
        pitch_signal = self._pid_output(
            "pitch", desired_pitch - pitch, 0.8, 0.02, 0.05, 10.0
        )
        rotor_delta = self._discrete_step(rotor_signal, 0.03, 0.1)
        yaw_delta = self._discrete_step(yaw_signal, 2.0, 5.0)
        pitch_delta = self._discrete_step(pitch_signal, 0.2, 1.0)
        return self.environment.ACTIONS.index(
            self._action(rotor_delta, yaw_delta, pitch_delta)
        )

    def _pid_output(
        self,
        name: str,
        error: float,
        proportional_gain: float,
        integral_gain: float,
        derivative_gain: float,
        integral_limit: float,
    ) -> float:
        # Combine bounded integral and finite-difference derivative terms with proportional error.
        dt = self.environment.spec.time_step_s
        integral = max(
            -integral_limit,
            min(integral_limit, self._integrals[name] + error * dt),
        )
        previous_error = self._previous_errors.get(name, error)
        derivative = (error - previous_error) / dt
        self._integrals[name] = integral
        self._previous_errors[name] = error
        return (
            proportional_gain * error
            + integral_gain * integral
            + derivative_gain * derivative
        )

    @staticmethod
    def _discrete_step(error: float, deadband: float, step: float) -> float:
        # Convert a signed controller error into a bounded discrete command.
        if error > deadband:
            return step
        if error < -deadband:
            return -step
        return 0.0

    @staticmethod
    def _action(
        rotor_delta: float, yaw_delta: float, pitch_delta: float
    ) -> BEMControlAction:
        # Construct the exact action-table entry without duplicating action ordering.
        return BEMControlAction(rotor_delta, yaw_delta, pitch_delta)


class ValueIterationController:
    """Tabular VI baseline fitted from sampled transitions of the BEM environment."""

    def __init__(
        self,
        environment: BEMWindEnvironment,
        discount_factor: float = 0.95,
    ) -> None:
        # Initialize a compact empirical transition model and safe fallback policy.
        if not 0.0 <= discount_factor < 1.0:
            raise ValueError("Discount factor must be in [0, 1)")
        self.environment = environment
        self.discount_factor = discount_factor
        self._policy: dict[tuple[int, ...], int] = {}
        self._fallback = PIDController(environment)

    def fit(
        self,
        sample_steps: int = 5000,
        iterations: int = 100,
        seed: int = 7,
        tolerance: float = 1e-4,
    ) -> None:
        # Estimate a coarse MDP from random BEM rollouts, then solve it by value iteration.
        if sample_steps < 1 or iterations < 1 or tolerance <= 0:
            raise ValueError("Sample steps, iterations, and tolerance must be positive")
        rng = random.Random(seed)
        transitions: dict[
            tuple[tuple[int, ...], int],
            list[tuple[tuple[int, ...], float, int]],
        ] = defaultdict(list)
        state = self.environment.reset()
        for _ in range(sample_steps):
            # Collect one stochastic-policy transition for the empirical model.
            action = rng.randrange(self.environment.action_size)
            next_state, reward, done = self.environment.step(action)
            transitions[self._state_key(state), action].append(
                (self._state_key(next_state), reward, int(done))
            )
            state = self.environment.reset() if done else next_state

        states = {
            state_key
            for state_action in transitions
            for state_key in (state_action[0],)
        }
        values = {state_key: 0.0 for state_key in states}
        policy: dict[tuple[int, ...], int] = {}
        for _ in range(iterations):
            # Bellman backups use only observed actions and empirical next-state frequencies.
            delta = 0.0
            updated_values: dict[tuple[int, ...], float] = {}
            for state_key in states:
                action_values: list[tuple[float, int]] = []
                for (source, action), outcomes in transitions.items():
                    if source != state_key:
                        continue
                    action_values.append(
                        (self._expected_return(outcomes, values), action)
                    )
                if action_values:
                    best_value, best_action = max(action_values)
                    policy[state_key] = best_action
                else:
                    best_value = 0.0
                updated_values[state_key] = best_value
                delta = max(delta, abs(best_value - values[state_key]))
            values = updated_values
            if delta < tolerance:
                break
        self._policy = policy

    def select_action(self, observation: list[float]) -> int:
        # Use the VI policy for known bins and a PID rule for unseen bins.
        state_key = self._state_key(observation)
        if state_key in self._policy:
            return self._policy[state_key]
        return self._fallback.select_action(observation)

    def _state_key(self, observation: list[float]) -> tuple[int, ...]:
        # Quantize continuous BEM features into a small reproducible table state.
        yaw_error = math.degrees(math.atan2(observation[1], observation[2]))
        return (
            min(2, max(0, int(observation[0] * 3))),
            min(2, max(0, int((yaw_error + 30.0) / 20.0))),
            min(1, max(0, int(observation[3] * 2))),
            min(1, max(0, int(observation[4] * 2))),
            min(1, max(0, int(observation[6] * 2))),
        )

    def _expected_return(
        self,
        outcomes: list[tuple[tuple[int, ...], float, int]],
        values: dict[tuple[int, ...], float],
    ) -> float:
        # Average immediate reward and discounted continuation over observed outcomes.
        return sum(
            reward
            + self.discount_factor * (1 - done) * values.get(next_state, 0.0)
            for next_state, reward, done in outcomes
        ) / len(outcomes)


def evaluate_policy(
    environment: BEMWindEnvironment, policy: DiscretePolicy
) -> BenchmarkResult:
    # Run one greedy rollout and report reward plus annualized mean power.
    reset = getattr(policy, "reset", None)
    if reset is not None:
        reset()
    state = environment.reset()
    total_reward = 0.0
    done = False
    while not done:
        state, reward, done = environment.step(policy.select_action(state))
        total_reward += reward
    return BenchmarkResult(
        reward=total_reward,
        annual_energy_mwh=annual_energy_mwh(
            environment.episode_energy_j, environment.duration_seconds
        ),
    )


def compare_baselines(
    training_environment: BEMWindEnvironment,
    evaluation_environment: BEMWindEnvironment,
    vi_sample_steps: int = 2000,
    seed: int = 7,
) -> dict[str, BenchmarkResult]:
    # Fit VI on training winds and compare both baselines on held-out winds.
    pid_result = evaluate_policy(
        evaluation_environment, PIDController(evaluation_environment)
    )
    value_iteration = ValueIterationController(training_environment)
    value_iteration.fit(sample_steps=vi_sample_steps, seed=seed)
    vi_result = evaluate_policy(evaluation_environment, value_iteration)
    return {"PID": pid_result, "Value iteration": vi_result}
