"""DDQN training and evaluation workflow."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import torch

from turbine_rl.agent import DoubleDQNAgent
from turbine_rl.bem import BEMWindEnvironment, annual_energy_mwh
from turbine_rl.benchmarks import compare_baselines, evaluate_policy
from turbine_rl.config import CONFIG
from turbine_rl.environment import CsvWindEnvironment
from turbine_rl.training import EpisodeMetrics, evaluate, train


def _parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    # Provide configurable offline training and evaluation from the command line.
    parser = argparse.ArgumentParser(description="Train/evaluate the offline DDQN mock.")
    parser.add_argument(
        "--environment",
        choices=("bem", "csv-mock"),
        default="bem",
        help="Use the BEM simulation or the original CSV mock environment",
    )
    parser.add_argument("--csv", type=Path, default=CONFIG.csv_path, help="Input CSV path")
    parser.add_argument(
        "--wind-csv",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "TurbulentWind.csv",
        help="BEM wind profile (wind_speed, wind_direction_error)",
    )
    parser.add_argument(
        "--compare-baselines",
        action="store_true",
        help="Compare BEM DDQN with PID and empirical value iteration",
    )
    parser.add_argument("--vi-sample-steps", type=int, default=2000)
    parser.add_argument("--episodes", type=int, default=CONFIG.episodes)
    parser.add_argument("--batch-size", type=int, default=CONFIG.batch_size)
    parser.add_argument(
        "--target-update-interval", type=int, default=CONFIG.target_update_interval
    )
    parser.add_argument("--seed", type=int, default=CONFIG.seed)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=CONFIG.validation_fraction,
        help="Chronological holdout fraction; set 0 to disable validation",
    )
    parser.add_argument(
        "--validation-interval", type=int, default=CONFIG.validation_interval
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=CONFIG.early_stopping_patience,
        help="Enable early stopping after this many validation episodes without improvement",
    )
    parser.add_argument("--checkpoint", type=Path, default=CONFIG.checkpoint_path)
    parser.add_argument("--load", type=Path, default=CONFIG.load_path)
    parser.add_argument("--metrics", type=Path, default=CONFIG.metrics_path)
    parser.add_argument("--eval-only", action="store_true", default=CONFIG.eval_only)
    return parser.parse_args(arguments)


def _seed_everything(seed: int) -> None:
    # Seed Python and PyTorch generators for repeatable offline experiments.
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_metrics(path: Path, metrics: list[EpisodeMetrics]) -> None:
    # Export the structured episode history as a portable JSON artifact.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(metric) for metric in metrics], indent=2),
        encoding="utf-8",
    )


def main(arguments: Sequence[str] | None = None) -> None:
    # Run DDQN with either the BEM model or the original CSV-based mock.
    args = _parse_args(arguments)
    if (
        args.episodes < 1
        or args.batch_size < 1
        or args.target_update_interval < 1
    ):
        raise ValueError(
            "episodes, batch size, and target update interval must be positive"
        )
    if not 0.0 <= args.validation_fraction < 1.0:
        raise ValueError("validation fraction must be in [0, 1)")
    if args.validation_interval < 1:
        raise ValueError("validation interval must be positive")
    if args.patience is not None and args.patience < 1:
        raise ValueError("patience must be positive")
    if args.vi_sample_steps < 1:
        raise ValueError("value-iteration sample steps must be positive")
    if args.eval_only and args.load is None:
        raise ValueError("--eval-only requires --load CHECKPOINT")
    if args.compare_baselines and args.environment != "bem":
        raise ValueError("--compare-baselines requires --environment bem")
    if args.compare_baselines and args.eval_only:
        raise ValueError("--compare-baselines is available during training runs")

    _seed_everything(args.seed)

    full_environment = _make_environment(args)
    training_environment = full_environment
    validation_environment = None
    if not args.eval_only and args.validation_fraction > 0.0:
        # Hold out the latest contiguous time segment to avoid temporal leakage.
        sample_count = _environment_length(full_environment)
        validation_count = max(1, round(sample_count * args.validation_fraction))
        training_end = sample_count - validation_count
        if training_end < 1:
            raise ValueError("CSV needs at least two rows for train/validation splitting")
        training_environment = _make_environment(
            args, start_index=0, end_index=training_end
        )
        validation_environment = _make_environment(
            args, start_index=training_end
        )

    agent = DoubleDQNAgent(
        observation_size=full_environment.observation_size,
        action_size=full_environment.action_size,
        batch_size=args.batch_size,
        target_update_interval=args.target_update_interval,
    )
    if args.load is not None:
        agent.load(args.load)

    if args.eval_only:
        reward = evaluate(full_environment, agent)
        print(f"Greedy evaluation reward: {reward:.3f}")
        _print_annual_energy("DDQN", full_environment)
        return

    history = train(
        training_environment,
        agent,
        episodes=args.episodes,
        validation_environment=validation_environment,
        validation_interval=args.validation_interval,
        early_stopping_patience=args.patience,
    )
    if args.checkpoint is not None:
        # Save a versioned checkpoint after the final training episode.
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        agent.save(args.checkpoint)
    if args.metrics is not None:
        _write_metrics(args.metrics, history)
    if validation_environment is not None:
        print(
            "Greedy validation reward: "
            f"{evaluate(validation_environment, agent):.3f}"
        )
        _print_annual_energy("DDQN validation", validation_environment)
    else:
        print(
            "Greedy training-data reward: "
            f"{evaluate(training_environment, agent):.3f}"
        )
        _print_annual_energy("DDQN", training_environment)

    if args.compare_baselines:
        if not isinstance(training_environment, BEMWindEnvironment):
            raise TypeError("BEM baseline comparison requires a BEM environment")
        evaluation_environment = (
            validation_environment
            if isinstance(validation_environment, BEMWindEnvironment)
            else training_environment
        )
        results = compare_baselines(
            training_environment,
            evaluation_environment,
            vi_sample_steps=args.vi_sample_steps,
            seed=args.seed,
        )
        for name, result in results.items():
            print(
                f"{name} reward: {result.reward:.3f} | "
                f"annualized mean-power estimate: {result.annual_energy_mwh:.3f} MWh"
            )
        ddqn_result = evaluate_policy(evaluation_environment, agent)
        print(
            f"DDQN annualized mean-power estimate: "
            f"{ddqn_result.annual_energy_mwh:.3f} MWh"
        )


def _make_environment(
    args: argparse.Namespace,
    start_index: int = 0,
    end_index: int | None = None,
) -> CsvWindEnvironment | BEMWindEnvironment:
    # Construct the selected environment over a requested chronological segment.
    if args.environment == "bem":
        return BEMWindEnvironment(
            wind_csv_path=args.wind_csv,
            start_index=start_index,
            end_index=end_index,
        )
    return CsvWindEnvironment(
        args.csv,
        start_index=start_index,
        end_index=end_index,
    )


def _environment_length(
    environment: CsvWindEnvironment | BEMWindEnvironment,
) -> int:
    # Return the number of time-aligned rows available for chronological splitting.
    if isinstance(environment, BEMWindEnvironment):
        return len(environment.wind_profile)
    return len(environment.samples)


def _print_annual_energy(label: str, environment: object) -> None:
    # Report a clearly labeled mean-power annualization only for BEM simulations.
    if isinstance(environment, BEMWindEnvironment):
        estimate = annual_energy_mwh(
            environment.episode_energy_j, environment.duration_seconds
        )
        print(f"{label} annualized mean-power estimate: {estimate:.3f} MWh")
