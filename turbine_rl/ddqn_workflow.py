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
from turbine_rl.config import CONFIG
from turbine_rl.environment import CsvWindEnvironment
from turbine_rl.training import EpisodeMetrics, evaluate, train


def _parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    # Provide configurable offline training and evaluation from the command line.
    parser = argparse.ArgumentParser(description="Train/evaluate the offline DDQN mock.")
    parser.add_argument("--csv", type=Path, default=CONFIG.csv_path, help="Input CSV path")
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
    # Run configured training/validation or checkpoint-only evaluation.
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
    if args.eval_only and args.load is None:
        raise ValueError("--eval-only requires --load CHECKPOINT")

    _seed_everything(args.seed)

    full_environment = CsvWindEnvironment(args.csv)
    training_environment = full_environment
    validation_environment = None
    if not args.eval_only and args.validation_fraction > 0.0:
        # Hold out the latest contiguous time segment to avoid temporal leakage.
        validation_count = max(
            1, round(len(full_environment.samples) * args.validation_fraction)
        )
        training_end = len(full_environment.samples) - validation_count
        if training_end < 1:
            raise ValueError("CSV needs at least two rows for train/validation splitting")
        training_environment = CsvWindEnvironment(
            args.csv, end_index=training_end
        )
        validation_environment = CsvWindEnvironment(
            args.csv, start_index=training_end
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
        print(f"Greedy evaluation reward: {evaluate(full_environment, agent):.3f}")
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
    else:
        print(
            "Greedy training-data reward: "
            f"{evaluate(training_environment, agent):.3f}"
        )
