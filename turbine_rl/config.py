"""Default paths and run settings for the DDQN application."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    """Defaults for offline CSV training and held-out evaluation."""

    csv_path: Path = Path(__file__).resolve().parent.parent / "data" / "Input.csv"
    episodes: int = 300
    batch_size: int = 64
    target_update_interval: int = 100
    seed: int = 7
    validation_fraction: float = 0.2
    validation_interval: int = 1
    early_stopping_patience: int | None = None
    checkpoint_path: Path | None = None
    load_path: Path | None = None
    metrics_path: Path | None = None
    eval_only: bool = False


CONFIG = TrainingConfig()
