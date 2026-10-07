"""Default paths and run settings for the DDQN application."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    csv_path: Path = Path(__file__).resolve().parent.parent / "data" / "Input.csv"
    episodes: int = 300
    batch_size: int = 64
    target_update_interval: int = 100
    seed: int = 7
    checkpoint_path: Path | None = None
    load_path: Path | None = None
    eval_only: bool = False


CONFIG = TrainingConfig()
