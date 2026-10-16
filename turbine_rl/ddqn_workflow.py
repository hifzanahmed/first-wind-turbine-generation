"""DDQN training and evaluation workflow."""

from __future__ import annotations

import random

import torch

from turbine_rl.agent import DoubleDQNAgent
from turbine_rl.config import CONFIG
from turbine_rl.environment import CsvWindEnvironment
from turbine_rl.training import evaluate, train


def main() -> None:
    random.seed(CONFIG.seed)
    torch.manual_seed(CONFIG.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG.seed)

    # It will read CSV and store it to self.samples, where each row is a WindSample object. 
    # It will also initialize the index and yaw_offset to 0.
    environment = CsvWindEnvironment(CONFIG.csv_path)
    # It will create a DoubleDQNAgent object with the observation size and action size from the environment, as well as the batch size and target update interval from the config.
    agent = DoubleDQNAgent(
        observation_size=environment.observation_size,
        action_size=environment.action_size,
        batch_size=CONFIG.batch_size,
        target_update_interval=CONFIG.target_update_interval,
    )
    if CONFIG.load_path is not None:
        agent.load(CONFIG.load_path)

    if not CONFIG.eval_only:
        train(environment, agent, episodes=CONFIG.episodes)
        if CONFIG.checkpoint_path is not None:
            CONFIG.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            agent.save(CONFIG.checkpoint_path)

    print(f"Greedy evaluation reward: {evaluate(environment, agent):.3f}")
