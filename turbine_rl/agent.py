"""Double Deep Q-Network agent and experience replay."""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Sequence

import torch
from torch import nn
from torch.optim import Adam


@dataclass
class Experience:
    state: list[float]
    action: int
    reward: float
    next_state: list[float]
    done: bool


class ReplayBuffer:
    """Proportional prioritized replay with importance-sampling correction."""

    def __init__(
        self, capacity: int, priority_alpha: float = 0.6, priority_epsilon: float = 1e-5
    ) -> None:
        # Validate replay settings before allocating the bounded storage.
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        if priority_alpha < 0.0 or priority_epsilon <= 0.0:
            raise ValueError("Priority alpha must be non-negative and epsilon positive")
        self.capacity = capacity
        self.priority_alpha = priority_alpha
        self.priority_epsilon = priority_epsilon
        self._items: Deque[Experience] = deque(maxlen=capacity)
        self._priorities: Deque[float] = deque(maxlen=capacity)

    def add(self, experience: Experience) -> None:
        # New transitions start at the current maximum so they are sampled promptly.
        self._items.append(experience)
        self._priorities.append(max(self._priorities, default=1.0))

    def sample(
        self, batch_size: int, beta: float = 0.4
    ) -> tuple[list[Experience], list[int], list[float]]:
        # Sample by scaled TD-error priorities and return normalized IS weights.
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        if not 0.0 <= beta <= 1.0:
            raise ValueError("Importance-sampling beta must be between zero and one")
        scaled_priorities = [
            priority**self.priority_alpha for priority in self._priorities
        ]
        total_priority = sum(scaled_priorities)
        probabilities = [priority / total_priority for priority in scaled_priorities]
        indices = random.choices(
            range(len(self._items)), weights=probabilities, k=batch_size
        )
        minimum_probability = min(probabilities)
        maximum_weight = (len(self._items) * minimum_probability) ** (-beta)
        weights = [
            (len(self._items) * probabilities[index]) ** (-beta) / maximum_weight
            for index in indices
        ]
        items = tuple(self._items)
        return [items[index] for index in indices], indices, weights

    def update_priorities(
        self, indices: Sequence[int], priorities: Sequence[float]
    ) -> None:
        # Update sampled transitions using absolute TD errors plus a small floor.
        if len(indices) != len(priorities):
            raise ValueError("Indices and priorities must have matching lengths")
        priority_list = list(self._priorities)
        for index, priority in zip(indices, priorities):
            if not 0 <= index < len(priority_list):
                raise IndexError(f"Replay index out of range: {index}")
            if not math.isfinite(priority) or priority < 0.0:
                raise ValueError("Priorities must be finite and non-negative")
            priority_list[index] = priority + self.priority_epsilon
        self._priorities = deque(priority_list, maxlen=self.capacity)

    def __len__(self) -> int:
        # Report the number of stored transitions.
        return len(self._items)


class QNetwork(nn.Module):
    def __init__(self, observation_size: int, action_size: int) -> None:
        # Build a compact MLP mapping observations to one Q-value per action.
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(observation_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Return action values for each observation in the batch.
        return self.layers(observations)


class DoubleDQNAgent:
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        learning_rate: float = 1e-3,
        discount_factor: float = 0.99,
        replay_capacity: int = 10_000,
        batch_size: int = 64,
        target_update_interval: int = 100,
        device: torch.device | None = None,
    ) -> None:
        # Validate dimensions and establish the online/target learner pair.
        if observation_size < 1 or action_size < 1:
            raise ValueError("Observation and action sizes must be positive")
        if batch_size < 1 or target_update_interval < 1:
            raise ValueError("Batch size and target update interval must be positive")

        self.action_size = action_size
        self.observation_size = observation_size
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.online_network = QNetwork(observation_size, action_size).to(self.device)
        self.target_network = QNetwork(observation_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()
        self.optimizer = Adam(self.online_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(replay_capacity)
        self._updates = 0

    def select_action(
        self, observation: Sequence[float], epsilon: float = 0.0
    ) -> int:
        # Explore randomly or choose the highest-valued action greedily.
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between zero and one")
        if random.random() < epsilon:
            return random.randrange(self.action_size)

        state = torch.tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            return int(self.online_network(state).argmax(dim=1).item())

    def remember(
        self,
        state: list[float],
        action: int,
        reward: float,
        next_state: list[float],
        done: bool,
    ) -> None:
        # Store a transition for later prioritized replay.
        self.replay_buffer.add(Experience(state, action, reward, next_state, done))

    def train_step(self, beta: float = 0.4) -> float | None:
        # Apply a weighted Double-DQN update and refresh sampled priorities.
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch, indices, importance_weights = self.replay_buffer.sample(
            self.batch_size, beta
        )
        states = torch.tensor(
            [item.state for item in batch], dtype=torch.float32, device=self.device
        )
        actions = torch.tensor(
            [item.action for item in batch], dtype=torch.int64, device=self.device
        ).unsqueeze(1)
        rewards = torch.tensor(
            [item.reward for item in batch], dtype=torch.float32, device=self.device
        )
        next_states = torch.tensor(
            [item.next_state for item in batch],
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.tensor(
            [item.done for item in batch], dtype=torch.float32, device=self.device
        )
        current_q_values = self.online_network(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            # Double DQN separates action selection from action evaluation:
            # a* = argmax_a Q_online(s', a), then evaluate Q_target(s', a*).
            next_actions = self.online_network(next_states).argmax(
                dim=1, keepdim=True
            )
            next_q_values = self.target_network(next_states).gather(
                1, next_actions
            ).squeeze(1)
            # Terminal transitions have no future return to bootstrap from.
            targets = rewards + self.discount_factor * next_q_values * (1.0 - dones)

        td_errors = targets - current_q_values
        elementwise_loss = nn.functional.smooth_l1_loss(
            current_q_values, targets, reduction="none"
        )
        weights = torch.tensor(
            importance_weights, dtype=torch.float32, device=self.device
        )
        loss = (elementwise_loss * weights).mean()
        self.replay_buffer.update_priorities(
            indices, td_errors.detach().abs().cpu().tolist()
        )
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        self._updates += 1
        if self._updates % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())
        return float(loss.item())

    def save(self, path: Path) -> None:
        # Persist model dimensions and optimizer state for safe run continuation.
        torch.save(
            {
                "format_version": 1,
                "observation_size": self.observation_size,
                "action_size": self.action_size,
                "model_state_dict": self.online_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "updates": self._updates,
            },
            path,
        )

    def load(self, path: Path) -> None:
        # Load versioned checkpoints and continue accepting legacy state dictionaries.
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        if "model_state_dict" in state_dict:
            if (
                state_dict.get("observation_size") != self.observation_size
                or state_dict.get("action_size") != self.action_size
            ):
                raise ValueError(
                    "Checkpoint observation/action sizes do not match this environment"
                )
            model_state = state_dict["model_state_dict"]
            self.online_network.load_state_dict(model_state)
            self.target_network.load_state_dict(model_state)
            if "optimizer_state_dict" in state_dict:
                self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
            self._updates = int(state_dict.get("updates", 0))
        else:
            self.online_network.load_state_dict(state_dict)
            self.target_network.load_state_dict(state_dict)
