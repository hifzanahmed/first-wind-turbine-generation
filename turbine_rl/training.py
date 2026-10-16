"""Environment-agnostic DDQN training and evaluation loops."""

from __future__ import annotations

from dataclasses import dataclass

from turbine_rl.agent import DoubleDQNAgent
from turbine_rl.environment import Environment


@dataclass(frozen=True)
class EpisodeMetrics:
    episode: int
    reward: float
    mean_loss: float
    epsilon: float
    validation_reward: float | None


def train(
    environment: Environment,
    agent: DoubleDQNAgent,
    episodes: int,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
    validation_environment: Environment | None = None,
    validation_interval: int = 1,
    beta_start: float = 0.4,
    beta_end: float = 1.0,
    early_stopping_patience: int | None = None,
) -> list[EpisodeMetrics]:
    # Validate schedules before training and return structured metrics for logging.
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if not 0.0 <= epsilon_end <= epsilon_start <= 1.0:
        raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")
    if not 0.0 < epsilon_decay <= 1.0:
        raise ValueError("epsilon_decay must be in (0, 1]")
    if validation_interval < 1:
        raise ValueError("validation_interval must be positive")
    if not 0.0 <= beta_start <= beta_end <= 1.0:
        raise ValueError("PER beta values must satisfy 0 <= start <= end <= 1")
    if early_stopping_patience is not None and early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be positive")

    epsilon = epsilon_start
    history: list[EpisodeMetrics] = []
    best_validation_reward = float("-inf")
    episodes_without_improvement = 0
    for episode in range(1, episodes + 1):
        # Train one complete pass through the chronological training segment.
        state = environment.reset()
        episode_reward = 0.0
        losses: list[float] = []
        done = False
        beta = beta_start + (beta_end - beta_start) * (
            (episode - 1) / max(episodes - 1, 1)
        )

        while not done:
            # Record transitions and anneal PER correction while updating DDQN.
            action = agent.select_action(state, epsilon)
            next_state, reward, done = environment.step(action)
            agent.remember(state, action, reward, next_state, done)
            loss = agent.train_step(beta=beta)
            if loss is not None:
                losses.append(loss)
            state = next_state
            episode_reward += reward

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        mean_loss = sum(losses) / len(losses) if losses else 0.0
        validation_reward = None
        if (
            validation_environment is not None
            and episode % validation_interval == 0
        ):
            # Evaluate greedily on the held-out time segment without exploration.
            validation_reward = evaluate(validation_environment, agent)
            if validation_reward > best_validation_reward:
                best_validation_reward = validation_reward
                episodes_without_improvement = 0
            else:
                episodes_without_improvement += validation_interval
        history.append(
            EpisodeMetrics(
                episode=episode,
                reward=episode_reward,
                mean_loss=mean_loss,
                epsilon=epsilon,
                validation_reward=validation_reward,
            )
        )
        print(
            f"Episode {episode:4d}/{episodes} | reward={episode_reward:.3f} | "
            f"epsilon={epsilon:.3f} | loss={mean_loss:.5f}"
            + (
                f" | validation={validation_reward:.3f}"
                if validation_reward is not None
                else ""
            )
        )
        if (
            early_stopping_patience is not None
            and validation_reward is not None
            and episodes_without_improvement >= early_stopping_patience
        ):
            # Stop once held-out reward has plateaued for the configured patience.
            break

    # Return episode results to the caller for checkpointing and JSON export.
    return history


def evaluate(environment: Environment, agent: DoubleDQNAgent) -> float:
    # Measure a greedy episode without adding transitions or changing exploration.
    state = environment.reset()
    total_reward = 0.0
    done = False
    while not done:
        action = agent.select_action(state)
        state, reward, done = environment.step(action)
        total_reward += reward
    return total_reward
