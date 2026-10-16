import random
import tempfile
import unittest
from pathlib import Path

from turbine_rl.agent import DoubleDQNAgent, Experience, ReplayBuffer


class PrioritizedReplayTests(unittest.TestCase):
    def test_sampling_tracks_priority_and_returns_importance_weights(self) -> None:
        # A high-priority transition should be sampled more often with valid IS weights.
        random.seed(11)
        buffer = ReplayBuffer(capacity=2, priority_alpha=1.0)
        buffer.add(Experience([0.0], 0, 0.0, [1.0], False))
        buffer.add(Experience([1.0], 1, 0.0, [2.0], False))
        buffer.update_priorities([0, 1], [0.01, 100.0])

        batch, indices, weights = buffer.sample(1000, beta=0.7)

        self.assertEqual(len(batch), 1000)
        self.assertGreater(indices.count(1), 990)
        self.assertTrue(all(0.0 < weight <= 1.0 for weight in weights))

    def test_training_updates_priorities_and_checkpoint_round_trips(self) -> None:
        # Exercise weighted DDQN learning and retain model/optimizer checkpoint state.
        agent = DoubleDQNAgent(
            observation_size=2,
            action_size=3,
            batch_size=2,
            replay_capacity=4,
            target_update_interval=1,
        )
        agent.remember([0.0, 0.0], 0, 1.0, [0.1, 0.0], False)
        agent.remember([0.2, 0.0], 1, -0.5, [0.3, 0.0], True)
        agent.remember([0.4, 0.0], 2, 0.25, [0.5, 0.0], False)

        loss = agent.train_step(beta=1.0)

        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0.0)
        self.assertEqual(agent._updates, 1)
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "agent.pt"
            agent.save(checkpoint)
            restored = DoubleDQNAgent(
                observation_size=2,
                action_size=3,
                batch_size=2,
                replay_capacity=4,
                target_update_interval=1,
            )
            restored.load(checkpoint)

        self.assertEqual(restored._updates, agent._updates)
        self.assertEqual(
            restored.select_action([0.2, 0.1]),
            agent.select_action([0.2, 0.1]),
        )

    def test_checkpoint_dimensions_are_validated(self) -> None:
        # Prevent loading a checkpoint whose network shape belongs to another task.
        agent = DoubleDQNAgent(observation_size=2, action_size=3)
        incompatible = DoubleDQNAgent(observation_size=2, action_size=4)
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "agent.pt"
            agent.save(checkpoint)
            with self.assertRaisesRegex(ValueError, "sizes do not match"):
                incompatible.load(checkpoint)


if __name__ == "__main__":
    # Allow running these tests directly as well as through discovery.
    unittest.main()
