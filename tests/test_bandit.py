"""
Unit tests untuk simulasi Multi-Armed Bandit
"""

import unittest
import numpy as np
from bandit.environment import BernoulliBandit, GaussianBandit, NonStationaryBandit
from bandit.agents import (
    RandomAgent,
    EpsilonGreedyAgent,
    DecayingEpsilonAgent,
    UCB1Agent,
    ThompsonSamplingAgent,
    OptimisticGreedyAgent,
)
from bandit.simulator import Simulator


class TestBanditEnvironments(unittest.TestCase):
    def test_bernoulli_bandit(self):
        probs = [0.1, 0.5, 0.9]
        env = BernoulliBandit(probs, seed=42)
        self.assertEqual(env.n_arms, 3)
        self.assertEqual(env.get_optimal_arm(), 2)
        self.assertAlmostEqual(env.get_optimal_reward(), 0.9)

        rewards = [env.step(2) for _ in range(100)]
        self.assertTrue(all(r in (0.0, 1.0) for r in rewards))
        self.assertGreater(sum(rewards), 70)  # High prob for arm 2

    def test_gaussian_bandit(self):
        means = [1.0, 5.0, 2.0]
        env = GaussianBandit(means, stds=0.1, seed=42)
        self.assertEqual(env.n_arms, 3)
        self.assertEqual(env.get_optimal_arm(), 1)
        self.assertAlmostEqual(env.get_optimal_reward(), 5.0)

        rewards = [env.step(1) for _ in range(50)]
        self.assertAlmostEqual(float(np.mean(rewards)), 5.0, delta=0.2)

    def test_non_stationary_bandit(self):
        phases = [
            {"step_until": 10, "probabilities": [0.8, 0.2]},
            {"step_until": None, "probabilities": [0.1, 0.9]},
        ]
        env = NonStationaryBandit(phases, seed=42)
        self.assertEqual(env.get_optimal_arm(), 0)
        for _ in range(10):
            env.step(0)
        self.assertEqual(env.get_optimal_arm(), 1)


class TestBanditAgents(unittest.TestCase):
    def test_agents_selection_and_update(self):
        n_arms = 4
        agents = [
            RandomAgent(n_arms, seed=42),
            EpsilonGreedyAgent(n_arms, epsilon=0.1, seed=42),
            DecayingEpsilonAgent(n_arms, seed=42),
            UCB1Agent(n_arms, seed=42),
            ThompsonSamplingAgent(n_arms, seed=42),
            OptimisticGreedyAgent(n_arms, initial_value=2.0, seed=42),
        ]

        for agent in agents:
            arm = agent.select_arm()
            self.assertTrue(0 <= arm < n_arms)
            agent.update(arm, reward=1.0)
            self.assertEqual(agent.counts[arm], 1)
            self.assertEqual(agent.total_pulls, 1)

            agent.reset()
            self.assertEqual(agent.total_pulls, 0)
            self.assertTrue(all(c == 0 for c in agent.counts))


class TestSimulationIntegration(unittest.TestCase):
    def test_simulation_run(self):
        probs = [0.1, 0.8]
        agents = [
            RandomAgent(len(probs), seed=42),
            ThompsonSamplingAgent(len(probs), seed=42),
        ]
        sim = Simulator(
            env_factory=lambda s: BernoulliBandit(probs, seed=s),
            agents=agents,
            case_name="Test Run",
        )
        res = sim.run(n_steps=50, n_runs=10, verbose=False)
        self.assertIn("Random", res.agent_results)
        self.assertIn("Thompson Sampling", res.agent_results)
        # Thompson Sampling should have lower regret than pure random
        random_regret = res.agent_results["Random"].final_mean_regret
        ts_regret = res.agent_results["Thompson Sampling"].final_mean_regret
        self.assertLess(ts_regret, random_regret)


if __name__ == "__main__":
    unittest.main()
