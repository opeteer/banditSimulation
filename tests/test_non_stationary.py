"""
Unit tests untuk Environment Non-Stationer dan Agen Adaptif
"""

import unittest
import numpy as np
from bandit.non_stationary_env import DynamicNonStationaryBandit
from bandit.non_stationary_agents import (
    DiscountedThompsonSamplingAgent,
    SlidingWindowThompsonSamplingAgent,
    DiscountedUCBAgent,
    SlidingWindowUCBAgent,
    KalmanUCBAgent,
)


class TestNonStationaryEnvironment(unittest.TestCase):
    def test_abrupt_mode(self):
        phases = [
            {"step_until": 10, "means": [0.2, 0.8]},
            {"step_until": None, "means": [0.9, 0.1]},
        ]
        env = DynamicNonStationaryBandit(n_arms=2, mode="abrupt", phases=phases, seed=42)
        self.assertEqual(env.get_optimal_arm(), 1)
        for _ in range(10):
            env.step(1)
        self.assertEqual(env.get_optimal_arm(), 0)

    def test_drift_mode(self):
        env = DynamicNonStationaryBandit(
            n_arms=3, mode="drift", initial_means=[0.5, 0.5, 0.5], drift_std=0.05, seed=42
        )
        initial_means = env.get_expected_rewards().copy()
        for _ in range(50):
            env.step(0)
        new_means = env.get_expected_rewards()
        self.assertFalse(np.allclose(initial_means, new_means))
        self.assertTrue(np.all(new_means >= 0.05) and np.all(new_means <= 0.95))

    def test_seasonal_mode(self):
        params = [
            {"amplitude": 0.3, "period": 20.0, "phase": 0.0},
            {"amplitude": 0.3, "period": 20.0, "phase": np.pi},
        ]
        env = DynamicNonStationaryBandit(
            n_arms=2, mode="seasonal", initial_means=[0.5, 0.5], seasonal_params=params, seed=42
        )
        m0 = env.get_expected_rewards()
        self.assertAlmostEqual(m0[0], 0.5, delta=0.05)
        # Pada t = 5 (seperempat periode 20), sin(pi/2) = 1 -> naik ke 0.8
        for _ in range(5):
            env.step(0)
        m5 = env.get_expected_rewards()
        self.assertGreater(m5[0], 0.7)
        self.assertLess(m5[1], 0.3)

    def test_fatigue_mode(self):
        env = DynamicNonStationaryBandit(
            n_arms=2,
            mode="fatigue",
            initial_means=[0.8, 0.8],
            fatigue_params={"decay": 0.1, "recovery": 0.05},
            seed=42,
        )
        # Tarik arm 0 lima kali berturut-turut
        for _ in range(5):
            env.step(0)
        means = env.get_expected_rewards()
        self.assertLess(means[0], 0.5)
        self.assertEqual(means[1], 0.8)


class TestNonStationaryAgents(unittest.TestCase):
    def test_adaptive_agents_run(self):
        agents = [
            DiscountedThompsonSamplingAgent(n_arms=3, gamma=0.9, seed=42),
            SlidingWindowThompsonSamplingAgent(n_arms=3, window_size=20, seed=42),
            DiscountedUCBAgent(n_arms=3, gamma=0.9, seed=42),
            SlidingWindowUCBAgent(n_arms=3, window_size=20, seed=42),
            KalmanUCBAgent(n_arms=3, c=1.5, process_noise_std=0.02, seed=42),
        ]
        for agent in agents:
            for _ in range(30):
                arm = agent.select_arm()
                self.assertTrue(0 <= arm < 3)
                agent.update(arm, reward=1.0)
            agent.reset()
            self.assertEqual(agent.total_pulls, 0)


if __name__ == "__main__":
    unittest.main()
