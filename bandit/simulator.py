"""
Modul Simulasi Monte Carlo Multi-Armed Bandit
=============================================
Menjalankan eksperimen multi-run (Monte Carlo) untuk mengevaluasi performa
berbagai algoritma bandit secara statistik akurat dan stabil.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
import numpy as np

from .agents import BanditAgent
from .environment import BanditEnvironment


@dataclass
class AgentResult:
    """Menyimpan metrik hasil simulasi untuk satu agen/algoritma."""
    name: str
    n_runs: int
    n_steps: int
    n_arms: int

    # Array 2D (n_runs, n_steps)
    rewards: np.ndarray
    regrets: np.ndarray
    optimal_choices: np.ndarray
    cumulative_rewards: np.ndarray
    cumulative_regrets: np.ndarray

    # Array 2D (n_runs, n_arms)
    arm_pulls: np.ndarray

    # Metrik agregat (rata-rata per step)
    @property
    def mean_rewards(self) -> np.ndarray:
        return np.mean(self.rewards, axis=0)

    @property
    def mean_cumulative_rewards(self) -> np.ndarray:
        return np.mean(self.cumulative_rewards, axis=0)

    @property
    def std_cumulative_rewards(self) -> np.ndarray:
        return np.std(self.cumulative_rewards, axis=0)

    @property
    def mean_cumulative_regrets(self) -> np.ndarray:
        return np.mean(self.cumulative_regrets, axis=0)

    @property
    def std_cumulative_regrets(self) -> np.ndarray:
        return np.std(self.cumulative_regrets, axis=0)

    @property
    def optimal_action_rate(self) -> np.ndarray:
        """Persentase pemilihan arm optimal di setiap langkah waktu (0 sampai 1)."""
        return np.mean(self.optimal_choices, axis=0)

    @property
    def mean_arm_pulls(self) -> np.ndarray:
        return np.mean(self.arm_pulls, axis=0)

    @property
    def final_mean_reward(self) -> float:
        return float(self.mean_cumulative_rewards[-1])

    @property
    def final_mean_regret(self) -> float:
        return float(self.mean_cumulative_regrets[-1])

    @property
    def final_optimal_rate(self) -> float:
        return float(self.optimal_action_rate[-1] * 100.0)


@dataclass
class SimulationResult:
    """Kumpulan hasil seluruh agen beserta metadata simulasi."""
    case_name: str
    n_steps: int
    n_runs: int
    expected_rewards: np.ndarray
    optimal_arm: int
    optimal_reward: float
    agent_results: Dict[str, AgentResult]


class Simulator:
    """
    Runner simulasi Monte Carlo untuk membandingkan performa agen bandit.
    """

    def __init__(
        self,
        env_factory: Callable[[int], BanditEnvironment],
        agents: List[BanditAgent],
        case_name: str = "Multi-Armed Bandit Simulation",
    ):
        """
        env_factory: Fungsi pembuat environment baru yang menerima run_seed: int
        agents: Daftar agen yang akan diuji
        case_name: Deskripsi kasus studi yang dijalankan
        """
        self.env_factory = env_factory
        self.agents = agents
        self.case_name = case_name

    def run(
        self,
        n_steps: int = 1500,
        n_runs: int = 100,
        seed: int = 42,
        verbose: bool = True,
    ) -> SimulationResult:
        """
        Menjalankan simulasi sebanyak n_runs iterasi Monte Carlo.
        """
        # Dapatkan referensi environment untuk metadata
        sample_env = self.env_factory(seed)
        n_arms = sample_env.n_arms
        expected_rewards = sample_env.get_expected_rewards()
        optimal_arm = sample_env.get_optimal_arm()
        optimal_reward = sample_env.get_optimal_reward()

        if verbose:
            print(f"============================================================")
            print(f"Memulai Simulasi: {self.case_name}")
            print(f"Jumlah Arm: {n_arms} | Steps per Run: {n_steps} | Total Runs: {n_runs}")
            print(f"Probabilitas/Nilai Harapan Arm: {np.round(expected_rewards, 4)}")
            print(f"Arm Optimal: Index {optimal_arm} (Nilai: {optimal_reward:.4f})")
            print(f"Jumlah Algoritma diuji: {len(self.agents)}")
            print(f"============================================================")

        agent_results: Dict[str, AgentResult] = {}
        rng = np.random.default_rng(seed)

        for agent in self.agents:
            if verbose:
                print(f"-> Mensimulasikan {agent.name}...", end="", flush=True)

            rewards = np.zeros((n_runs, n_steps), dtype=float)
            regrets = np.zeros((n_runs, n_steps), dtype=float)
            optimal_choices = np.zeros((n_runs, n_steps), dtype=float)
            arm_pulls = np.zeros((n_runs, n_arms), dtype=int)

            for run_idx in range(n_runs):
                run_seed = int(rng.integers(0, 1_000_000_000))
                env = self.env_factory(run_seed)
                agent.reset()
                # Set agent seed per run agar variasi terisolasi
                agent.rng = np.random.default_rng(run_seed + 777)

                for step_idx in range(n_steps):
                    # Dapatkan expected rewards saat ini (bisa dinamis untuk non-stationary)
                    current_exp_rewards = env.get_expected_rewards()
                    current_optimal_reward = float(np.max(current_exp_rewards))
                    current_optimal_arm = int(np.argmax(current_exp_rewards))

                    arm = agent.select_arm()
                    reward = env.step(arm)
                    agent.update(arm, reward)

                    # Expected regret = mu* - mu_arm
                    regret = current_optimal_reward - current_exp_rewards[arm]

                    rewards[run_idx, step_idx] = reward
                    regrets[run_idx, step_idx] = regret
                    optimal_choices[run_idx, step_idx] = 1.0 if (arm == current_optimal_arm) else 0.0

                arm_pulls[run_idx] = agent.counts.copy()

            cum_rewards = np.cumsum(rewards, axis=1)
            cum_regrets = np.cumsum(regrets, axis=1)

            res = AgentResult(
                name=agent.name,
                n_runs=n_runs,
                n_steps=n_steps,
                n_arms=n_arms,
                rewards=rewards,
                regrets=regrets,
                optimal_choices=optimal_choices,
                cumulative_rewards=cum_rewards,
                cumulative_regrets=cum_regrets,
                arm_pulls=arm_pulls,
            )
            agent_results[agent.name] = res
            if verbose:
                print(f" Selesai! (Reward Rata-rata: {res.final_mean_reward:.2f}, Regret: {res.final_mean_regret:.2f}, % Optimal: {res.final_optimal_rate:.1f}%)")

        return SimulationResult(
            case_name=self.case_name,
            n_steps=n_steps,
            n_runs=n_runs,
            expected_rewards=expected_rewards,
            optimal_arm=optimal_arm,
            optimal_reward=optimal_reward,
            agent_results=agent_results,
        )
