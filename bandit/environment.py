"""
Modul Environment Multi-Armed Bandit
====================================
Menyediakan kelas environment untuk berbagai skenario Multi-Armed Bandit:
- BernoulliBandit (Binary rewards: 0 atau 1, cocok untuk CTR / Conversion)
- GaussianBandit (Continuous rewards: pendapatan / metrik kontinu)
- NonStationaryBandit (Reward distribution berubah seiring waktu)
"""

from abc import ABC, abstractmethod
from typing import List, Sequence, Union
import numpy as np


class BanditEnvironment(ABC):
    """Kelas abstrak dasar untuk environment Multi-Armed Bandit."""

    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.current_step = 0

    @abstractmethod
    def step(self, arm: int) -> float:
        """Menarik tuas (arm) tertentu dan mengembalikan reward."""
        pass

    @abstractmethod
    def get_expected_rewards(self) -> np.ndarray:
        """Mengembalikan nilai harapan (mean) reward aktual untuk setiap arm."""
        pass

    def get_optimal_arm(self) -> int:
        """Mengembalikan indeks arm dengan nilai harapan reward tertinggi."""
        return int(np.argmax(self.get_expected_rewards()))

    def get_optimal_reward(self) -> float:
        """Mengembalikan nilai harapan reward optimal (mu*)."""
        return float(np.max(self.get_expected_rewards()))

    def reset(self) -> None:
        """Mereset kondisi internal environment ke awal."""
        self.current_step = 0


class BernoulliBandit(BanditEnvironment):
    """
    Bernoulli Multi-Armed Bandit.
    Setiap arm k menghasilkan reward 1 dengan probabilitas p_k dan 0 sebaliknya.
    Sangat representatif untuk pemodelan Click-Through Rate (CTR) iklan
    atau tingkat konversi web (Conversion Rate).
    """

    def __init__(self, probabilities: Sequence[float], seed: Union[int, None] = None):
        super().__init__(n_arms=len(probabilities))
        self.probabilities = np.array(probabilities, dtype=float)
        if np.any(self.probabilities < 0.0) or np.any(self.probabilities > 1.0):
            raise ValueError("Semua probabilitas Bernoulli harus bernilai antara 0.0 dan 1.0.")
        self.rng = np.random.default_rng(seed)

    def step(self, arm: int) -> float:
        if not (0 <= arm < self.n_arms):
            raise IndexError(f"Indeks arm {arm} di luar jangkauan (0 sampai {self.n_arms - 1}).")
        self.current_step += 1
        p = self.probabilities[arm]
        # Mengembalikan 1.0 jika random float < probabilitas, else 0.0
        return float(self.rng.random() < p)

    def get_expected_rewards(self) -> np.ndarray:
        return self.probabilities.copy()

    def reset(self) -> None:
        super().reset()


class ContinuousBetaBandit(BanditEnvironment):
    """
    Continuous Beta Multi-Armed Bandit.
    Setiap arm k menghasilkan reward kontinu r in [0, 1] yang disampel dari
    distribusi Beta(alpha_k, beta_k) dengan nilai harapan (mean) mu_k.
    Sangat representatif untuk skor kepuasan / rating pelanggan kontinu (0 = buruk, 1 = sempurna).
    """

    def __init__(
        self,
        means: Sequence[float],
        precision: float = 20.0,
        seed: Union[int, None] = None,
    ):
        super().__init__(n_arms=len(means))
        self.means = np.array(means, dtype=float)
        if np.any(self.means <= 0.0) or np.any(self.means >= 1.0):
            raise ValueError("Semua mean harus berada dalam rentang (0.0, 1.0).")
        self.precision = precision
        # alpha = mu * precision, beta = (1 - mu) * precision
        self.alphas = self.means * precision
        self.betas = (1.0 - self.means) * precision
        self.rng = np.random.default_rng(seed)

    def step(self, arm: int) -> float:
        if not (0 <= arm < self.n_arms):
            raise IndexError(f"Indeks arm {arm} di luar jangkauan (0 sampai {self.n_arms - 1}).")
        self.current_step += 1
        return float(self.rng.beta(self.alphas[arm], self.betas[arm]))

    def get_expected_rewards(self) -> np.ndarray:
        return self.means.copy()

    def reset(self) -> None:
        super().reset()


class CappedPoissonBandit(BanditEnvironment):
    """
    Capped Poisson Multi-Armed Bandit.
    Setiap arm k menghasilkan reward diskret non-negatif X ~ Poisson(lambda_k),
    namun dibatasi oleh kapasitas fisik maksimal (max_cap, misal 7 piring).
    Reward riil r = min(max_cap, X).
    """

    def __init__(
        self,
        lambdas: Sequence[float],
        max_cap: int = 7,
        seed: Union[int, None] = None,
    ):
        super().__init__(n_arms=len(lambdas))
        self.lambdas = np.array(lambdas, dtype=float)
        if np.any(self.lambdas <= 0.0):
            raise ValueError("Semua nilai lambda Poisson harus bernilai positif.")
        self.max_cap = int(max_cap)
        self.rng = np.random.default_rng(seed)

        # Hitung nilai harapan eksak E[min(max_cap, X)]
        self._expected_rewards = np.zeros(self.n_arms, dtype=float)
        from scipy.stats import poisson as sp_poisson
        k_range = np.arange(0, 100)
        for i, lam in enumerate(self.lambdas):
            probs = sp_poisson.pmf(k_range, lam)
            capped_vals = np.minimum(self.max_cap, k_range)
            self._expected_rewards[i] = float(np.sum(capped_vals * probs))

    def step(self, arm: int) -> float:
        if not (0 <= arm < self.n_arms):
            raise IndexError(f"Indeks arm {arm} di luar jangkauan (0 sampai {self.n_arms - 1}).")
        self.current_step += 1
        raw_x = self.rng.poisson(self.lambdas[arm])
        capped_x = min(self.max_cap, int(raw_x))
        return float(capped_x)

    def get_expected_rewards(self) -> np.ndarray:
        return self._expected_rewards.copy()

    def reset(self) -> None:
        super().reset()


class GaussianBandit(BanditEnvironment):
    """
    Gaussian Multi-Armed Bandit.
    Setiap arm k menghasilkan reward yang berdistribusi normal N(mu_k, sigma_k^2).
    Cocok untuk pemodelan pendapatan (revenue) rata-rata per transaksi pengguna.
    """

    def __init__(
        self,
        means: Sequence[float],
        stds: Union[float, Sequence[float]] = 1.0,
        seed: Union[int, None] = None,
    ):
        super().__init__(n_arms=len(means))
        self.means = np.array(means, dtype=float)
        if isinstance(stds, (int, float)):
            self.stds = np.full(self.n_arms, float(stds))
        else:
            self.stds = np.array(stds, dtype=float)
        if np.any(self.stds <= 0.0):
            raise ValueError("Standar deviasi harus bernilai positif.")
        self.rng = np.random.default_rng(seed)

    def step(self, arm: int) -> float:
        if not (0 <= arm < self.n_arms):
            raise IndexError(f"Indeks arm {arm} di luar jangkauan (0 sampai {self.n_arms - 1}).")
        self.current_step += 1
        mu = self.means[arm]
        sigma = self.stds[arm]
        return float(self.rng.normal(mu, sigma))

    def get_expected_rewards(self) -> np.ndarray:
        return self.means.copy()

    def reset(self) -> None:
        super().reset()


class NonStationaryBandit(BanditEnvironment):
    """
    Non-Stationary Multi-Armed Bandit.
    Probabilitas / mean reward arm dapat berubah pada waktu tertentu (misal: ad fatigue,
    perubahan tren pasar mendadak pada pertengahan kampanye).
    """

    def __init__(
        self,
        phases: List[dict],
        seed: Union[int, None] = None,
    ):
        """
        phases: List konfigurasi fase, contoh:
        [
            {"step_until": 1000, "probabilities": [0.05, 0.12, 0.08, 0.02]},
            {"step_until": None, "probabilities": [0.04, 0.02, 0.16, 0.03]}
        ]
        """
        if not phases:
            raise ValueError("Harus menyediakan minimal 1 konfigurasi fase.")
        self.phases = phases
        self.n_arms = len(phases[0]["probabilities"])
        super().__init__(n_arms=self.n_arms)
        self.rng = np.random.default_rng(seed)

    def _get_current_phase_probs(self) -> np.ndarray:
        for phase in self.phases:
            limit = phase.get("step_until")
            if limit is None or self.current_step < limit:
                return np.array(phase["probabilities"], dtype=float)
        return np.array(self.phases[-1]["probabilities"], dtype=float)

    def step(self, arm: int) -> float:
        if not (0 <= arm < self.n_arms):
            raise IndexError(f"Indeks arm {arm} di luar jangkauan (0 sampai {self.n_arms - 1}).")
        probs = self._get_current_phase_probs()
        p = probs[arm]
        self.current_step += 1
        return float(self.rng.random() < p)

    def get_expected_rewards(self) -> np.ndarray:
        return self._get_current_phase_probs()

    def reset(self) -> None:
        super().reset()
