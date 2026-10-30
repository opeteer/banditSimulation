"""
Modul Algoritma Agent Multi-Armed Bandit
========================================
Berisi implementasi berbagai algoritma bandit:
1. RandomAgent (Baseline acak)
2. EpsilonGreedyAgent (Epsilon tetap)
3. DecayingEpsilonAgent (Epsilon meluruh seiring waktu)
4. UCB1Agent (Upper Confidence Bound)
5. ThompsonSamplingAgent (Bayesian posterior sampling - Beta-Bernoulli)
6. OptimisticGreedyAgent (Greedy dengan inisialisasi optimistik)
"""

from abc import ABC, abstractmethod
from typing import Optional, Union
import numpy as np


class BanditAgent(ABC):
    """Kelas abstrak dasar untuk agen Multi-Armed Bandit."""

    def __init__(self, n_arms: int, name: str, seed: Optional[int] = None):
        self.n_arms = n_arms
        self.name = name
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.counts = np.zeros(n_arms, dtype=int)
        self.values = np.zeros(n_arms, dtype=float)
        self.total_pulls = 0

    @abstractmethod
    def select_arm(self) -> int:
        """Memilih tuas (arm) yang akan ditarik pada langkah berikutnya."""
        pass

    def update(self, arm: int, reward: float) -> None:
        """Memperbarui estimasi nilai (Q-value) dan counter penarikan."""
        self.counts[arm] += 1
        self.total_pulls += 1
        # Pembaruan nilai rata-rata secara inkremental: Q_new = Q_old + (R - Q_old) / N
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]

    def reset(self) -> None:
        """Mereset state agen ke kondisi awal."""
        self.rng = np.random.default_rng(self.seed)
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.values = np.zeros(self.n_arms, dtype=float)
        self.total_pulls = 0

    def _argmax_random_tiebreak(self, array: np.ndarray) -> int:
        """Helper untuk memilih indeks maksimum dengan pemecah dasi (tie-breaking) acak."""
        max_val = np.max(array)
        candidates = np.flatnonzero(array == max_val)
        if len(candidates) == 1:
            return int(candidates[0])
        return int(self.rng.choice(candidates))


class RandomAgent(BanditAgent):
    """Agen baseline yang memilih tuas secara acak seragam (pure random)."""

    def __init__(self, n_arms: int, name: str = "Random", seed: Optional[int] = None):
        super().__init__(n_arms=n_arms, name=name, seed=seed)

    def select_arm(self) -> int:
        return int(self.rng.integers(0, self.n_arms))


class EpsilonGreedyAgent(BanditAgent):
    """
    Algoritma Epsilon-Greedy.
    Dengan probabilitas epsilon, eksplorasi tuas acak.
    Dengan probabilitas (1 - epsilon), eksploitasi tuas dengan estimasi nilai tertinggi.
    """

    def __init__(
        self,
        n_arms: int,
        epsilon: float = 0.1,
        name: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        label = name or f"ε-Greedy (ε={epsilon})"
        super().__init__(n_arms=n_arms, name=label, seed=seed)
        self.epsilon = epsilon

    def select_arm(self) -> int:
        if self.rng.random() < self.epsilon:
            # Eksplorasi acak
            return int(self.rng.integers(0, self.n_arms))
        # Eksploitasi nilai Q tertinggi
        return self._argmax_random_tiebreak(self.values)


class DecayingEpsilonAgent(BanditAgent):
    """
    Algoritma Decaying Epsilon-Greedy.
    Epsilon berkurang secara bertahap seiring berjalannya langkah (t):
    epsilon_t = max(min_epsilon, initial_epsilon / (1 + decay_rate * t))
    """

    def __init__(
        self,
        n_arms: int,
        initial_epsilon: float = 1.0,
        decay_rate: float = 0.005,
        min_epsilon: float = 0.01,
        name: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        label = name or f"Decaying ε (ε0={initial_epsilon}, d={decay_rate})"
        super().__init__(n_arms=n_arms, name=label, seed=seed)
        self.initial_epsilon = initial_epsilon
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon

    def get_current_epsilon(self) -> float:
        eps = self.initial_epsilon / (1.0 + self.decay_rate * self.total_pulls)
        return float(max(self.min_epsilon, eps))

    def select_arm(self) -> int:
        current_eps = self.get_current_epsilon()
        if self.rng.random() < current_eps:
            return int(self.rng.integers(0, self.n_arms))
        return self._argmax_random_tiebreak(self.values)


class UCB1Agent(BanditAgent):
    """
    Algoritma Upper Confidence Bound (UCB1).
    Memilih arm berdasarkan nilai estimasi ditambah batas ketidakpastian:
    a_t = argmax_k [ Q_k + c * sqrt( (2 * ln(t)) / N_k ) ]
    """

    def __init__(
        self,
        n_arms: int,
        c: float = 1.0,
        name: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        label = name or f"UCB1 (c={c})"
        super().__init__(n_arms=n_arms, name=label, seed=seed)
        self.c = c

    def select_arm(self) -> int:
        # Tarik setiap arm minimal satu kali untuk inisialisasi kepastian
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm

        # Hitung UCB untuk setiap arm
        total_steps = self.total_pulls
        exploration_term = np.sqrt((2.0 * np.log(total_steps)) / self.counts)
        ucb_values = self.values + (self.c * exploration_term)
        return self._argmax_random_tiebreak(ucb_values)


class ThompsonSamplingAgent(BanditAgent):
    """
    Algoritma Bayesian Thompson Sampling (Posterior Sampling).
    Menggunakan distribusi prior Beta(alpha, beta) untuk reward biner Bernoulli.
    Setiap langkah, sampel nilai theta_k ~ Beta(alpha_k, beta_k),
    lalu pilih arm dengan sampel theta tertinggi.
    """

    def __init__(
        self,
        n_arms: int,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        name: str = "Thompson Sampling",
        seed: Optional[int] = None,
    ):
        super().__init__(n_arms=n_arms, name=name, seed=seed)
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.alphas = np.full(n_arms, prior_alpha, dtype=float)
        self.betas = np.full(n_arms, prior_beta, dtype=float)

    def select_arm(self) -> int:
        # Sample theta dari distribusi Beta setiap arm
        samples = self.rng.beta(self.alphas, self.betas)
        return self._argmax_random_tiebreak(samples)

    def update(self, arm: int, reward: float) -> None:
        super().update(arm, reward)
        # Update parameter Beta posterior
        # Jika reward biner: sukses (reward=1) -> alpha + 1, gagal (reward=0) -> beta + 1
        r_clipped = np.clip(reward, 0.0, 1.0)
        self.alphas[arm] += r_clipped
        self.betas[arm] += (1.0 - r_clipped)

    def reset(self) -> None:
        super().reset()
        self.alphas = np.full(self.n_arms, self.prior_alpha, dtype=float)
        self.betas = np.full(self.n_arms, self.prior_beta, dtype=float)


class GammaPoissonThompsonAgent(BanditAgent):
    """
    Algoritma Bayesian Thompson Sampling model Gamma-Poisson.
    Sangat cocok untuk data reward diskret cacah Poisson (misal piring makan, jumlah order).
    Prior: Lambda_k ~ Gamma(alpha, beta).
    Setiap langkah, sampel lambda_k ~ Gamma(alpha_k, scale = 1 / beta_k).
    Pilih arm dengan sampel tertinggi.
    Setelah reward r teramati:
        alpha_k <- alpha_k + r
        beta_k <- beta_k + 1
    """

    def __init__(
        self,
        n_arms: int,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        name: str = "Gamma-Poisson Thompson Sampling",
        seed: Optional[int] = None,
    ):
        super().__init__(n_arms=n_arms, name=name, seed=seed)
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.alphas = np.full(n_arms, prior_alpha, dtype=float)
        self.betas = np.full(n_arms, prior_beta, dtype=float)

    def select_arm(self) -> int:
        # Sample lambda dari distribusi Gamma (shape=alpha, scale=1/beta)
        samples = self.rng.gamma(shape=self.alphas, scale=1.0 / self.betas)
        return self._argmax_random_tiebreak(samples)

    def update(self, arm: int, reward: float) -> None:
        super().update(arm, reward)
        self.alphas[arm] += reward
        self.betas[arm] += 1.0

    def reset(self) -> None:
        super().reset()
        self.alphas = np.full(self.n_arms, self.prior_alpha, dtype=float)
        self.betas = np.full(self.n_arms, self.prior_beta, dtype=float)


class PoissonUCBAgent(BanditAgent):
    """
    Algoritma UCB khusus data Poisson.
    Menggunakan estimasi variansi Poisson Var(X) = lambda = mean:
    UCB_k = Q_k + c * sqrt( (2 * Q_k * ln(t)) / N_k )
    """

    def __init__(
        self,
        n_arms: int,
        c: float = 1.0,
        name: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        label = name or f"Poisson-UCB (c={c})"
        super().__init__(n_arms=n_arms, name=label, seed=seed)
        self.c = c

    def select_arm(self) -> int:
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
        total_steps = self.total_pulls
        mean_rewards = np.maximum(self.values, 0.1)
        exploration_term = np.sqrt((2.0 * mean_rewards * np.log(total_steps)) / self.counts)
        ucb_values = self.values + (self.c * exploration_term)
        return self._argmax_random_tiebreak(ucb_values)


class OptimisticGreedyAgent(BanditAgent):
    """
    Greedy dengan Inisialisasi Optimistik (Optimistic Initial Values).
    Menginisialisasi nilai Q_0 jauh lebih tinggi daripada reward maksimal yang diharapkan,
    sehingga agen terdorong mengeksplorasi semua arm saat nilai aktualnya mengecewakan.
    """

    def __init__(
        self,
        n_arms: int,
        initial_value: float = 1.0,
        name: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        label = name or f"Optimistic Greedy (Q0={initial_value})"
        super().__init__(n_arms=n_arms, name=label, seed=seed)
        self.initial_value = initial_value
        self.values = np.full(n_arms, initial_value, dtype=float)

    def select_arm(self) -> int:
        return self._argmax_random_tiebreak(self.values)

    def reset(self) -> None:
        super().reset()
        self.values = np.full(self.n_arms, self.initial_value, dtype=float)
