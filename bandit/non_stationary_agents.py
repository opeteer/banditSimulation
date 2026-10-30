"""
Modul Agen Multi-Armed Bandit untuk Lingkungan Non-Stationer
============================================================
Menyediakan algoritma adaptif yang dirancang khusus untuk mendeteksi
dan menyesuaikan diri terhadap perubahan lingkungan dinamis:
1. DiscountedThompsonSamplingAgent (D-TS)
2. SlidingWindowThompsonSamplingAgent (SW-TS)
3. DiscountedUCBAgent (D-UCB)
4. SlidingWindowUCBAgent (SW-UCB)
"""

from typing import Optional, List, Tuple
from collections import deque
import numpy as np
from .agents import BanditAgent


class DiscountedThompsonSamplingAgent(BanditAgent):
    """
    Discounted Thompson Sampling (D-TS).
    Menggunakan faktor diskonto gamma in (0, 1) untuk meluruhkan observasi masa lalu.
    Membuat agen memiliki memori adaptif (effective sample size ~ 1 / (1 - gamma)).
    """

    def __init__(
        self,
        n_arms: int,
        gamma: float = 0.98,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        name: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        label = name or f"Discounted TS (γ={gamma})"
        super().__init__(n_arms=n_arms, name=label, seed=seed)
        self.gamma = gamma
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.alphas = np.full(n_arms, prior_alpha, dtype=float)
        self.betas = np.full(n_arms, prior_beta, dtype=float)

    def select_arm(self) -> int:
        # Sample dari distribusi Beta terkini
        samples = self.rng.beta(np.maximum(0.1, self.alphas), np.maximum(0.1, self.betas))
        return self._argmax_random_tiebreak(samples)

    def update(self, arm: int, reward: float) -> None:
        super().update(arm, reward)
        r = float(np.clip(reward, 0.0, 1.0))

        # Diskon semua arm mendekati prior awal
        self.alphas = self.gamma * self.alphas + (1.0 - self.gamma) * self.prior_alpha
        self.betas = self.gamma * self.betas + (1.0 - self.gamma) * self.prior_beta

        # Tambahkan observasi baru pada arm yang ditarik
        self.alphas[arm] += r
        self.betas[arm] += (1.0 - r)

    def reset(self) -> None:
        super().reset()
        self.alphas = np.full(self.n_arms, self.prior_alpha, dtype=float)
        self.betas = np.full(self.n_arms, self.prior_beta, dtype=float)


class SlidingWindowThompsonSamplingAgent(BanditAgent):
    """
    Sliding-Window Thompson Sampling (SW-TS).
    Hanya menggunakan jendela memori geser sepanjang W langkah terakhir.
    Observasi yang lebih tua dari W langkah dihapus dari perhitungan posterior.
    """

    def __init__(
        self,
        n_arms: int,
        window_size: int = 150,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        name: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        label = name or f"Sliding-Window TS (W={window_size})"
        super().__init__(n_arms=n_arms, name=label, seed=seed)
        self.window_size = window_size
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.window: deque[Tuple[int, float]] = deque(maxlen=window_size)

    def select_arm(self) -> int:
        alphas = np.full(self.n_arms, self.prior_alpha, dtype=float)
        betas = np.full(self.n_arms, self.prior_beta, dtype=float)

        for arm, rew in self.window:
            r = np.clip(rew, 0.0, 1.0)
            alphas[arm] += r
            betas[arm] += (1.0 - r)

        samples = self.rng.beta(alphas, betas)
        return self._argmax_random_tiebreak(samples)

    def update(self, arm: int, reward: float) -> None:
        super().update(arm, reward)
        self.window.append((arm, reward))

    def reset(self) -> None:
        super().reset()
        self.window.clear()


class DiscountedUCBAgent(BanditAgent):
    """
    Discounted UCB (D-UCB) (Garivier & Moulines, 2011).
    Menghitung rata-rata nilai dan jumlah penarikan yang didiskonto eksponensial dengan gamma.
    """

    def __init__(
        self,
        n_arms: int,
        gamma: float = 0.98,
        c: float = 1.0,
        name: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        label = name or f"Discounted UCB (γ={gamma})"
        super().__init__(n_arms=n_arms, name=label, seed=seed)
        self.gamma = gamma
        self.c = c
        self.discounted_counts = np.zeros(n_arms, dtype=float)
        self.discounted_rewards = np.zeros(n_arms, dtype=float)

    def select_arm(self) -> int:
        # Coba tiap arm yang belum pernah ditarik
        for arm in range(self.n_arms):
            if self.discounted_counts[arm] < 0.5:
                return arm

        total_discounted = np.sum(self.discounted_counts)
        mean_rewards = self.discounted_rewards / self.discounted_counts
        exploration = self.c * np.sqrt((2.0 * np.log(max(1.1, total_discounted))) / self.discounted_counts)
        ucb_values = mean_rewards + exploration
        return self._argmax_random_tiebreak(ucb_values)

    def update(self, arm: int, reward: float) -> None:
        super().update(arm, reward)
        # Luruhkan riwayat masa lalu
        self.discounted_counts *= self.gamma
        self.discounted_rewards *= self.gamma

        # Tambahkan observasi baru
        self.discounted_counts[arm] += 1.0
        self.discounted_rewards[arm] += reward

    def reset(self) -> None:
        super().reset()
        self.discounted_counts = np.zeros(self.n_arms, dtype=float)
        self.discounted_rewards = np.zeros(self.n_arms, dtype=float)


class SlidingWindowUCBAgent(BanditAgent):
    """
    Sliding-Window UCB (SW-UCB) (Garivier & Moulines, 2011).
    Hanya menghitung rata-rata reward dan frekuensi tarikan dalam W langkah terakhir.
    """

    def __init__(
        self,
        n_arms: int,
        window_size: int = 150,
        c: float = 1.0,
        name: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        label = name or f"Sliding-Window UCB (W={window_size})"
        super().__init__(n_arms=n_arms, name=label, seed=seed)
        self.window_size = window_size
        self.c = c
        self.window: deque[Tuple[int, float]] = deque(maxlen=window_size)

    def select_arm(self) -> int:
        # Hitung counts dan sum_rewards dalam window
        counts = np.zeros(self.n_arms, dtype=float)
        sums = np.zeros(self.n_arms, dtype=float)

        for arm, rew in self.window:
            counts[arm] += 1.0
            sums[arm] += rew

        # Prioritaskan arm yang belum ada di window
        for arm in range(self.n_arms):
            if counts[arm] == 0:
                return arm

        curr_window_len = len(self.window)
        means = sums / counts
        exploration = self.c * np.sqrt((2.0 * np.log(curr_window_len)) / counts)
        ucb_values = means + exploration
        return self._argmax_random_tiebreak(ucb_values)

    def update(self, arm: int, reward: float) -> None:
        super().update(arm, reward)
        self.window.append((arm, reward))

    def reset(self) -> None:
        super().reset()
        self.window.clear()
