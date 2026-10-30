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


class KalmanUCBAgent(BanditAgent):
    """
    Kalman-UCB Agent (Dynamic Gaussian State-Space Bandit).
    Memodelkan nilai harapan reward setiap arm sebagai keadaan tersembunyi (latent state)
    yang bergerak acak kontinu: mu_k(t+1) = mu_k(t) + w_t, w_t ~ N(0, sigma_w^2).

    Karakteristik Kunci:
    - Predict: Varians ketidakpastian semua arm bertambah di setiap langkah:
               sigma_k^2(t+1|t) = sigma_k^2(t) + sigma_w^2.
    - Select: Memilih arm berdasarkan Upper Confidence Bound:
              A_t = argmax_k [ mu_hat_k + c * sigma_k ].
    - Update: Arm yang ditarik diperbarui menggunakan Kalman Gain:
              K = prior_var / (prior_var + sigma_v^2).
    """

    def __init__(
        self,
        n_arms: int,
        process_noise_std: float = 0.02,
        observation_noise_std: float = 0.15,
        c: float = 1.5,
        initial_mean: float = 0.5,
        initial_std: float = 0.5,
        name: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        label = name or f"Kalman-UCB (c={c}, σ_w={process_noise_std})"
        super().__init__(n_arms=n_arms, name=label, seed=seed)
        self.process_noise_std = process_noise_std
        self.observation_noise_std = observation_noise_std
        self.sigma_w2 = float(process_noise_std ** 2)
        self.sigma_v2 = float(observation_noise_std ** 2)
        self.c = c
        self.initial_mean = initial_mean
        self.initial_std = initial_std
        self.means = np.full(n_arms, initial_mean, dtype=float)
        self.variances = np.full(n_arms, initial_std ** 2, dtype=float)

    def select_arm(self) -> int:
        # Predict step: Ketidakpastian SEMUA arm meningkat seiring berjalannya waktu
        self.variances += self.sigma_w2

        # Hitung UCB = mean + c * std
        stds = np.sqrt(np.maximum(1e-8, self.variances))
        ucb_values = self.means + self.c * stds
        return self._argmax_random_tiebreak(ucb_values)

    def update(self, arm: int, reward: float) -> None:
        super().update(arm, reward)

        # Update step: Kalman filter correction pada arm yang diobservasi
        prior_var = self.variances[arm]
        kalman_gain = prior_var / (prior_var + self.sigma_v2)

        # Perbarui estimasi nilai harapan
        self.means[arm] += kalman_gain * (reward - self.means[arm])

        # Perbarui estimasi varians ketidakpastian
        self.variances[arm] = (1.0 - kalman_gain) * prior_var

    def reset(self) -> None:
        super().reset()
        self.means = np.full(self.n_arms, self.initial_mean, dtype=float)
        self.variances = np.full(self.n_arms, self.initial_std ** 2, dtype=float)

