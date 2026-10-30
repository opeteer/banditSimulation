"""
Modul Environment Multi-Armed Bandit Non-Stationer Tingkat Lanjut
=================================================================
Menyediakan lingkungan dinamis di mana distribusi reward (rata-rata/kemungkinan)
berubah seiring berjalannya waktu (non-stationary).

Mendukung 4 mode perubahan dinamis:
1. 'abrupt'       : Pergeseran mendadak pada langkah/fase tertentu (Piecewise-Stationary)
2. 'drift'        : Pergeseran kontinu perlahan (Random Walk / Brownian Motion)
3. 'seasonal'     : Fluktuasi siklikal/musiman periodik (Sinusoidal Cycle)
4. 'fatigue'      : Kualitas menurun jika dieksploitasi terus-menerus dan pulih jika diistirahatkan
"""

from typing import List, Optional, Sequence, Union, Dict, Any
import numpy as np
from .environment import BanditEnvironment


class DynamicNonStationaryBandit(BanditEnvironment):
    """
    Environment Multi-Armed Bandit Non-Stationer Komprehensif.
    Memungkinkan reward berubah secara dinamis berdasarkan waktu (step) atau interaksi agen.
    """

    def __init__(
        self,
        n_arms: int,
        mode: str = "abrupt",
        reward_type: str = "beta",
        precision: float = 25.0,
        initial_means: Optional[Sequence[float]] = None,
        phases: Optional[List[Dict[str, Any]]] = None,
        drift_std: float = 0.015,
        seasonal_params: Optional[List[Dict[str, float]]] = None,
        fatigue_params: Optional[Dict[str, float]] = None,
        min_bound: float = 0.05,
        max_bound: float = 0.95,
        seed: Optional[int] = None,
    ):
        """
        Parameter:
            n_arms: Jumlah arm pilihan
            mode: 'abrupt', 'drift', 'seasonal', atau 'fatigue'
            reward_type: 'beta' (kontinu [0,1]), 'bernoulli' (0/1), atau 'gaussian'
            precision: Presisi untuk distribusi Beta (kappa)
            initial_means: Rata-rata nilai harapan awal masing-masing arm
            phases: Konfigurasi pergeseran bertahap untuk mode 'abrupt'
            drift_std: Simpangan baku drift acak per langkah untuk mode 'drift'
            seasonal_params: Konfigurasi amplitude, period, dan phase untuk mode 'seasonal'
            fatigue_params: {'decay': 0.05, 'recovery': 0.02} untuk mode 'fatigue'
        """
        super().__init__(n_arms=n_arms)
        self.mode = mode.lower()
        self.reward_type = reward_type.lower()
        self.precision = precision
        self.drift_std = drift_std
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Inisialisasi means awal
        if initial_means is not None:
            self.initial_means = np.array(initial_means, dtype=float)
        else:
            self.initial_means = np.linspace(0.3, 0.8, n_arms)

        self.current_means = self.initial_means.copy()
        self.phases = phases or []
        self.seasonal_params = seasonal_params or []
        self.fatigue_params = fatigue_params or {"decay": 0.06, "recovery": 0.02}

        # Track riwayat nilai harapan untuk visualisasi
        self.means_history: List[np.ndarray] = []

    def get_expected_rewards(self) -> np.ndarray:
        """Mengembalikan nilai harapan (mean) reward saat ini."""
        return self.current_means.copy()

    def _update_dynamics(self, pulled_arm: int) -> None:
        """Memperbarui nilai rata-rata arm berdasarkan mode non-stationer."""
        t = self.current_step

        if self.mode == "abrupt":
            # Periksa apakah ada fase yang cocok untuk langkah saat ini
            for phase in self.phases:
                limit = phase.get("step_until")
                if limit is None or t < limit:
                    self.current_means = np.array(phase["means"], dtype=float)
                    break

        elif self.mode == "drift":
            # Random walk perlahan dengan batasan min_bound dan max_bound
            perturbation = self.rng.normal(0, self.drift_std, size=self.n_arms)
            self.current_means = np.clip(
                self.current_means + perturbation,
                self.min_bound,
                self.max_bound,
            )

        elif self.mode == "seasonal":
            # Fluktuasi periodik sinusoidal: mu_k(t) = base_k + A_k * sin(2*pi*t / Period + phi)
            for k in range(self.n_arms):
                params = self.seasonal_params[k] if k < len(self.seasonal_params) else {}
                amp = params.get("amplitude", 0.25)
                period = params.get("period", 365.0)
                phase_shift = params.get("phase", 0.0)
                base = self.initial_means[k]

                val = base + amp * np.sin((2.0 * np.pi * t / period) + phase_shift)
                self.current_means[k] = np.clip(val, self.min_bound, self.max_bound)

        elif self.mode == "fatigue":
            # Resting bandit: Arm yang ditarik mengalami penurunan kualitas, arm yang diam pulih
            decay = self.fatigue_params.get("decay", 0.06)
            recovery = self.fatigue_params.get("recovery", 0.02)

            for k in range(self.n_arms):
                if k == pulled_arm:
                    # Kualitas arm yang ditarik menurun
                    self.current_means[k] = max(self.min_bound, self.current_means[k] - decay)
                else:
                    # Arm yang istirahat pulih mendekati baseline awal
                    target = self.initial_means[k]
                    if self.current_means[k] < target:
                        self.current_means[k] = min(target, self.current_means[k] + recovery)

        self.means_history.append(self.current_means.copy())

    def step(self, arm: int) -> float:
        if not (0 <= arm < self.n_arms):
            raise IndexError(f"Arm {arm} di luar jangkauan (0 sampai {self.n_arms - 1}).")

        # Ambil mean saat ini
        mean_val = float(self.current_means[arm])

        # Bangkitkan stochastic reward berdasarkan jenis reward
        if self.reward_type == "beta":
            alpha = max(0.1, mean_val * self.precision)
            beta = max(0.1, (1.0 - mean_val) * self.precision)
            reward = float(self.rng.beta(alpha, beta))
        elif self.reward_type == "bernoulli":
            reward = float(self.rng.random() < mean_val)
        elif self.reward_type == "gaussian":
            reward = float(self.rng.normal(mean_val, 0.10))
        else:
            reward = mean_val

        self.current_step += 1
        # Update dinamika lingkungan untuk langkah berikutnya
        self._update_dynamics(pulled_arm=arm)

        return reward

    def reset(self) -> None:
        super().reset()
        self.rng = np.random.default_rng(self.seed)
        self.current_means = self.initial_means.copy()
        self.means_history.clear()
