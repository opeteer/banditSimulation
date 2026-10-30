"""
Modul Environment Multi-Armed Bandit Non-Stationer: Kasus Rekomendasi YouTube Bob
================================================================================
Memodelkan perilaku scrolling smartphone seorang pengguna (Bob) selama 1 tahun (365 hari).
Lingkungan ini bersifat non-stasioner skriptif (scripted timeline) dengan dinamika stochastic:
1. Frekuensi scrolling harian tidak konstan, melainkan terdistribusi Poisson bounded [1, 10]
2. Pada akhir pekan (Sabtu & Minggu), intensitas scrolling meningkat sebesar 20%
3. Preferensi topik tontonan Bob bertransformasi mengikuti 7 fase kehidupan (bootcamp coding,
   tren game blockbuster viral, diet/gym, liburan musik, kepanikan finansial, hobi masak, liburan santai).
4. (Opsional) Efek kejenuhan (Fatigue) jika suatu kategori disodorkan beruntun.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from .environment import BanditEnvironment


class BobYouTubeEnv(BanditEnvironment):
    """
    Environment Non-Stationer Skriptif untuk Rekomendasi YouTube Bob selama 1 Tahun (365 Hari)
    dengan Jumlah Sesi Scrolling Harian Berdistribusi Poisson Dinamis.

    Kategori Arm (K = 6):
    0: 'Teknologi & Coding'   - Tutorial Python, Web Dev, AI Tools, Gadget
    1: 'Resep & Kuliner'       - Meal-Prep Sehat, Masakan Rumahan, Baking
    2: 'Kebugaran & Olahraga' - Latihan Gym, Kardio, Running, Nutrisi
    3: 'Gaming & Esports'      - Walkthrough Game AAA, Turnamen Esports, Streaming
    4: 'Berita & Finansial'    - Investasi Saham, Kripto, Makroekonomi, Edukasi Finansial
    5: 'Musik & Hiburan'       - Podcast Komedi, Musik Konser, Vlog Santai
    """

    CATEGORIES = [
        "Teknologi & Coding",
        "Resep & Kuliner",
        "Kebugaran & Olahraga",
        "Gaming & Esports",
        "Berita & Finansial",
        "Musik & Hiburan",
    ]

    def __init__(
        self,
        total_days: int = 365,
        base_lambda: float = 4.5,
        weekend_boost: float = 0.20,
        min_scrolls_per_day: int = 1,
        max_scrolls_per_day: int = 10,
        first_day_of_week: int = 0,  # 0: Senin, 1: Selasa, ..., 5: Sabtu, 6: Minggu
        reward_type: str = "continuous",
        precision: float = 30.0,
        enable_fatigue: bool = False,
        fatigue_decay: float = 0.08,
        fatigue_recovery: float = 0.03,
        seed: Optional[int] = None,
    ):
        """
        Parameter:
            total_days: Durasi total dunia simulasi (default: 365 hari / 1 tahun).
            base_lambda: Parameter laju (lambda) Poisson rata-rata scrolling di hari kerja (default: 4.5).
            weekend_boost: Persentase kenaikan laju scrolling di akhir pekan (default: 0.20 = 20%).
            min_scrolls_per_day: Batas minimal scrolling dalam sehari (default: 1).
            max_scrolls_per_day: Batas maksimal scrolling dalam sehari (default: 10).
            first_day_of_week: Hari pembuka kalender (0 = Senin, 6 = Minggu).
            reward_type: 'continuous' (skor rasio tontonan [0, 1] via distribusi Beta)
                         atau 'bernoulli' (0 = lewati/skip, 1 = tonton/klik).
            precision: Presisi konsistensi respons Bob (kappa distribusi Beta).
            enable_fatigue: Jika True, Bob akan merasa jenuh jika kategori yang sama disodorkan berulang.
            fatigue_decay: Penurunan nilai harapan per tarikan beruntun jika fatigue aktif.
            fatigue_recovery: Pemulihan nilai harapan per langkah jika arm tidak dipilih.
            seed: Seed untuk random number generator (reproducibility).
        """
        super().__init__(n_arms=len(self.CATEGORIES))
        self.total_days = total_days
        self.base_lambda = base_lambda
        self.weekend_boost = weekend_boost
        self.min_scrolls_per_day = min_scrolls_per_day
        self.max_scrolls_per_day = max_scrolls_per_day
        self.first_day_of_week = first_day_of_week
        self.reward_type = reward_type.lower()
        self.precision = precision
        self.enable_fatigue = enable_fatigue
        self.fatigue_decay = fatigue_decay
        self.fatigue_recovery = fatigue_recovery
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Melacak consecutive pulls untuk fatigue
        self.consecutive_pulls = np.zeros(self.n_arms, dtype=int)
        self.current_fatigue = np.zeros(self.n_arms, dtype=float)

        # Mendefinisikan Alur Cerita Skriptif 1 Tahun Bob
        self.phases = self._init_scripted_phases()

        # Membangkitkan Jadwal Scrolling Harian Berdistribusi Poisson
        self._generate_daily_schedule()

    def _init_scripted_phases(self) -> List[Dict[str, Any]]:
        """Membangun alur cerita scripted preferensi Bob sepanjang 365 hari."""
        return [
            {
                "phase_id": 1,
                "name": "Resolusi Tahun Baru: Belajar Coding Python",
                "days": (1, 50),
                "story": "Bob bersemangat mengawali tahun dengan ikut bootcamp programming. Konten coding sangat ia cari.",
                "type": "abrupt",
                "base_means": [0.88, 0.35, 0.50, 0.38, 0.32, 0.55],
                "optimal_arm": 0,  # Teknologi & Coding
            },
            {
                "phase_id": 2,
                "name": "Rilis Game AAA Viral (Abrupt Trend Shock)",
                "days": (51, 85),
                "story": "Game blockbuster baru rilis di pasaran. Bob rehat coding dan malam-malamnya dipenuhi nonton walkthrough & tips game.",
                "type": "abrupt",
                "base_means": [0.42, 0.35, 0.45, 0.92, 0.30, 0.52],
                "optimal_arm": 3,  # Gaming & Esports
            },
            {
                "phase_id": 3,
                "name": "Fase Diet & Gym Musim Panas (Continuous Drift)",
                "days": (86, 145),
                "story": "Bob mulai rutin ke tempat gym dan meal-prep makanan sehat. Kebugaran merangkak naik perlahan menggantikan gaming.",
                "type": "drift",
                "start_means": [0.45, 0.40, 0.55, 0.40, 0.35, 0.50],
                "end_means":   [0.40, 0.75, 0.88, 0.28, 0.35, 0.50],
                "optimal_arm": 2,  # Kebugaran & Olahraga
            },
            {
                "phase_id": 4,
                "name": "Musim Liburan & Festival Musik (Seasonal Vibe)",
                "days": (146, 205),
                "story": "Tengah tahun musim liburan, Bob traveling dan menikmati konser musik. Bob tidak ingin menonton konten berat.",
                "type": "abrupt",
                "base_means": [0.30, 0.50, 0.42, 0.35, 0.25, 0.90],
                "optimal_arm": 5,  # Musik & Hiburan
            },
            {
                "phase_id": 5,
                "name": "Kepanikan Finansial & Minat Investasi",
                "days": (206, 265),
                "story": "Membaca berita inflasi dan tren pasar saham, Bob termotivasi mengatur portofolio dan literasi keuangan.",
                "type": "drift",
                "start_means": [0.35, 0.45, 0.40, 0.30, 0.60, 0.50],
                "end_means":   [0.40, 0.45, 0.40, 0.28, 0.89, 0.45],
                "optimal_arm": 4,  # Berita & Finansial
            },
            {
                "phase_id": 6,
                "name": "Hobi Masak Rumahan & World Championship Gaming",
                "days": (266, 320),
                "story": "Bob mulai senang memasak sendiri di rumah (kuliner terbaik). Di akhir pekan ada turnamen esports dunia.",
                "type": "abrupt",
                "base_means": [0.38, 0.90, 0.45, 0.72, 0.40, 0.55],
                "optimal_arm": 1,  # Resep & Kuliner
            },
            {
                "phase_id": 7,
                "name": "Liburan Akhir Tahun: Santai & Nostalgia",
                "days": (321, 365),
                "story": "Tahun menjelang usai, Bob mengambil cuti. Dia hanya ingin menonton video santai, musik, dan podcast komedi.",
                "type": "abrupt",
                "base_means": [0.25, 0.45, 0.35, 0.40, 0.20, 0.92],
                "optimal_arm": 5,  # Musik & Hiburan
            },
        ]

    def is_weekend(self, day: int) -> bool:
        """
        Menentukan apakah hari ke-`day` (1-based) adalah akhir pekan (Sabtu/Minggu).
        Diasumsikan 0 = Senin, 1 = Selasa, ..., 5 = Sabtu, 6 = Minggu.
        """
        day_of_week = (self.first_day_of_week + (day - 1)) % 7
        return day_of_week in (5, 6)

    def _generate_daily_schedule(self) -> None:
        """
        Membangkitkan jumlah scrolling untuk setiap hari d = 1..total_days.
        - Hari kerja: lambda = base_lambda
        - Akhir pekan: lambda = base_lambda * (1 + weekend_boost) (naik 20%)
        - Jumlah scrolling dibatasi (clipped) dalam interval [min_scrolls, max_scrolls] (1 s.d 10).
        """
        self.daily_scroll_counts = np.zeros(self.total_days, dtype=int)
        self.daily_lambdas = np.zeros(self.total_days, dtype=float)

        step_to_day_list: List[int] = []
        step_in_day_list: List[int] = []

        for d in range(1, self.total_days + 1):
            if self.is_weekend(d):
                lam = self.base_lambda * (1.0 + self.weekend_boost)
            else:
                lam = self.base_lambda

            self.daily_lambdas[d - 1] = lam

            # Sampling Poisson dengan pembatasan [min_scrolls, max_scrolls]
            raw_k = self.rng.poisson(lam)
            k = int(np.clip(raw_k, self.min_scrolls_per_day, self.max_scrolls_per_day))
            self.daily_scroll_counts[d - 1] = k

            for s_idx in range(k):
                step_to_day_list.append(d)
                step_in_day_list.append(s_idx + 1)

        self.step_to_day = np.array(step_to_day_list, dtype=int)
        self.step_in_day = np.array(step_in_day_list, dtype=int)
        self.total_steps = len(self.step_to_day)

    @property
    def current_day(self) -> int:
        """Mengembalikan hari saat ini (1 sampai total_days)."""
        if self.current_step >= len(self.step_to_day):
            return self.total_days
        return int(self.step_to_day[self.current_step])

    @property
    def current_scroll_in_day(self) -> Tuple[int, int]:
        """
        Mengembalikan tuple (scroll_ke, total_scroll_hari_ini)
        Misal: (3, 7) artinya Bob sedang membuka sesi scrolling ke-3 dari 7 sesi hari ini.
        """
        if self.current_step >= len(self.step_to_day):
            day = self.total_days
            max_s = int(self.daily_scroll_counts[day - 1])
            return (max_s, max_s)

        current_s = int(self.step_in_day[self.current_step])
        day = int(self.step_to_day[self.current_step])
        total_s = int(self.daily_scroll_counts[day - 1])
        return (current_s, total_s)

    def get_current_phase(self) -> Dict[str, Any]:
        """Mengembalikan informasi fase scripted yang sedang aktif saat ini."""
        day = self.current_day
        for phase in self.phases:
            start_d, end_d = phase["days"]
            if start_d <= day <= end_d:
                return phase
        return self.phases[-1]

    def get_expected_rewards(self) -> np.ndarray:
        """
        Menghitung nilai harapan (ground-truth mean mu_k) untuk setiap kategori
        pada hari saat ini sesuai jadwal skriptif.
        """
        day = self.current_day
        phase = self.get_current_phase()

        if phase["type"] == "abrupt":
            means = np.array(phase["base_means"], dtype=float)
        elif phase["type"] == "drift":
            start_d, end_d = phase["days"]
            start_m = np.array(phase["start_means"], dtype=float)
            end_m = np.array(phase["end_means"], dtype=float)
            progress = (day - start_d) / max(1, (end_d - start_d))
            means = start_m + progress * (end_m - start_m)
        else:
            means = np.array(phase["base_means"], dtype=float)

        # Terapkan penalti fatigue jika diaktifkan
        if self.enable_fatigue:
            means = means - self.current_fatigue

        return np.clip(means, 0.05, 0.95)

    def step(self, arm: int) -> float:
        """
        Rekomendasikan video dari kategori `arm` kepada Bob pada langkah saat ini.
        Mengembalikan reward berupa rasio tontonan / CTR Bob.
        """
        if not (0 <= arm < self.n_arms):
            raise IndexError(f"Indeks kategori arm {arm} tidak valid (0 sampai {self.n_arms - 1}).")

        expected_means = self.get_expected_rewards()
        mu = float(expected_means[arm])

        if self.reward_type == "bernoulli":
            reward = 1.0 if self.rng.random() < mu else 0.0
        else:
            alpha = max(1e-3, mu * self.precision)
            beta_param = max(1e-3, (1.0 - mu) * self.precision)
            reward = float(self.rng.beta(alpha, beta_param))

        # Update dinamika fatigue jika aktif
        if self.enable_fatigue:
            for a in range(self.n_arms):
                if a == arm:
                    self.consecutive_pulls[a] += 1
                    if self.consecutive_pulls[a] > 1:
                        self.current_fatigue[a] = min(
                            0.50, self.current_fatigue[a] + self.fatigue_decay
                        )
                else:
                    self.consecutive_pulls[a] = 0
                    self.current_fatigue[a] = max(
                        0.0, self.current_fatigue[a] - self.fatigue_recovery
                    )

        self.current_step += 1
        return reward

    def reset(self) -> None:
        """Mereset lingkungan kembali ke Hari 1 / Langkah 0 dengan jadwal Poisson baru/konsisten."""
        self.current_step = 0
        self.consecutive_pulls = np.zeros(self.n_arms, dtype=int)
        self.current_fatigue = np.zeros(self.n_arms, dtype=float)
        self.rng = np.random.default_rng(self.seed)
        self._generate_daily_schedule()

    def get_daily_scroll_statistics(self) -> Dict[str, Any]:
        """
        Mengembalikan statistik komparatif scrolling antara hari kerja vs akhir pekan.
        """
        weekend_mask = np.array([self.is_weekend(d) for d in range(1, self.total_days + 1)])
        weekday_mask = ~weekend_mask

        weekday_counts = self.daily_scroll_counts[weekday_mask]
        weekend_counts = self.daily_scroll_counts[weekend_mask]

        return {
            "total_days": self.total_days,
            "total_steps": self.total_steps,
            "weekday_days": int(np.sum(weekday_mask)),
            "weekend_days": int(np.sum(weekend_mask)),
            "weekday_mean_scrolls": float(np.mean(weekday_counts)),
            "weekend_mean_scrolls": float(np.mean(weekend_counts)),
            "empirical_boost_pct": float(
                (np.mean(weekend_counts) / np.mean(weekday_counts) - 1.0) * 100
            ),
            "theoretical_boost_pct": self.weekend_boost * 100,
            "min_scrolls_observed": int(np.min(self.daily_scroll_counts)),
            "max_scrolls_observed": int(np.max(self.daily_scroll_counts)),
        }

    def get_ground_truth_trajectory(self) -> Dict[str, Any]:
        """
        Membangkitkan kurva trajectory ground-truth ekspektasi seluruh kategori
        dari Hari 1 hingga total_days tanpa mengubah state internal lingkungan.
        """
        original_step = self.current_step
        saved_fatigue = self.current_fatigue.copy()
        saved_consec = self.consecutive_pulls.copy()

        daily_means = np.zeros((self.total_days, self.n_arms))
        optimal_arms = np.zeros(self.total_days, dtype=int)

        for d in range(self.total_days):
            # Lompat ke awal hari d+1
            day_num = d + 1
            # Temukan index langkah pertama hari tersebut
            step_indices = np.where(self.step_to_day == day_num)[0]
            if len(step_indices) > 0:
                self.current_step = int(step_indices[0])
            daily_means[d] = self.get_expected_rewards()
            optimal_arms[d] = int(np.argmax(daily_means[d]))

        self.current_step = original_step
        self.current_fatigue = saved_fatigue
        self.consecutive_pulls = saved_consec

        return {
            "days": np.arange(1, self.total_days + 1),
            "categories": self.CATEGORIES,
            "daily_means": daily_means,
            "optimal_arms": optimal_arms,
            "phases": self.phases,
            "daily_scroll_counts": self.daily_scroll_counts.copy(),
            "daily_lambdas": self.daily_lambdas.copy(),
        }
