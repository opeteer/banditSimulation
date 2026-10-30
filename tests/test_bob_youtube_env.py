"""
Unit Test untuk Environment Non-Stationer Rekomendasi YouTube Bob (Poisson Bounded [1, 10])
==========================================================================================
"""

import unittest
import numpy as np
from bandit.bob_youtube_env import BobYouTubeEnv


class TestBobYouTubeEnv(unittest.TestCase):
    """Pengujian fungsionalitas dan jadwal Poisson BobYouTubeEnv."""

    def setUp(self):
        self.env = BobYouTubeEnv(
            total_days=365,
            base_lambda=4.5,
            weekend_boost=0.20,
            min_scrolls_per_day=1,
            max_scrolls_per_day=10,
            seed=42,
        )

    def test_initialization(self):
        """Memeriksa atribut awal environment."""
        self.assertEqual(self.env.n_arms, 6)
        self.assertEqual(len(self.env.CATEGORIES), 6)
        self.assertEqual(self.env.total_days, 365)
        self.assertGreater(self.env.total_steps, 365)  # karena min 1, rata-rata ~4.5-5.4
        self.assertEqual(self.env.current_step, 0)
        self.assertEqual(self.env.current_day, 1)

    def test_poisson_daily_scroll_bounds(self):
        """
        Memverifikasi bahwa seluruh hari dalam setahun memiliki jumlah scrolling
        yang dibatasi secara ketat dalam interval [1, 10].
        """
        counts = self.env.daily_scroll_counts
        self.assertEqual(len(counts), 365)
        self.assertTrue(np.all(counts >= 1), "Ditemukan hari dengan scrolling < 1")
        self.assertTrue(np.all(counts <= 10), "Ditemukan hari dengan scrolling > 10")

    def test_weekend_boost_statistics(self):
        """
        Memverifikasi bahwa rata-rata scrolling akhir pekan (Sabtu & Minggu)
        lebih tinggi daripada hari kerja mendekati target boost 20%.
        """
        stats = self.env.get_daily_scroll_statistics()
        self.assertEqual(stats["total_days"], 365)
        self.assertEqual(stats["weekday_days"] + stats["weekend_days"], 365)

        # Rata-rata akhir pekan harus lebih tinggi dari hari kerja
        self.assertGreater(
            stats["weekend_mean_scrolls"],
            stats["weekday_mean_scrolls"],
            f"Weekend mean ({stats['weekend_mean_scrolls']:.2f}) seharusnya lebih tinggi dari weekday ({stats['weekday_mean_scrolls']:.2f})",
        )

        # Boost empiris berada di sekitar 20% (toleransi wajar untuk sample 365 hari)
        self.assertGreater(stats["empirical_boost_pct"], 10.0)
        self.assertLess(stats["empirical_boost_pct"], 35.0)

    def test_step_to_day_mapping_and_progression(self):
        """
        Memverifikasi mapping langkah demi langkah:
        - Jumlah total langkah sama dengan jumlah akumulatif daily_scroll_counts.
        - current_scroll_in_day berjalan dari 1 sampai k harian lalu berganti ke hari berikutnya.
        """
        total_steps = self.env.total_steps
        self.assertEqual(total_steps, int(np.sum(self.env.daily_scroll_counts)))

        # Simulasikan interaksi hari ke-1
        day1_max = self.env.daily_scroll_counts[0]
        for s in range(day1_max):
            self.assertEqual(self.env.current_day, 1)
            scroll_now, max_s = self.env.current_scroll_in_day
            self.assertEqual(scroll_now, s + 1)
            self.assertEqual(max_s, day1_max)
            self.env.step(0)

        # Setelah menghabiskan jatah hari ke-1, langkah berikutnya harus masuk ke hari ke-2
        self.assertEqual(self.env.current_day, 2)
        scroll_now, max_s = self.env.current_scroll_in_day
        self.assertEqual(scroll_now, 1)
        self.assertEqual(max_s, self.env.daily_scroll_counts[1])

    def test_optimal_arms_throughout_storyline(self):
        """
        Memverifikasi bahwa arm optimal di setiap fase sesuai dengan alur cerita:
        - Hari 25  : Teknologi & Coding (Arm 0)
        - Hari 70  : Gaming & Esports (Arm 3)
        - Hari 130 : Kebugaran & Olahraga (Arm 2)
        - Hari 175 : Musik & Hiburan (Arm 5)
        - Hari 240 : Berita & Finansial (Arm 4)
        - Hari 290 : Resep & Kuliner (Arm 1)
        - Hari 350 : Musik & Hiburan (Arm 5)
        """
        test_checkpoints = [
            (25, 0, "Teknologi & Coding"),
            (70, 3, "Gaming & Esports"),
            (130, 2, "Kebugaran & Olahraga"),
            (175, 5, "Musik & Hiburan"),
            (240, 4, "Berita & Finansial"),
            (290, 1, "Resep & Kuliner"),
            (350, 5, "Musik & Hiburan"),
        ]

        for day, expected_arm, expected_name in test_checkpoints:
            # Cari step index pertama pada hari tersebut
            step_idx = int(np.where(self.env.step_to_day == day)[0][0])
            self.env.current_step = step_idx
            self.assertEqual(self.env.current_day, day)

            opt_arm = self.env.get_optimal_arm()
            self.assertEqual(
                opt_arm,
                expected_arm,
                f"Di Hari {day}, arm optimal seharusnya {expected_name} ({expected_arm}), tapi didapat {opt_arm}",
            )

    def test_step_and_reward_bounds(self):
        """Memeriksa reward bounded kontinu [0, 1]."""
        self.env.reset()
        rewards = []
        for _ in range(50):
            arm = np.random.randint(0, self.env.n_arms)
            r = self.env.step(arm)
            self.assertTrue(0.0 <= r <= 1.0)
            rewards.append(r)

        self.assertEqual(self.env.current_step, 50)
        self.assertTrue(any(r > 0 for r in rewards))

    def test_bernoulli_mode(self):
        """Memeriksa reward binary (0 atau 1) saat mode Bernoulli diaktifkan."""
        env_b = BobYouTubeEnv(total_days=100, reward_type="bernoulli", seed=123)
        rewards = [env_b.step(0) for _ in range(30)]
        for r in rewards:
            self.assertIn(r, [0.0, 1.0])

    def test_ground_truth_trajectory_structure(self):
        """Memverifikasi kalkulasi trajectory ground truth tanpa merusak state."""
        self.env.current_step = 100
        traj = self.env.get_ground_truth_trajectory()

        self.assertEqual(len(traj["days"]), 365)
        self.assertEqual(traj["daily_means"].shape, (365, 6))
        self.assertEqual(len(traj["optimal_arms"]), 365)
        self.assertEqual(len(traj["daily_scroll_counts"]), 365)
        self.assertEqual(self.env.current_step, 100)

    def test_fatigue_mechanism(self):
        """Memeriksa penurunan expected reward jika arm ditarik beruntun."""
        env_f = BobYouTubeEnv(
            total_days=100,
            enable_fatigue=True,
            fatigue_decay=0.10,
            seed=777,
        )
        base_mu = env_f.get_expected_rewards()[0]

        # Tarik arm 0 lima kali berturut-turut
        for _ in range(5):
            env_f.step(0)

        fatigued_mu = env_f.get_expected_rewards()[0]
        self.assertLess(fatigued_mu, base_mu)

        # Istirahatkan arm 0
        for _ in range(10):
            env_f.step(1)

        recovered_mu = env_f.get_expected_rewards()[0]
        self.assertGreater(recovered_mu, fatigued_mu)


if __name__ == "__main__":
    unittest.main()
