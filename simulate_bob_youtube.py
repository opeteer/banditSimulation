"""
Simulasi Head-to-Head: SW-Thompson Sampling vs Kalman-UCB
pada Lingkungan Rekomendasi YouTube Bob (Poisson Bounded [1, 10] & 1 Tahun)
==========================================================================
"""

import os
import json
import time
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt

from bandit.bob_youtube_env import BobYouTubeEnv
from bandit.non_stationary_agents import (
    SlidingWindowThompsonSamplingAgent,
    KalmanUCBAgent,
)
from bandit.agents import RandomAgent, ThompsonSamplingAgent


def run_duel_simulation(
    n_runs: int = 100,
    total_days: int = 365,
    seed: int = 42,
    output_dir: str = "results/bob_youtube",
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    print("=" * 76)
    print("SIMULASI DUEL MULTI-ARMED BANDIT: SW-THOMPSON SAMPLING VS KALMAN-UCB")
    print(f"Lingkungan: Rekomendasi YouTube Bob ({total_days} Hari, Poisson Bounded [1, 10])")
    print(f"Replikasi Monte Carlo: {n_runs} runs | Seed: {seed}")
    print("=" * 76)

    # 1. Inisialisasi representasi jadwal ground truth
    sample_env = BobYouTubeEnv(total_days=total_days, seed=seed)
    ground_truth = sample_env.get_ground_truth_trajectory()
    n_arms = sample_env.n_arms
    categories = sample_env.CATEGORIES
    phases = sample_env.phases

    # Tentukan total steps rata-rata dari sample run
    print(f"\n[1] Menganalisis jadwal kalender dan intensitas Poisson...")
    sample_stats = sample_env.get_daily_scroll_statistics()
    print(f"    - Hari Kerja: {sample_stats['weekday_mean_scrolls']:.2f} scroll/hari")
    print(f"    - Akhir Pekan (+20%): {sample_stats['weekend_mean_scrolls']:.2f} scroll/hari")
    print(f"    - Estimasi Total Interaksi per Tahun: ~{sample_stats['total_steps']} sesi")

    # Siapkan Algoritma yang Diadu
    # Kita bandingkan SW-TS dan Kalman-UCB, plus Standard TS & Random sebagai pembanding
    agent_configs = [
        (
            "Sliding-Window TS (W=120)",
            lambda s: SlidingWindowThompsonSamplingAgent(
                n_arms=n_arms, window_size=120, name="Sliding-Window TS (W=120)", seed=s
            ),
            "#10b981",  # Emerald Green
            "-",
        ),
        (
            "Kalman-UCB (c=1.5, σ_w=0.02)",
            lambda s: KalmanUCBAgent(
                n_arms=n_arms,
                c=1.5,
                process_noise_std=0.02,
                observation_noise_std=0.15,
                name="Kalman-UCB (c=1.5, σ_w=0.02)",
                seed=s,
            ),
            "#3b82f6",  # Royal Blue
            "-",
        ),
        (
            "Standard TS (Statis / Inerte)",
            lambda s: ThompsonSamplingAgent(
                n_arms=n_arms, name="Standard TS (Statis / Inerte)", seed=s
            ),
            "#f59e0b",  # Amber
            "--",
        ),
        (
            "Pure Random (Baseline)",
            lambda s: RandomAgent(
                n_arms=n_arms, name="Pure Random (Baseline)", seed=s
            ),
            "#94a3b8",  # Slate Gray
            ":",
        ),
    ]

    # Kita tentukan n_steps seragam untuk matriks Monte Carlo (berdasarkan langkah run)
    # Rata-rata 365 * 4.75 ≈ 1735 steps. Kita pakai horizon 1750 langkah.
    n_steps = 1750

    results: Dict[str, Dict[str, Any]] = {}
    for name, _, color, linestyle in agent_configs:
        results[name] = {
            "rewards": np.zeros((n_runs, n_steps), dtype=float),
            "regrets": np.zeros((n_runs, n_steps), dtype=float),
            "optimal_choices": np.zeros((n_runs, n_steps), dtype=float),
            "arm_pulls": np.zeros((n_runs, n_arms), dtype=int),
            "color": color,
            "linestyle": linestyle,
        }

    # Jalankan Simulasi Monte Carlo
    print(f"\n[2] Menjalankan {n_runs} Replikasi Monte Carlo...")
    start_time = time.time()

    for run_idx in range(n_runs):
        run_seed = int(rng.integers(0, 1_000_000_000))

        # Untuk setiap agen, buat instance baru dan uji di environment dengan run_seed yang sama persis
        for name, agent_factory, _, _ in agent_configs:
            env = BobYouTubeEnv(total_days=total_days, seed=run_seed)
            agent = agent_factory(run_seed + 101)

            # Jika env.total_steps berbeda tipis dengan n_steps, sesuaikan loop
            steps_this_run = min(n_steps, env.total_steps)

            for step_idx in range(steps_this_run):
                current_exp = env.get_expected_rewards()
                opt_reward = float(np.max(current_exp))
                opt_arm = int(np.argmax(current_exp))

                arm = agent.select_arm()
                reward = env.step(arm)
                agent.update(arm, reward)

                regret = opt_reward - current_exp[arm]

                results[name]["rewards"][run_idx, step_idx] = reward
                results[name]["regrets"][run_idx, step_idx] = regret
                results[name]["optimal_choices"][run_idx, step_idx] = (
                    1.0 if arm == opt_arm else 0.0
                )

            # Isi sisa jika ada
            if steps_this_run < n_steps:
                last_r = results[name]["rewards"][run_idx, steps_this_run - 1]
                last_reg = results[name]["regrets"][run_idx, steps_this_run - 1]
                results[name]["rewards"][run_idx, steps_this_run:] = last_r
                results[name]["regrets"][run_idx, steps_this_run:] = last_reg

            results[name]["arm_pulls"][run_idx] = agent.counts.copy()

        if (run_idx + 1) % 25 == 0 or (run_idx + 1) == n_runs:
            print(f"    -> Progress: {run_idx + 1}/{n_runs} runs selesai...")

    elapsed = time.time() - start_time
    print(f"    Selesai dalam {elapsed:.2f} detik.")

    # 3. Hitung Statistik Rangkuman
    summary_data = []
    print("\n" + "=" * 80)
    print(f"{'Algoritma':<30} | {'Total Reward (Kepuasan)':<22} | {'Cumulative Regret':<18} | {'% Optimal Akhir':<14}")
    print("-" * 80)

    for name, _, _, _ in agent_configs:
        cum_rewards = np.sum(results[name]["rewards"], axis=1)
        cum_regrets = np.sum(results[name]["regrets"], axis=1)
        # 100 langkah terakhir untuk persentase optimal akhir
        final_opt_pct = np.mean(results[name]["optimal_choices"][:, -100:]) * 100.0

        mean_rew = float(np.mean(cum_rewards))
        std_rew = float(np.std(cum_rewards))
        mean_reg = float(np.mean(cum_regrets))
        std_reg = float(np.std(cum_regrets))

        summary_data.append({
            "name": name,
            "mean_reward": mean_rew,
            "std_reward": std_rew,
            "mean_regret": mean_reg,
            "std_regret": std_reg,
            "final_optimal_pct": float(final_opt_pct),
        })

        print(f"{name:<30} | {mean_rew:>9.2f} ± {std_rew:<8.2f} | {mean_reg:>8.2f} ± {std_reg:<6.2f} | {final_opt_pct:>10.1f}%")

    print("=" * 80)

    # 4. Generate Visualisasi Grafik Lengkap
    print(f"\n[3] Membuat grafik visualisasi perbandingan...")
    _plot_all_visualizations(
        ground_truth=ground_truth,
        results=results,
        n_steps=n_steps,
        categories=categories,
        phases=phases,
        output_dir=output_dir,
    )

    # 5. Simpan Ringkasan JSON
    summary_json_path = os.path.join(output_dir, "bob_youtube_duel_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    return {
        "summary": summary_data,
        "n_runs": n_runs,
        "n_steps": n_steps,
        "elapsed": elapsed,
        "output_dir": output_dir,
    }


def _plot_all_visualizations(
    ground_truth: Dict[str, Any],
    results: Dict[str, Any],
    n_steps: int,
    categories: List[str],
    phases: List[Dict[str, Any]],
    output_dir: str,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    steps_arr = np.arange(1, n_steps + 1)

    # ----------------------------------------------------
    # Plot 1: Ground Truth Trajectory Preferensi Bob (365 Hari)
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 6))
    days = ground_truth["days"]
    daily_means = ground_truth["daily_means"]

    colors = ["#2563eb", "#d97706", "#059669", "#dc2626", "#7c3aed", "#ec4899"]
    for i, cat in enumerate(categories):
        ax.plot(days, daily_means[:, i], label=cat, color=colors[i], linewidth=2.2, alpha=0.85)

    # Garis pembatas fase
    for phase in phases:
        start_d, end_d = phase["days"]
        ax.axvline(start_d, color="#94a3b8", linestyle=":", alpha=0.6)
        # Label fase di atas
        mid_d = (start_d + end_d) / 2
        ax.text(
            mid_d,
            0.98,
            f"Fase {phase['phase_id']}",
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cbd5e1", alpha=0.85),
        )

    ax.set_title("Ground Truth Preferensi Konten YouTube Bob (365 Hari / 7 Fase)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Hari ke- (1 sampai 365)", fontsize=11)
    ax.set_ylabel("Nilai Harapan Kepuasan (μ)", fontsize=11)
    ax.set_ylim(0.15, 1.02)
    ax.set_xlim(1, 365)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "bob_youtube_ground_truth_trajectory.png"), dpi=300)
    plt.close(fig)

    # ----------------------------------------------------
    # Plot 2: Cumulative Regret Comparison (Duel Utama)
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))

    for name, data in results.items():
        cum_reg = np.cumsum(data["regrets"], axis=1)
        mean_reg = np.mean(cum_reg, axis=0)
        std_reg = np.std(cum_reg, axis=0)

        # Plot garis rata-rata
        ax.plot(
            steps_arr,
            mean_reg,
            label=f"{name} (Akhir: {mean_reg[-1]:.1f})",
            color=data["color"],
            linestyle=data["linestyle"],
            linewidth=2.4 if "SW" in name or "Kalman" in name else 1.8,
        )
        # Shaded confidence band (± 1 std)
        if "SW" in name or "Kalman" in name:
            ax.fill_between(
                steps_arr,
                np.maximum(0, mean_reg - std_reg),
                mean_reg + std_reg,
                color=data["color"],
                alpha=0.15,
            )

    ax.set_title("Perbandingan Cumulative Regret: SW-Thompson Sampling vs Kalman-UCB", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Langkah Interaksi Scrolling (t)", fontsize=11)
    ax.set_ylabel("Cumulative Expected Regret", fontsize=11)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=10)
    ax.set_xlim(1, n_steps)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "bob_youtube_duel_cumulative_regret.png"), dpi=300)
    plt.close(fig)

    # ----------------------------------------------------
    # Plot 3: Optimal Choice Rate (%) Over Time (Adaptation Speed)
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    window_smooth = 40

    for name, data in results.items():
        raw_rates = np.mean(data["optimal_choices"], axis=0) * 100.0
        # Rolling mean smoothing
        smoothed = np.convolve(raw_rates, np.ones(window_smooth) / window_smooth, mode="valid")
        valid_steps = steps_arr[window_smooth - 1:]

        ax.plot(
            valid_steps,
            smoothed,
            label=name,
            color=data["color"],
            linestyle=data["linestyle"],
            linewidth=2.2 if "SW" in name or "Kalman" in name else 1.6,
        )

    ax.set_title("Tingkat Pemilihan Kategori Optimal (Adaptation Tracking Speed)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Langkah Interaksi Scrolling (t)", fontsize=11)
    ax.set_ylabel("Peluang Memilih Kategori Optimal (%)", fontsize=11)
    ax.set_ylim(-2, 102)
    ax.set_xlim(1, n_steps)
    ax.axhline(100.0 / len(categories), color="gray", linestyle=":", label="Peluang Acak (16.7%)", alpha=0.7)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "bob_youtube_duel_optimal_choice_rate.png"), dpi=300)
    plt.close(fig)

    # ----------------------------------------------------
    # Plot 4: Cumulative Rewards (Total Kepuasan / Watch Time)
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))

    for name, data in results.items():
        cum_rew = np.cumsum(data["rewards"], axis=1)
        mean_rew = np.mean(cum_rew, axis=0)

        ax.plot(
            steps_arr,
            mean_rew,
            label=f"{name} (Total: {mean_rew[-1]:.1f})",
            color=data["color"],
            linestyle=data["linestyle"],
            linewidth=2.2 if "SW" in name or "Kalman" in name else 1.6,
        )

    ax.set_title("Akumulasi Skor Kepuasan Menonton Bob (Cumulative Reward)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Langkah Interaksi Scrolling (t)", fontsize=11)
    ax.set_ylabel("Total Skor Kepuasan", fontsize=11)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=10)
    ax.set_xlim(1, n_steps)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "bob_youtube_duel_cumulative_reward.png"), dpi=300)
    plt.close(fig)

    # ----------------------------------------------------
    # Plot 5: Dashboard Komprehensif (4 Subplots)
    # ----------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # Subplot (0, 0): Ground truth
    for i, cat in enumerate(categories):
        axes[0, 0].plot(days, daily_means[:, i], label=cat, color=colors[i], linewidth=1.8, alpha=0.8)
    axes[0, 0].set_title("(A) Ground Truth Dinamika Preferensi Bob (365 Hari)", fontweight="bold", fontsize=11)
    axes[0, 0].set_xlabel("Hari ke-")
    axes[0, 0].set_ylabel("Nilai Harapan (μ)")
    axes[0, 0].set_ylim(0.15, 1.02)
    axes[0, 0].legend(fontsize=8, loc="lower right")

    # Subplot (0, 1): Cumulative Regret
    for name, data in results.items():
        cum_reg = np.cumsum(data["regrets"], axis=1)
        mean_reg = np.mean(cum_reg, axis=0)
        axes[0, 1].plot(steps_arr, mean_reg, label=name, color=data["color"], linestyle=data["linestyle"], linewidth=2.0)
    axes[0, 1].set_title("(B) Perbandingan Cumulative Regret", fontweight="bold", fontsize=11)
    axes[0, 1].set_xlabel("Langkah Interaksi (t)")
    axes[0, 1].set_ylabel("Regret Kumulatif")
    axes[0, 1].legend(fontsize=8.5, loc="upper left")

    # Subplot (1, 0): Optimal Choice Rate
    for name, data in results.items():
        raw_rates = np.mean(data["optimal_choices"], axis=0) * 100.0
        smoothed = np.convolve(raw_rates, np.ones(window_smooth) / window_smooth, mode="valid")
        axes[1, 0].plot(valid_steps, smoothed, label=name, color=data["color"], linestyle=data["linestyle"], linewidth=1.8)
    axes[1, 0].set_title("(C) Kecepatan Adaptasi & Optimal Choice Rate (%)", fontweight="bold", fontsize=11)
    axes[1, 0].set_xlabel("Langkah Interaksi (t)")
    axes[1, 0].set_ylabel("% Pilihan Optimal")
    axes[1, 0].legend(fontsize=8.5, loc="lower right")

    # Subplot (1, 1): Total Pulls per Arm Distribution
    bar_width = 0.20
    x_indices = np.arange(len(categories))
    agent_names = list(results.keys())

    for idx, name in enumerate(agent_names):
        avg_pulls = np.mean(results[name]["arm_pulls"], axis=0)
        pos = x_indices + (idx - 1.5) * bar_width
        axes[1, 1].bar(
            pos,
            avg_pulls,
            width=bar_width,
            label=name.split("(")[0].strip(),
            color=results[name]["color"],
            alpha=0.85,
        )

    axes[1, 1].set_title("(D) Distribusi Alokasi Rekomendasi per Kategori", fontweight="bold", fontsize=11)
    axes[1, 1].set_xticks(x_indices)
    axes[1, 1].set_xticklabels([c.replace(" & ", "\n& ") for c in categories], fontsize=8.5)
    axes[1, 1].set_ylabel("Rata-rata Tarikan")
    axes[1, 1].legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "bob_youtube_duel_dashboard.png"), dpi=300)
    plt.close(fig)
    print("    [✓] Semua grafik berhasil disimpan ke folder:", output_dir)


if __name__ == "__main__":
    run_duel_simulation(n_runs=100, total_days=365, seed=42)
