"""
Simulasi Multi-Armed Bandit pada Lingkungan Non-Stationer (Dynamic Environment)
==============================================================================
Skenario:
- 1 Agen memilih di antara 10 restoran selama 1.500 sesi makan (3 Fase × 500 sesi).
- Terjadi 'Concept Drift' / Pergeseran Koki & Tren Kuliner mendadak:
  * Fase 1 (1 - 500)   : RM Padang Terbaik (μ = 0.88)
  * Fase 2 (501 - 1000): Koki Pindah! Bistro Western Melonjak Jadi Terbaik (μ = 0.90), RM Padang Anjlok (μ = 0.35)
  * Fase 3 (1001- 1500): Tren Musim Panas! Seafood Pesisir Melejit Jadi Terbaik (μ = 0.92)
- Membandingkan:
  1. Discounted Thompson Sampling (D-TS, γ=0.97) [Adaptif]
  2. Sliding-Window Thompson Sampling (SW-TS, W=150) [Adaptif]
  3. Sliding-Window UCB (SW-UCB, W=150) [Adaptif]
  4. Discounted UCB (D-UCB, γ=0.97) [Adaptif]
  5. Standard Thompson Sampling (Statis / Keras Kepala)
  6. Standard UCB1 (Statis)
  7. Pure Random (Baseline)
"""

import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from bandit.non_stationary_env import DynamicNonStationaryBandit
from bandit.non_stationary_agents import (
    DiscountedThompsonSamplingAgent,
    SlidingWindowThompsonSamplingAgent,
    DiscountedUCBAgent,
    SlidingWindowUCBAgent,
)
from bandit.agents import ThompsonSamplingAgent, UCB1Agent, RandomAgent
from bandit.simulator import Simulator


RESTAURANT_NAMES = [
    "Warung Barokah",
    "Kantin Kampus",
    "Mie Ayam Mas Giri",
    "RM Padang Minang",
    "Bebek Madura",
    "Bistro Western",
    "Fast Food Burger",
    "Kedai Ramen",
    "Warung Pecel Lele",
    "Seafood Pesisir",
]

# Konfigurasi 3 Fase Pergeseran Koki & Tren Kuliner
PHASES = [
    {
        "name": "Fase 1: Kejayaan RM Padang",
        "step_until": 500,
        "means": [0.45, 0.25, 0.55, 0.88, 0.65, 0.50, 0.40, 0.52, 0.48, 0.70],
        "best_arm": 3,
        "best_name": "RM Padang (μ=0.88)",
    },
    {
        "name": "Fase 2: Koki Dibajak Bistro Western",
        "step_until": 1000,
        "means": [0.45, 0.25, 0.55, 0.35, 0.65, 0.90, 0.40, 0.52, 0.48, 0.70],
        "best_arm": 5,
        "best_name": "Bistro Western (μ=0.90)",
    },
    {
        "name": "Fase 3: Tren Seafood Pesisir Viral",
        "step_until": None,
        "means": [0.45, 0.25, 0.55, 0.50, 0.60, 0.55, 0.40, 0.52, 0.48, 0.92],
        "best_arm": 9,
        "best_name": "Seafood Pesisir (μ=0.92)",
    },
]


def run_non_stationary_simulation(
    n_steps: int = 1500,
    n_runs: int = 100,
    seed: int = 42,
    output_dir: str = "results/non_stationary",
):
    os.makedirs(output_dir, exist_ok=True)
    n_arms = len(RESTAURANT_NAMES)

    print("============================================================================")
    print("SIMULASI MULTI-ARMED BANDIT: LINGKUNGAN NON-STATIONER (DYNAMIC CONCEPT DRIFT)")
    print(f"Total Langkah: {n_steps} Sesi Makan (3 Fase × 500 Langkah)")
    print("Fase 1 (1-500)    : Restoran Terbaik = RM Padang (Index 3, μ=0.88)")
    print("Fase 2 (501-1000) : Restoran Terbaik = Bistro Western (Index 5, μ=0.90)")
    print("Fase 3 (1001-1500): Restoran Terbaik = Seafood Pesisir (Index 9, μ=0.92)")
    print("============================================================================")

    # Inisialisasi Suite Agen
    agents = [
        DiscountedThompsonSamplingAgent(n_arms=n_arms, gamma=0.97, name="Discounted TS (γ=0.97) [Adaptif]"),
        SlidingWindowThompsonSamplingAgent(n_arms=n_arms, window_size=150, name="Sliding-Window TS (W=150) [Adaptif]"),
        SlidingWindowUCBAgent(n_arms=n_arms, window_size=150, c=1.0, name="Sliding-Window UCB (W=150) [Adaptif]"),
        DiscountedUCBAgent(n_arms=n_arms, gamma=0.97, c=1.0, name="Discounted UCB (γ=0.97) [Adaptif]"),
        ThompsonSamplingAgent(n_arms=n_arms, name="Standard TS (Statis / Keras Kepala)"),
        UCB1Agent(n_arms=n_arms, c=1.0, name="Standard UCB1 (Statis)"),
        RandomAgent(n_arms=n_arms, name="Pure Random (Baseline)"),
    ]

    def env_factory(s: int) -> DynamicNonStationaryBandit:
        return DynamicNonStationaryBandit(
            n_arms=n_arms,
            mode="abrupt",
            reward_type="beta",
            precision=30.0,
            phases=PHASES,
            initial_means=PHASES[0]["means"],
            seed=s,
        )

    simulator = Simulator(
        env_factory=env_factory,
        agents=agents,
        case_name="Non-Stationary Concept Drift (Pergeseran Koki 3 Fase)",
    )

    # 1. Jalankan Simulasi Monte Carlo
    print(f"\n[1] Menjalankan {n_runs} replikasi Monte Carlo ({n_steps} langkah)...")
    start_time = time.time()
    results = simulator.run(n_steps=n_steps, n_runs=n_runs, seed=seed, verbose=True)
    elapsed = time.time() - start_time
    print(f"    Selesai dalam {elapsed:.2f} detik.")

    # 2. Visualisasi Grafik Komparatif
    print(f"\n[2] Membuat visualisasi grafik non-stationer ke folder: '{output_dir}'...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    colors = {
        "Discounted TS (γ=0.97) [Adaptif]": "#10b981",         # Hijau Emerald
        "Sliding-Window TS (W=150) [Adaptif]": "#059669",      # Hijau Tua
        "Sliding-Window UCB (W=150) [Adaptif]": "#3b82f6",     # Biru
        "Discounted UCB (γ=0.97) [Adaptif]": "#8b5cf6",        # Ungu
        "Standard TS (Statis / Keras Kepala)": "#f59e0b",      # Oranye / Amber
        "Standard UCB1 (Statis)": "#ea580c",                  # Cokelat Oranye
        "Pure Random (Baseline)": "#ef4444",                  # Merah
    }

    steps = np.arange(1, n_steps + 1)

    # Grafik 0: Visualisasi Ground Truth Perubahan Lingkungan (Dinamika Nilai Harapan)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sample_env = env_factory(42)
    means_over_time = np.zeros((n_steps, n_arms))
    for t in range(n_steps):
        means_over_time[t] = sample_env.get_expected_rewards()
        sample_env.step(0)

    for k in range(n_arms):
        is_hero = (k in [3, 5, 9])
        lw = 2.5 if is_hero else 1.2
        alpha = 1.0 if is_hero else 0.4
        label = f"{RESTAURANT_NAMES[k]}" + (" ★" if is_hero else "")
        ax.plot(steps, means_over_time[:, k], label=label, linewidth=lw, alpha=alpha)

    # Garis vertikal penanda pergantian fase
    ax.axvline(500, color="red", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.text(505, 0.95, "Fase 2 Mulai\n(Bistro Western Terbaik)", color="red", fontsize=8.5, fontweight="bold")
    ax.axvline(1000, color="purple", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.text(1005, 0.95, "Fase 3 Mulai\n(Seafood Terbaik)", color="purple", fontsize=8.5, fontweight="bold")

    ax.set_title("Dinamika Nilai Harapan Kepuasan Restoran Seiring Waktu (Ground Truth)", fontweight="bold", pad=12)
    ax.set_xlabel("Langkah Waktu (Sesi Makan 1 s/d 1.500)")
    ax.set_ylabel("Nilai Harapan Kepuasan (Mean μ)")
    ax.set_ylim(0.15, 1.02)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8.5, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    path_ground_truth = os.path.join(output_dir, "non_stationary_environment_ground_truth.png")
    fig.tight_layout()
    fig.savefig(path_ground_truth, dpi=300)
    plt.close(fig)

    # Grafik 1: Akumulasi Reward Kumulatif
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, a_res in results.agent_results.items():
        c = colors.get(name, "#333333")
        mean = a_res.mean_cumulative_rewards
        std = a_res.std_cumulative_rewards / np.sqrt(n_runs)
        ax.plot(steps, mean, label=name, color=c, linewidth=2.0)

    # Garis Oracle Dinamis
    oracle_dynamic = np.zeros(n_steps)
    for t in range(n_steps):
        if t < 500:
            oracle_dynamic[t] = 0.88
        elif t < 1000:
            oracle_dynamic[t] = 0.90
        else:
            oracle_dynamic[t] = 0.92
    oracle_cum = np.cumsum(oracle_dynamic)
    ax.plot(steps, oracle_cum, label="Oracle Dinamis (Ideal)", color="black", linestyle="--", linewidth=1.8, alpha=0.7)

    ax.axvline(500, color="gray", linestyle=":", alpha=0.7)
    ax.axvline(1000, color="gray", linestyle=":", alpha=0.7)
    ax.set_title("Perbandingan Akumulasi Skor Kepuasan pada Lingkungan Non-Stationer", fontweight="bold", pad=12)
    ax.set_xlabel("Langkah Waktu (Sesi Makan 1 s/d 1.500)")
    ax.set_ylabel("Total Skor Kepuasan Kumulatif")
    ax.legend(loc="upper left", frameon=True, fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.6)
    path_cum_rew = os.path.join(output_dir, "non_stationary_cumulative_rewards.png")
    fig.tight_layout()
    fig.savefig(path_cum_rew, dpi=300)
    plt.close(fig)

    # Grafik 2: Akumulasi Penyesalan (Cumulative Regret)
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, a_res in results.agent_results.items():
        c = colors.get(name, "#333333")
        mean = a_res.mean_cumulative_regrets
        ax.plot(steps, mean, label=name, color=c, linewidth=2.2)

    ax.axvline(500, color="red", linestyle=":", alpha=0.7)
    ax.text(505, 50, "Pergeseran 1", color="red", fontsize=8)
    ax.axvline(1000, color="purple", linestyle=":", alpha=0.7)
    ax.text(1005, 50, "Pergeseran 2", color="purple", fontsize=8)

    ax.set_title("Akumulasi Penyesalan (Cumulative Regret): Agen Adaptif vs Agen Statis", fontweight="bold", pad=12)
    ax.set_xlabel("Langkah Waktu (Sesi Makan 1 s/d 1.500)")
    ax.set_ylabel("Cumulative Expected Regret")
    ax.legend(loc="upper left", frameon=True, fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.6)
    path_cum_reg = os.path.join(output_dir, "non_stationary_cumulative_regrets.png")
    fig.tight_layout()
    fig.savefig(path_cum_reg, dpi=300)
    plt.close(fig)

    # Grafik 3: Persentase Memilih Arm Optimal Saat Ini (% Optimal Choice Rate)
    # Ini grafik paling dramatis: menunjukkan drop tajam di step 500 & 1000 dan recovery!
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, a_res in results.agent_results.items():
        c = colors.get(name, "#333333")
        rate = a_res.optimal_action_rate * 100.0
        # Smooth window 15 langkah
        kernel = np.ones(15) / 15.0
        rate_smooth = np.convolve(rate, kernel, mode="same")
        rate_smooth[:15] = rate[:15]
        ax.plot(steps, rate_smooth, label=name, color=c, linewidth=2.0)

    ax.axvline(500, color="red", linestyle="--", alpha=0.8)
    ax.axvline(1000, color="purple", linestyle="--", alpha=0.8)
    ax.axhline(10.0, color="gray", linestyle=":", label="Peluang Acak (10%)")

    ax.set_title("Dinamika Adaptasi: % Memilih Restoran Terbaik Saat Ini Sepanjang Waktu", fontweight="bold", pad=12)
    ax.set_xlabel("Langkah Waktu (Sesi Makan)")
    ax.set_ylabel("Persentase Memilih Restoran Terbaik (%)")
    ax.set_ylim(-2, 102)
    ax.legend(loc="lower right", frameon=True, fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.6)
    path_opt_rate = os.path.join(output_dir, "non_stationary_optimal_choice_rate.png")
    fig.tight_layout()
    fig.savefig(path_opt_rate, dpi=300)
    plt.close(fig)

    # Grafik 4: Dashboard Komprehensif 4-in-1
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # A: Ground Truth Dynamics
    for k in [3, 5, 9, 0]:
        ax1.plot(steps, means_over_time[:, k], label=RESTAURANT_NAMES[k], linewidth=2.0)
    ax1.axvline(500, color="red", linestyle="--")
    ax1.axvline(1000, color="purple", linestyle="--")
    ax1.set_title("(A) Dinamika Pergeseran Restoran Terbaik (Ground Truth)", fontweight="bold")
    ax1.set_xlabel("Langkah (Steps)")
    ax1.set_ylabel("Nilai Harapan Rasa (μ)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # B: Cumulative Regrets
    for name, a_res in results.agent_results.items():
        c = colors.get(name, "#333333")
        ax2.plot(steps, a_res.mean_cumulative_regrets, label=name, color=c, linewidth=1.8)
    ax2.set_title("(B) Akumulasi Penyesalan (Cumulative Regret)", fontweight="bold")
    ax2.set_xlabel("Langkah (Steps)")
    ax2.set_ylabel("Cumulative Regret")
    ax2.legend(loc="upper left", fontsize=7.5)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # C: Optimal Choice Rate
    for name, a_res in results.agent_results.items():
        c = colors.get(name, "#333333")
        rate = a_res.optimal_action_rate * 100.0
        kernel = np.ones(15) / 15.0
        rate_smooth = np.convolve(rate, kernel, mode="same")
        rate_smooth[:15] = rate[:15]
        ax3.plot(steps, rate_smooth, label=name, color=c, linewidth=1.8)
    ax3.axvline(500, color="red", linestyle="--")
    ax3.axvline(1000, color="purple", linestyle="--")
    ax3.set_title("(C) Kecepatan Adaptasi (% Memilih Restoran Terbaik)", fontweight="bold")
    ax3.set_xlabel("Langkah (Steps)")
    ax3.set_ylabel("% Restoran Terbaik")
    ax3.set_ylim(-2, 102)
    ax3.legend(loc="lower right", fontsize=7.5)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # D: Total Poin Akhir Bar Chart
    sorted_agents = sorted(results.agent_results.items(), key=lambda x: x[1].final_mean_reward, reverse=True)
    names_bar = [k.split(" [")[0].replace(" (Statis / Keras Kepala)", " (Statis)") for k, v in sorted_agents]
    rewards_bar = [v.final_mean_reward for k, v in sorted_agents]
    colors_bar = [colors.get(k, "#555") for k, v in sorted_agents]

    ax4.barh(range(len(names_bar)), rewards_bar, color=colors_bar, alpha=0.85)
    ax4.set_yticks(range(len(names_bar)))
    ax4.set_yticklabels(names_bar, fontsize=8)
    ax4.invert_yaxis()
    ax4.set_title("(D) Peringkat Total Skor Kepuasan Akhir", fontweight="bold")
    ax4.set_xlabel("Total Skor Kepuasan")
    ax4.grid(axis="x", linestyle="--", alpha=0.5)

    fig.suptitle("Evaluasi Multi-Armed Bandit pada Lingkungan Non-Stationer (Dynamic Concept Drift)", fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout()
    path_dashboard = os.path.join(output_dir, "non_stationary_dashboard.png")
    fig.savefig(path_dashboard, dpi=300)
    plt.close(fig)

    # 4. Ringkasan & Ekspor Data
    summary_data = {
        "case": "Non-Stationary Concept Drift (Pergeseran Koki 3 Fase)",
        "n_steps": n_steps,
        "n_runs": n_runs,
        "phases": PHASES,
        "algorithms": {},
    }

    print("\n" + "=" * 86)
    print(f"{'Algoritma':<36} | {'Total Kepuasan':<16} | {'Cumulative Regret':<18} | {'% Optimal Akhir':<12}")
    print("-" * 86)

    md_lines = []
    md_lines.append("# Ringkasan Hasil Evaluasi Multi-Armed Bandit Non-Stationer\n")
    md_lines.append(f"- **Horizon:** {n_steps} Sesi Makan (3 Fase $\\times$ 500 Sesi)")
    md_lines.append("- **Pergeseran:** Fase 1 (RM Padang 0.88) $\\rightarrow$ Fase 2 (Bistro Western 0.90) $\\rightarrow$ Fase 3 (Seafood Pesisir 0.92)\n")
    md_lines.append("| Peringkat | Algoritma | Total Skor Kepuasan (Mean ± Std) | Cumulative Regret | % Arm Optimal Akhir |")
    md_lines.append("| :---: | :--- | :---: | :---: | :---: |")

    for rank, (name, a_res) in enumerate(sorted_agents, 1):
        rew = a_res.final_mean_reward
        rew_std = a_res.std_cumulative_rewards[-1]
        reg = a_res.final_mean_regret
        rate = a_res.final_optimal_rate

        print(f"{name:<36} | {rew:>10.2f} ± {rew_std:<4.1f} | {reg:>12.2f}       | {rate:>8.1f}%   ")
        md_lines.append(f"| {rank} | **{name}** | {rew:.2f} ± {rew_std:.2f} | {reg:.2f} | {rate:.1f}% |")

        summary_data["algorithms"][name] = {
            "rank": rank,
            "total_reward_mean": float(rew),
            "total_reward_std": float(rew_std),
            "cumulative_regret": float(reg),
            "final_optimal_rate": float(rate),
        }

    print("=" * 86)

    with open(os.path.join(output_dir, "non_stationary_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    with open(os.path.join(output_dir, "non_stationary_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\n[✓] Simulasi Non-Stationer Selesai! File tersimpan di: '{output_dir}'")
    return summary_data


if __name__ == "__main__":
    run_non_stationary_simulation()
