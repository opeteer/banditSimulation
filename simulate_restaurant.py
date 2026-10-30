"""
Simulasi Multi-Armed Bandit: Kasus Pemilihan 10 Restoran (1 Tahun, 2x Makan/Hari)
================================================================================
Skenario:
- Seorang pendatang (1 agen) tinggal di sebuah kota baru selama 1 tahun (365 hari).
- Setiap hari, agen makan 2 kali di restoran (makan siang & makan malam) -> Total 730 kali makan.
- Terdapat 10 pilihan restoran dengan tingkat kepuasan (reward) riil antara 0 sampai 1.
- Membandingkan performa antara:
  1. Thompson Sampling (Bayesian Posterior Sampling)
  2. Upper Confidence Bound (UCB1)
  (Dilengkapi Pure Random sebagai baseline kontrol dan Oracle sebagai batas ideal).
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

from bandit.environment import ContinuousBetaBandit
from bandit.agents import ThompsonSamplingAgent, UCB1Agent, RandomAgent
from bandit.simulator import Simulator


RESTAURANTS = [
    {"name": "Warung Makan Barokah (Prasmanan)", "mean": 0.48, "category": "Kaki Lima"},
    {"name": "Kantin Kampus Murah", "mean": 0.35, "category": "Kantin"},
    {"name": "Kedai Mie Ayam Mas Giri", "mean": 0.62, "category": "Kedai Tradisional"},
    {"name": "RM Padang Minang Asli [TERBAIK]", "mean": 0.88, "category": "Rumah Makan Padang"},
    {"name": "Bebek & Ayam Goreng Madura", "mean": 0.74, "category": "Khas Daerah"},
    {"name": "Bistro Western & Steakhouse", "mean": 0.70, "category": "Modern Casual"},
    {"name": "Fast Food Burger & Chicken", "mean": 0.52, "category": "Waralaba Cepat Saji"},
    {"name": "Kedai Ramen & Gyoza", "mean": 0.65, "category": "Kuliner Asia"},
    {"name": "Warung Pecel Lele Lamongan", "mean": 0.58, "category": "Tenda Malam"},
    {"name": "Restoran Seafood Pesisir", "mean": 0.78, "category": "Seafood Segar"},
]


def run_restaurant_simulation(
    days: int = 365,
    meals_per_day: int = 2,
    n_runs: int = 100,
    seed: int = 42,
    output_dir: str = "results/restaurant",
):
    os.makedirs(output_dir, exist_ok=True)
    total_steps = days * meals_per_day  # 730 steps
    n_arms = len(RESTAURANTS)
    means = [r["mean"] for r in RESTAURANTS]
    optimal_arm = int(np.argmax(means))
    optimal_reward = float(np.max(means))

    print("============================================================================")
    print("SIMULASI MULTI-ARMED BANDIT: KASUS PEMILIHAN 10 RESTORAN MAKAN")
    print(f"Durasi: {days} Hari ({meals_per_day}x makan/hari) -> Total {total_steps} Sesi Makan")
    print(f"Skala Kepuasan: 0.0 (Sangat Tidak Enak) sampai 1.0 (Sangatlah Enak)")
    print(f"Restoran Terbaik: Arm #{optimal_arm} ({RESTAURANTS[optimal_arm]['name']} - Skor Mean: {optimal_reward})")
    print("============================================================================")

    # 1. Setup Agen yang Dibandingkan
    agents = [
        ThompsonSamplingAgent(n_arms=n_arms, name="Thompson Sampling (Bayesian)"),
        UCB1Agent(n_arms=n_arms, c=1.0, name="UCB1 (c=1.0)"),
        UCB1Agent(n_arms=n_arms, c=0.707, name="UCB (c=√0.5)"),
        RandomAgent(n_arms=n_arms, name="Pure Random (Baseline)"),
    ]

    # Factory environment menggunakan ContinuousBetaBandit dengan rentang reward [0, 1]
    def env_factory(s: int) -> ContinuousBetaBandit:
        return ContinuousBetaBandit(means=means, precision=25.0, seed=s)

    simulator = Simulator(
        env_factory=env_factory,
        agents=agents,
        case_name=f"Pemilihan 10 Restoran ({days} Hari, 2x Makan/Hari)",
    )

    # Jalankan simulasi Monte Carlo
    print(f"\n[1] Menjalankan {n_runs} replikasi Monte Carlo...")
    results = simulator.run(n_steps=total_steps, n_runs=n_runs, seed=seed, verbose=True)

    # 2. Jalankan Single Agent Trajectory (1 Agen Riil selama 1 Tahun)
    print("\n[2] Menjalankan Trajektori 1 Agen Riil Selama 365 Hari (730 Makan)...")
    single_env = env_factory(777)
    single_ts = ThompsonSamplingAgent(n_arms=n_arms, name="Thompson Sampling")
    single_ucb = UCB1Agent(n_arms=n_arms, c=1.0, name="UCB1")

    journey_ts = []
    journey_ucb = []

    # Run single agent trajectory
    for step in range(total_steps):
        # TS
        arm_ts = single_ts.select_arm()
        rew_ts = single_env.step(arm_ts)
        single_ts.update(arm_ts, rew_ts)
        journey_ts.append({"step": step + 1, "day": (step // 2) + 1, "meal": "Siang" if (step % 2 == 0) else "Malam", "arm": arm_ts, "reward": rew_ts})

        # UCB (menggunakan sample env yang sama untuk perbandingan adil)
        arm_ucb = single_ucb.select_arm()
        rew_ucb = single_env.step(arm_ucb)
        single_ucb.update(arm_ucb, rew_ucb)
        journey_ucb.append({"step": step + 1, "day": (step // 2) + 1, "meal": "Siang" if (step % 2 == 0) else "Malam", "arm": arm_ucb, "reward": rew_ucb})

    # 3. Generate Visualisasi Grafik Khusus Kasus Restoran
    print("\n[3] Membuat visualisasi grafik hasil simulasi...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    colors = {
        "Thompson Sampling (Bayesian)": "#10b981",  # Hijau Emerald
        "UCB1 (c=1.0)": "#3b82f6",                 # Biru
        "UCB (c=√0.5)": "#8b5cf6",                 # Ungu
        "Pure Random (Baseline)": "#ef4444",       # Merah
    }

    days_axis = np.arange(1, total_steps + 1) / meals_per_day  # 0.5 sampai 365 hari

    # Grafik 1: Akumulasi Kepuasan (Cumulative Satisfaction / Reward)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, a_res in results.agent_results.items():
        c = colors.get(name, "#333333")
        mean = a_res.mean_cumulative_rewards
        std = a_res.std_cumulative_rewards / np.sqrt(n_runs)
        ax.plot(days_axis, mean, label=name, color=c, linewidth=2.2)
        ax.fill_between(days_axis, mean - 1.96 * std, mean + 1.96 * std, color=c, alpha=0.15)

    oracle_line = np.arange(1, total_steps + 1) * optimal_reward
    ax.plot(days_axis, oracle_line, label=f"Oracle / Restoran Terbaik ({RESTAURANTS[optimal_arm]['name']})", color="black", linestyle="--", linewidth=1.8, alpha=0.7)
    ax.set_title("Total Akumulasi Tingkat Kepuasan Makan Selama 365 Hari (730 Sesi)", fontweight="bold", pad=12)
    ax.set_xlabel("Hari ke- (1 sampai 365 Hari)")
    ax.set_ylabel("Total Skor Kepuasan Kumulatif")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    path_cum_rew = os.path.join(output_dir, "restaurant_cumulative_satisfaction.png")
    fig.tight_layout()
    fig.savefig(path_cum_rew, dpi=300)
    plt.close(fig)

    # Grafik 2: Akumulasi Penyesalan (Cumulative Regret)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, a_res in results.agent_results.items():
        c = colors.get(name, "#333333")
        mean = a_res.mean_cumulative_regrets
        std = a_res.std_cumulative_regrets / np.sqrt(n_runs)
        ax.plot(days_axis, mean, label=name, color=c, linewidth=2.2)
        ax.fill_between(days_axis, mean - 1.96 * std, mean + 1.96 * std, color=c, alpha=0.15)
    ax.set_title("Akumulasi Penyesalan (Cumulative Regret): Kehilangan Kepuasan Makan", fontweight="bold", pad=12)
    ax.set_xlabel("Hari ke- (1 sampai 365 Hari)")
    ax.set_ylabel("Total Cumulative Regret (Skor Kepuasan yang Hilang)")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    path_cum_reg = os.path.join(output_dir, "restaurant_cumulative_regret.png")
    fig.tight_layout()
    fig.savefig(path_cum_reg, dpi=300)
    plt.close(fig)

    # Grafik 3: Persentase Memilih Restoran Terbaik over 365 Hari
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, a_res in results.agent_results.items():
        c = colors.get(name, "#333333")
        rate = a_res.optimal_action_rate * 100.0
        # Smooth window 14 sesi (1 minggu makan)
        kernel = np.ones(14) / 14.0
        rate_smooth = np.convolve(rate, kernel, mode="same")
        rate_smooth[:14] = rate[:14]
        ax.plot(days_axis, rate_smooth, label=name, color=c, linewidth=2.2)
    ax.axhline(10.0, color="gray", linestyle=":", label="Peluang Acak Murni (10%)")
    ax.set_title("Tingkat Frekuensi Memilih Restoran Terbaik (%) Seiring Waktu", fontweight="bold", pad=12)
    ax.set_xlabel("Hari ke- (1 sampai 365 Hari)")
    ax.set_ylabel("Persentase Kunjungan ke RM Padang (Restoran Terbaik) (%)")
    ax.set_ylim(-2, 102)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    path_opt_rate = os.path.join(output_dir, "restaurant_optimal_choice_rate.png")
    fig.tight_layout()
    fig.savefig(path_opt_rate, dpi=300)
    plt.close(fig)

    # Grafik 4: Distribusi Kunjungan ke Masing-Masing Restoran (10 Restoran)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n_arms)
    width = 0.22

    # Plot UCB1 vs Thompson Sampling vs Random
    ts_pulls = results.agent_results["Thompson Sampling (Bayesian)"].mean_arm_pulls
    ucb1_pulls = results.agent_results["UCB1 (c=1.0)"].mean_arm_pulls
    ucb_tuned_pulls = results.agent_results["UCB (c=√0.5)"].mean_arm_pulls
    rand_pulls = results.agent_results["Pure Random (Baseline)"].mean_arm_pulls

    ax.bar(x - 1.5 * width, ts_pulls, width, label="Thompson Sampling (Bayesian)", color="#10b981", alpha=0.9)
    ax.bar(x - 0.5 * width, ucb1_pulls, width, label="UCB1 (c=1.0)", color="#3b82f6", alpha=0.9)
    ax.bar(x + 0.5 * width, ucb_tuned_pulls, width, label="UCB (c=√0.5)", color="#8b5cf6", alpha=0.9)
    ax.bar(x + 1.5 * width, rand_pulls, width, label="Pure Random", color="#ef4444", alpha=0.6)

    ax.set_title("Distribusi Total Kunjungan Makan ke 10 Restoran Selama 1 Tahun", fontweight="bold", pad=12)
    ax.set_xlabel("Pilihan Restoran (Skor Mean Kepuasan)")
    ax.set_ylabel("Rata-rata Jumlah Kunjungan Makan (Total = 730)")
    ax.set_xticks(x)
    labels = [f"{r['name']}\n(μ={r['mean']})" for r in RESTAURANTS]
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    path_visits = os.path.join(output_dir, "restaurant_visits_distribution.png")
    fig.tight_layout()
    fig.savefig(path_visits, dpi=300)
    plt.close(fig)

    # Grafik 5: Dashboard Komprehensif 4-in-1
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # A: Cumulative Rewards
    for name, a_res in results.agent_results.items():
        c = colors.get(name, "#333333")
        ax1.plot(days_axis, a_res.mean_cumulative_rewards, label=name, color=c, linewidth=2.0)
    ax1.plot(days_axis, oracle_line, label="Oracle (RM Padang)", color="black", linestyle="--", alpha=0.7)
    ax1.set_title("(A) Akumulasi Total Skor Kepuasan Makan", fontweight="bold")
    ax1.set_xlabel("Hari (1-365)")
    ax1.set_ylabel("Total Skor Kepuasan")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # B: Cumulative Regrets
    for name, a_res in results.agent_results.items():
        c = colors.get(name, "#333333")
        ax2.plot(days_axis, a_res.mean_cumulative_regrets, label=name, color=c, linewidth=2.0)
    ax2.set_title("(B) Akumulasi Penyesalan (Cumulative Regret)", fontweight="bold")
    ax2.set_xlabel("Hari (1-365)")
    ax2.set_ylabel("Skor Kepuasan yang Hilang")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # C: Optimal Restaurant Choice Rate
    for name, a_res in results.agent_results.items():
        c = colors.get(name, "#333333")
        rate = a_res.optimal_action_rate * 100.0
        kernel = np.ones(14) / 14.0
        smooth = np.convolve(rate, kernel, mode="same")
        smooth[:14] = rate[:14]
        ax3.plot(days_axis, smooth, label=name, color=c, linewidth=2.0)
    ax3.axhline(10.0, color="gray", linestyle=":", label="Random (10%)")
    ax3.set_title("(C) % Mengunjungi Restoran Terbaik (RM Padang)", fontweight="bold")
    ax3.set_xlabel("Hari (1-365)")
    ax3.set_ylabel("% Pilihan Restoran Terbaik")
    ax3.set_ylim(-2, 102)
    ax3.legend(loc="lower right", fontsize=8)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # D: Bar Comparison Thompson Sampling vs UCB1
    x_sub = np.arange(n_arms)
    w_sub = 0.35
    ax4.bar(x_sub - w_sub/2, ts_pulls, w_sub, label="Thompson Sampling", color="#10b981", alpha=0.9)
    ax4.bar(x_sub + w_sub/2, ucb1_pulls, w_sub, label="UCB1 (c=1.0)", color="#3b82f6", alpha=0.9)
    ax4.set_title("(D) Perbandingan Total Kunjungan: Thompson Sampling vs UCB1", fontweight="bold")
    ax4.set_xlabel("Restoran")
    ax4.set_ylabel("Jumlah Kunjungan Makan")
    ax4.set_xticks(x_sub)
    ax4.set_xticklabels([f"R#{i}\n({r['mean']})" for i, r in enumerate(RESTAURANTS)], fontsize=8)
    ax4.legend(loc="upper left", fontsize=8)
    ax4.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle(f"Hasil Evaluasi Multi-Armed Bandit: Kasus 10 Restoran (365 Hari, 730 Sesi Makan)", fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout()
    path_dashboard = os.path.join(output_dir, "restaurant_dashboard_comparison.png")
    fig.savefig(path_dashboard, dpi=300)
    plt.close(fig)

    # 4. Export JSON & Markdown Summary
    summary_data = {
        "case": "Kasus 10 Restoran Makan (1 Tahun, 2x Sehari = 730 Sesi)",
        "days": days,
        "meals_per_day": meals_per_day,
        "total_meals": total_steps,
        "n_runs": n_runs,
        "restaurants": RESTAURANTS,
        "optimal_restaurant": RESTAURANTS[optimal_arm],
        "algorithms": {}
    }

    print("\n" + "=" * 80)
    print(f"{'Algoritma':<28} | {'Total Kepuasan':<16} | {'Cumulative Regret':<18} | {'% Restoran Terbaik':<16}")
    print("-" * 80)

    md_lines = []
    md_lines.append(f"# Ringkasan Hasil Simulasi: Kasus 10 Restoran Makan (1 Tahun, 2x Makan/Hari)\n")
    md_lines.append(f"- **Subjek:** 1 Agen yang tinggal di kota selama {days} hari.")
    md_lines.append(f"- **Total Sesi Makan:** {total_steps} kali makan ({meals_per_day}x sehari: Siang & Malam).")
    md_lines.append(f"- **Restoran Terbaik:** {RESTAURANTS[optimal_arm]['name']} (Skor Rata-rata: {optimal_reward})\n")
    md_lines.append("| Algoritma | Total Skor Kepuasan (Mean ± Std) | Cumulative Regret | % Makan di RM Padang Terbaik | Total Kunjungan RM Padang |")
    md_lines.append("| :--- | :---: | :---: | :---: | :---: |")

    for name, a_res in sorted(results.agent_results.items(), key=lambda x: x[1].final_mean_regret):
        rew = a_res.final_mean_reward
        rew_std = a_res.std_cumulative_rewards[-1]
        reg = a_res.final_mean_regret
        rate = a_res.final_optimal_rate
        best_pulls = a_res.mean_arm_pulls[optimal_arm]

        print(f"{name:<28} | {rew:>10.2f} ± {rew_std:<4.1f} | {reg:>12.2f}       | {rate:>10.1f}%       ")
        md_lines.append(f"| **{name}** | {rew:.2f} ± {rew_std:.2f} | {reg:.2f} | {rate:.1f}% | {best_pulls:.1f} / {total_steps} ({best_pulls/total_steps*100:.1f}%) |")

        summary_data["algorithms"][name] = {
            "total_satisfaction_mean": float(rew),
            "total_satisfaction_std": float(rew_std),
            "cumulative_regret": float(reg),
            "optimal_restaurant_percent": float(rate),
            "mean_visits_per_restaurant": [float(p) for p in a_res.mean_arm_pulls],
        }

    print("=" * 80)

    # Save summary files
    summary_json_path = os.path.join(output_dir, "restaurant_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    summary_md_path = os.path.join(output_dir, "restaurant_summary.md")
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\n[✓] Grafik dan laporan berhasil disimpan di: '{output_dir}'")
    return summary_data


if __name__ == "__main__":
    run_restaurant_simulation(days=365, meals_per_day=2, n_runs=100)
