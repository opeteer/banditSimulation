"""
Simulasi Multi-Armed Bandit: Kasus 10 Restoran Selama 10 Tahun (2x Makan/Hari = 7.300 Sesi)
========================================================================================
Skenario:
- 1 Agen tinggal di sebuah kota selama 10 TAHUN (3.650 hari).
- Makan di restoran 2 kali sehari (makan siang & makan malam) -> Total 7.300 kali makan.
- Terdapat 10 pilihan restoran dengan tingkat kepuasan (reward) riil r in [0, 1].
- Perbandingan utama:
  1. Thompson Sampling (Bayesian Posterior Sampling)
  2. UCB1 dengan c = sqrt(2) ≈ 1.4142
  (Dilengkapi UCB1 c=1.0 dan Pure Random sebagai pembanding komprehensif).
"""

import json
import os
import time
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


def run_10year_simulation(
    years: int = 10,
    days_per_year: int = 365,
    meals_per_day: int = 2,
    n_runs: int = 100,
    seed: int = 42,
    output_dir: str = "results/restaurant_10years",
):
    os.makedirs(output_dir, exist_ok=True)
    total_days = years * days_per_year  # 3.650 hari
    total_steps = total_days * meals_per_day  # 7.300 sesi makan
    n_arms = len(RESTAURANTS)
    means = [r["mean"] for r in RESTAURANTS]
    optimal_arm = int(np.argmax(means))
    optimal_reward = float(np.max(means))
    c_sqrt2 = float(np.sqrt(2.0))  # ≈ 1.4142

    print("============================================================================")
    print("SIMULASI MULTI-ARMED BANDIT: KASUS 10 RESTORAN SELAMA 10 TAHUN")
    print(f"Durasi: {years} Tahun ({total_days} Hari, {meals_per_day}x makan/hari) -> Total {total_steps} Sesi Makan")
    print(f"Skala Kepuasan: 0.0 (Sangat Tidak Enak) sampai 1.0 (Sangatlah Enak)")
    print(f"Algoritma Diuji: Thompson Sampling (Bayesian) vs UCB1 (c = √2 ≈ {c_sqrt2:.4f})")
    print(f"Restoran Terbaik: Arm #{optimal_arm} ({RESTAURANTS[optimal_arm]['name']} - Skor Mean: {optimal_reward})")
    print("============================================================================")

    # Inisialisasi Agen
    agents = [
        ThompsonSamplingAgent(n_arms=n_arms, name="Thompson Sampling (Bayesian)"),
        UCB1Agent(n_arms=n_arms, c=c_sqrt2, name=f"UCB1 (c=√2 ≈ {c_sqrt2:.2f})"),
        UCB1Agent(n_arms=n_arms, c=1.0, name="UCB1 (c=1.0)"),
        RandomAgent(n_arms=n_arms, name="Pure Random (Baseline)"),
    ]

    def env_factory(s: int) -> ContinuousBetaBandit:
        return ContinuousBetaBandit(means=means, precision=25.0, seed=s)

    simulator = Simulator(
        env_factory=env_factory,
        agents=agents,
        case_name=f"Pemilihan 10 Restoran (10 Tahun = 7.300 Sesi Makan)",
    )

    # 1. Jalankan Simulasi Monte Carlo
    print(f"\n[1] Menjalankan {n_runs} replikasi Monte Carlo (7.300 langkah per run)...")
    start_time = time.time()
    results = simulator.run(n_steps=total_steps, n_runs=n_runs, seed=seed, verbose=True)
    elapsed = time.time() - start_time
    print(f"    Selesai dalam {elapsed:.2f} detik.")

    # 2. Jalankan Trajektori 1 Agen Riil Sepanjang 10 Tahun (Breakdown per Tahun)
    print("\n[2] Menjalankan Trajektori 1 Agen Riil Sepanjang 10 Tahun (Year-by-Year)...")
    single_env = env_factory(777)
    agent_ts = ThompsonSamplingAgent(n_arms=n_arms, name="Thompson Sampling")
    agent_ucb = UCB1Agent(n_arms=n_arms, c=c_sqrt2, name="UCB1 (c=√2)")

    yearly_visits_ts = np.zeros((years, n_arms), dtype=int)
    yearly_visits_ucb = np.zeros((years, n_arms), dtype=int)
    yearly_satisfaction_ts = np.zeros(years, dtype=float)
    yearly_satisfaction_ucb = np.zeros(years, dtype=float)

    for step in range(total_steps):
        year_idx = step // (days_per_year * meals_per_day)  # 0 sampai 9

        # Thompson Sampling
        arm_ts = agent_ts.select_arm()
        rew_ts = single_env.step(arm_ts)
        agent_ts.update(arm_ts, rew_ts)
        yearly_visits_ts[year_idx, arm_ts] += 1
        yearly_satisfaction_ts[year_idx] += rew_ts

        # UCB1
        arm_ucb = agent_ucb.select_arm()
        rew_ucb = single_env.step(arm_ucb)
        agent_ucb.update(arm_ucb, rew_ucb)
        yearly_visits_ucb[year_idx, arm_ucb] += 1
        yearly_satisfaction_ucb[year_idx] += rew_ucb

    # 3. Visualisasi Grafik Komparatif 10 Tahun
    print("\n[3] Membuat grafik visualisasi 10 tahun ke folder: " + output_dir)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    colors = {
        "Thompson Sampling (Bayesian)": "#10b981",  # Hijau Emerald
        f"UCB1 (c=√2 ≈ {c_sqrt2:.2f})": "#2563eb",  # Biru Royal
        "UCB1 (c=1.0)": "#8b5cf6",                 # Ungu
        "Pure Random (Baseline)": "#ef4444",       # Merah
    }

    # Sumbu waktu: Tahun (0 sampai 10 tahun)
    years_axis = np.arange(1, total_steps + 1) / (days_per_year * meals_per_day)

    # Grafik 1: Akumulasi Kepuasan Selama 10 Tahun
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, a_res in results.agent_results.items():
        c = colors.get(name, "#333333")
        mean = a_res.mean_cumulative_rewards
        std = a_res.std_cumulative_rewards / np.sqrt(n_runs)
        ax.plot(years_axis, mean, label=name, color=c, linewidth=2.2)
        ax.fill_between(years_axis, mean - 1.96 * std, mean + 1.96 * std, color=c, alpha=0.12)

    oracle_line = np.arange(1, total_steps + 1) * optimal_reward
    ax.plot(years_axis, oracle_line, label=f"Oracle / RM Padang Terbaik (Maksimal Teoritis)", color="black", linestyle="--", linewidth=1.8, alpha=0.7)
    ax.set_title("Akumulasi Total Skor Kepuasan Makan Selama 10 Tahun (7.300 Sesi)", fontweight="bold", pad=12)
    ax.set_xlabel("Tahun ke- (1 sampai 10 Tahun)")
    ax.set_ylabel("Total Skor Kepuasan Kumulatif")
    ax.set_xlim(0, 10)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    path_cum_rew = os.path.join(output_dir, "restaurant_10yr_cumulative_satisfaction.png")
    fig.tight_layout()
    fig.savefig(path_cum_rew, dpi=300)
    plt.close(fig)

    # Grafik 2: Akumulasi Penyesalan (Cumulative Regret) 10 Tahun
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, a_res in results.agent_results.items():
        c = colors.get(name, "#333333")
        mean = a_res.mean_cumulative_regrets
        std = a_res.std_cumulative_regrets / np.sqrt(n_runs)
        ax.plot(years_axis, mean, label=name, color=c, linewidth=2.2)
        ax.fill_between(years_axis, mean - 1.96 * std, mean + 1.96 * std, color=c, alpha=0.12)

    ax.set_title("Akumulasi Penyesalan (Cumulative Regret) Selama 10 Tahun: Kehilangan Kepuasan Makan", fontweight="bold", pad=12)
    ax.set_xlabel("Tahun ke- (1 sampai 10 Tahun)")
    ax.set_ylabel("Cumulative Expected Regret")
    ax.set_xlim(0, 10)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    path_cum_reg = os.path.join(output_dir, "restaurant_10yr_cumulative_regret.png")
    fig.tight_layout()
    fig.savefig(path_cum_reg, dpi=300)
    plt.close(fig)

    # Grafik 3: Persentase Memilih RM Padang (Restoran Terbaik) over 10 Years
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, a_res in results.agent_results.items():
        c = colors.get(name, "#333333")
        rate = a_res.optimal_action_rate * 100.0
        # Rolling smooth window 60 sesi (1 bulan makan)
        kernel = np.ones(60) / 60.0
        rate_smooth = np.convolve(rate, kernel, mode="same")
        rate_smooth[:60] = rate[:60]
        ax.plot(years_axis, rate_smooth, label=name, color=c, linewidth=2.2)

    ax.axhline(10.0, color="gray", linestyle=":", label="Peluang Acak (10%)")
    ax.set_title("Tingkat Frekuensi Makan di RM Padang Terbaik (%) Selama 10 Tahun", fontweight="bold", pad=12)
    ax.set_xlabel("Tahun ke- (1 sampai 10 Tahun)")
    ax.set_ylabel("Persentase Kunjungan ke Restoran Terbaik (%)")
    ax.set_ylim(-2, 102)
    ax.set_xlim(0, 10)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    path_opt_rate = os.path.join(output_dir, "restaurant_10yr_optimal_choice_rate.png")
    fig.tight_layout()
    fig.savefig(path_opt_rate, dpi=300)
    plt.close(fig)

    # Grafik 4: Distribusi Kunjungan ke 10 Restoran Selama 10 Tahun
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n_arms)
    width = 0.22

    ts_pulls = results.agent_results["Thompson Sampling (Bayesian)"].mean_arm_pulls
    ucb_sqrt2_pulls = results.agent_results[f"UCB1 (c=√2 ≈ {c_sqrt2:.2f})"].mean_arm_pulls
    ucb1_pulls = results.agent_results["UCB1 (c=1.0)"].mean_arm_pulls
    rand_pulls = results.agent_results["Pure Random (Baseline)"].mean_arm_pulls

    ax.bar(x - 1.5 * width, ts_pulls, width, label="Thompson Sampling", color="#10b981", alpha=0.9)
    ax.bar(x - 0.5 * width, ucb_sqrt2_pulls, width, label=f"UCB1 (c=√2)", color="#2563eb", alpha=0.9)
    ax.bar(x + 0.5 * width, ucb1_pulls, width, label="UCB1 (c=1.0)", color="#8b5cf6", alpha=0.9)
    ax.bar(x + 1.5 * width, rand_pulls, width, label="Pure Random", color="#ef4444", alpha=0.6)

    ax.set_title("Distribusi Total Kunjungan ke 10 Restoran Selama 10 Tahun (Total = 7.300 Sesi)", fontweight="bold", pad=12)
    ax.set_xlabel("Pilihan Restoran")
    ax.set_ylabel("Rata-rata Jumlah Kunjungan Makan")
    ax.set_xticks(x)
    labels = [f"{r['name']}\n(μ={r['mean']})" for r in RESTAURANTS]
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    path_visits = os.path.join(output_dir, "restaurant_10yr_visits_distribution.png")
    fig.tight_layout()
    fig.savefig(path_visits, dpi=300)
    plt.close(fig)

    # Grafik 5: Dashboard Komprehensif 4-in-1
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # A: Cumulative Rewards
    for name, a_res in results.agent_results.items():
        c = colors.get(name, "#333333")
        ax1.plot(years_axis, a_res.mean_cumulative_rewards, label=name, color=c, linewidth=2.0)
    ax1.plot(years_axis, oracle_line, label="Oracle (RM Padang)", color="black", linestyle="--", alpha=0.7)
    ax1.set_title("(A) Akumulasi Total Skor Kepuasan (10 Tahun)", fontweight="bold")
    ax1.set_xlabel("Tahun (1-10)")
    ax1.set_ylabel("Total Kepuasan Kumulatif")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # B: Cumulative Regrets
    for name, a_res in results.agent_results.items():
        c = colors.get(name, "#333333")
        ax2.plot(years_axis, a_res.mean_cumulative_regrets, label=name, color=c, linewidth=2.0)
    ax2.set_title("(B) Akumulasi Penyesalan (Cumulative Regret)", fontweight="bold")
    ax2.set_xlabel("Tahun (1-10)")
    ax2.set_ylabel("Skor Kepuasan yang Hilang")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # C: Optimal Restaurant Choice Rate
    for name, a_res in results.agent_results.items():
        c = colors.get(name, "#333333")
        rate = a_res.optimal_action_rate * 100.0
        kernel = np.ones(60) / 60.0
        smooth = np.convolve(rate, kernel, mode="same")
        smooth[:60] = rate[:60]
        ax3.plot(years_axis, smooth, label=name, color=c, linewidth=2.0)
    ax3.axhline(10.0, color="gray", linestyle=":", label="Random (10%)")
    ax3.set_title("(C) % Mengunjungi Restoran Terbaik (RM Padang)", fontweight="bold")
    ax3.set_xlabel("Tahun (1-10)")
    ax3.set_ylabel("% Pilihan Restoran Terbaik")
    ax3.set_ylim(-2, 102)
    ax3.legend(loc="lower right", fontsize=8)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # D: Bar Comparison Thompson Sampling vs UCB1 c=√2
    x_sub = np.arange(n_arms)
    w_sub = 0.35
    ax4.bar(x_sub - w_sub/2, ts_pulls, w_sub, label="Thompson Sampling", color="#10b981", alpha=0.9)
    ax4.bar(x_sub + w_sub/2, ucb_sqrt2_pulls, w_sub, label=f"UCB1 (c=√2)", color="#2563eb", alpha=0.9)
    ax4.set_title(f"(D) Total Kunjungan: Thompson Sampling vs UCB1 (c=√2)", fontweight="bold")
    ax4.set_xlabel("Restoran")
    ax4.set_ylabel("Jumlah Kunjungan Makan (Total 7.300)")
    ax4.set_xticks(x_sub)
    ax4.set_xticklabels([f"R#{i}\n({r['mean']})" for i, r in enumerate(RESTAURANTS)], fontsize=8)
    ax4.legend(loc="upper left", fontsize=8)
    ax4.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle("Hasil Evaluasi Multi-Armed Bandit: Kasus 10 Restoran Selama 10 Tahun (7.300 Sesi Makan)", fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout()
    path_dashboard = os.path.join(output_dir, "restaurant_10yr_dashboard.png")
    fig.savefig(path_dashboard, dpi=300)
    plt.close(fig)

    # Grafik 6: Evolusi Kunjungan per Tahun (Year-by-Year Evolution)
    fig, (ax_ts, ax_ucb) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    year_labels = [f"Thn {y+1}" for y in range(years)]
    x_years = np.arange(years)

    # Ambil 3 kategori: Restoran Terbaik (R3 Padang), Restoran Runner-up (R9 Seafood & R4 Bebek), dan Restoran Lainnya (R0, R1, R2, R5, R6, R7, R8)
    best_ts = yearly_visits_ts[:, optimal_arm]
    runner_ts = yearly_visits_ts[:, 9] + yearly_visits_ts[:, 4]
    others_ts = 730 - best_ts - runner_ts

    best_ucb = yearly_visits_ucb[:, optimal_arm]
    runner_ucb = yearly_visits_ucb[:, 9] + yearly_visits_ucb[:, 4]
    others_ucb = 730 - best_ucb - runner_ucb

    # Plot TS
    ax_ts.bar(x_years, best_ts, label="RM Padang (Terbaik, μ=0.88)", color="#10b981")
    ax_ts.bar(x_years, runner_ts, bottom=best_ts, label="Runner-Up (Seafood & Bebek)", color="#3b82f6")
    ax_ts.bar(x_years, others_ts, bottom=best_ts + runner_ts, label="7 Restoran Lainnya", color="#94a3b8")
    ax_ts.set_title("Evolusi Pola Makan Tahunan: Thompson Sampling", fontweight="bold")
    ax_ts.set_xlabel("Tahun ke-")
    ax_ts.set_ylabel("Jumlah Makan per Tahun (Total = 730)")
    ax_ts.set_xticks(x_years)
    ax_ts.set_xticklabels(year_labels)
    ax_ts.legend(loc="upper left", fontsize=8)
    ax_ts.grid(axis="y", linestyle="--", alpha=0.5)

    # Plot UCB c=sqrt(2)
    ax_ucb.bar(x_years, best_ucb, label="RM Padang (Terbaik, μ=0.88)", color="#10b981")
    ax_ucb.bar(x_years, runner_ucb, bottom=best_ucb, label="Runner-Up (Seafood & Bebek)", color="#3b82f6")
    ax_ucb.bar(x_years, others_ucb, bottom=best_ucb + runner_ucb, label="7 Restoran Lainnya", color="#94a3b8")
    ax_ucb.set_title(f"Evolusi Pola Makan Tahunan: UCB1 (c=√2 ≈ {c_sqrt2:.2f})", fontweight="bold")
    ax_ucb.set_xlabel("Tahun ke-")
    ax_ucb.set_xticks(x_years)
    ax_ucb.set_xticklabels(year_labels)
    ax_ucb.legend(loc="upper left", fontsize=8)
    ax_ucb.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle("Dinamika Perubahan Kebiasaan Makan Agen dari Tahun 1 hingga Tahun 10", fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout()
    path_evolution = os.path.join(output_dir, "restaurant_10yr_yearly_evolution.png")
    fig.savefig(path_evolution, dpi=300)
    plt.close(fig)

    # 4. Ringkasan Numerik & Export
    summary_data = {
        "case": "Kasus 10 Restoran Makan (10 Tahun = 7.300 Sesi)",
        "years": years,
        "days_total": total_days,
        "meals_total": total_steps,
        "n_runs": n_runs,
        "c_sqrt2": c_sqrt2,
        "optimal_restaurant": RESTAURANTS[optimal_arm],
        "algorithms": {}
    }

    print("\n" + "=" * 84)
    print(f"{'Algoritma':<30} | {'Total Kepuasan':<16} | {'Cumulative Regret':<18} | {'% RM Padang':<12}")
    print("-" * 84)

    md_lines = []
    md_lines.append(f"# Ringkasan Hasil Simulasi 10 Restoran Selama 10 Tahun (7.300 Sesi Makan)\n")
    md_lines.append(f"- **Horizon:** {years} Tahun ({total_days:,} Hari $\\times$ 2 makan/hari = {total_steps:,} Sesi Makan)")
    md_lines.append(f"- **Restoran Terbaik:** {RESTAURANTS[optimal_arm]['name']} (Skor Rata-rata: {optimal_reward})\n")
    md_lines.append("| Algoritma | Total Skor Kepuasan (Mean ± Std) | Cumulative Regret | % Kunjungan RM Padang | Total Kunjungan RM Padang |")
    md_lines.append("| :--- | :---: | :---: | :---: | :---: |")

    sorted_agents = sorted(results.agent_results.items(), key=lambda x: x[1].final_mean_regret)

    for name, a_res in sorted_agents:
        rew = a_res.final_mean_reward
        rew_std = a_res.std_cumulative_rewards[-1]
        reg = a_res.final_mean_regret
        rate = a_res.final_optimal_rate
        best_pulls = a_res.mean_arm_pulls[optimal_arm]

        print(f"{name:<30} | {rew:>10.2f} ± {rew_std:<4.1f} | {reg:>12.2f}       | {rate:>8.1f}%   ")
        md_lines.append(f"| **{name}** | {rew:.2f} ± {rew_std:.2f} | {reg:.2f} | {rate:.1f}% | {best_pulls:.1f} / {total_steps} ({best_pulls/total_steps*100:.1f}%) |")

        summary_data["algorithms"][name] = {
            "total_satisfaction_mean": float(rew),
            "total_satisfaction_std": float(rew_std),
            "cumulative_regret": float(reg),
            "optimal_restaurant_percent": float(rate),
            "mean_visits_per_restaurant": [float(p) for p in a_res.mean_arm_pulls],
        }

    print("=" * 84)

    # Simpan JSON dan Markdown
    with open(os.path.join(output_dir, "restaurant_10yr_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    with open(os.path.join(output_dir, "restaurant_10yr_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\n[✓] Seluruh grafik dan data 10 tahun berhasil disimpan ke: '{output_dir}'")
    return summary_data


if __name__ == "__main__":
    run_10year_simulation()
