"""
Simulasi Multi-Armed Bandit: Kasus Pecinta Kuliner Porsi Besar (Poisson Capped at 7 Piring)
========================================================================================
Skenario:
- 1 Agen pecinta kuliner porsi besar tinggal di kota baru selama 2 TAHUN (730 hari).
- Makan di restoran 2 kali sehari (siang & malam) -> Total 1.460 kali makan.
- Terdapat 10 pilihan restoran dengan laju daya pikat piringan Poisson (lambda).
- Batas Kapasitas Fisik: Agen maksimal hanya bisa makan sampai 7 piring per sesi (Reward: min(7, X)).
- Algoritma yang dibandingkan:
  1. Gamma-Poisson Thompson Sampling
  2. Poisson-UCB (c = 1.0)
  3. Poisson-UCB (c = √2 ≈ 1.414)
  4. Pure Random (Baseline)
"""

import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson as sp_poisson

from bandit.environment import CappedPoissonBandit
from bandit.agents import GammaPoissonThompsonAgent, PoissonUCBAgent, RandomAgent
from bandit.simulator import Simulator


RESTAURANTS_POISSON = [
    {"name": "Warung Makan Barokah", "lambda": 2.0, "category": "Kaki Lima"},
    {"name": "Kantin Kampus Murah", "lambda": 1.0, "category": "Kantin"},
    {"name": "Kedai Mie Ayam Mas Giri", "lambda": 2.8, "category": "Tradisional"},
    {"name": "RM Padang Minang Asli [TERBAIK]", "lambda": 6.0, "category": "Rumah Makan Padang"},
    {"name": "Bebek & Ayam Madura", "lambda": 4.0, "category": "Khas Daerah"},
    {"name": "Bistro Western & Steakhouse", "lambda": 3.5, "category": "Modern Casual"},
    {"name": "Fast Food Burger & Chicken", "lambda": 1.8, "category": "Cepat Saji"},
    {"name": "Kedai Ramen & Gyoza", "lambda": 3.0, "category": "Kuliner Asia"},
    {"name": "Warung Pecel Lele Lamongan", "lambda": 2.4, "category": "Tenda Malam"},
    {"name": "Restoran Seafood Pesisir", "lambda": 5.0, "category": "Seafood Segar"},
]


def run_poisson_simulation(
    years: int = 2,
    days_per_year: int = 365,
    meals_per_day: int = 2,
    max_cap: int = 7,
    n_runs: int = 100,
    seed: int = 42,
    output_dir: str = "results/poisson_restaurant",
):
    os.makedirs(output_dir, exist_ok=True)
    total_days = years * days_per_year  # 730 hari
    total_steps = total_days * meals_per_day  # 1.460 sesi makan
    n_arms = len(RESTAURANTS_POISSON)
    lambdas = [r["lambda"] for r in RESTAURANTS_POISSON]

    # Hitung nilai ekspektasi capped eksak
    k_range = np.arange(0, 100)
    expected_rewards = []
    for lam in lambdas:
        probs = sp_poisson.pmf(k_range, lam)
        capped = np.minimum(max_cap, k_range)
        expected_rewards.append(float(np.sum(capped * probs)))
    expected_rewards = np.array(expected_rewards)

    optimal_arm = int(np.argmax(expected_rewards))
    optimal_reward = float(np.max(expected_rewards))

    print("============================================================================")
    print("SIMULASI MULTI-ARMED BANDIT: POISSON PORSI BESAR (MAKS 7 PIRING TAMBAH)")
    print(f"Durasi: {years} Tahun ({total_days} Hari, {meals_per_day}x makan/hari) -> Total {total_steps} Sesi Makan")
    print(f"Kapasitas Maksimal Perut: {max_cap} Piring Tambah per Sesi Makan")
    print(f"Restoran Terbaik: Arm #{optimal_arm} ({RESTAURANTS_POISSON[optimal_arm]['name']})")
    print(f"Laju Lambda Murni: {lambdas[optimal_arm]} -> Nilai Harapan Capped (Maks 7): {optimal_reward:.4f} piring/makan")
    print("============================================================================")

    # Inisialisasi Suite Agen Poisson
    agents = [
        GammaPoissonThompsonAgent(n_arms=n_arms, name="Gamma-Poisson Thompson Sampling"),
        PoissonUCBAgent(n_arms=n_arms, c=1.0, name="Poisson-UCB (c=1.0)"),
        PoissonUCBAgent(n_arms=n_arms, c=float(np.sqrt(2.0)), name="Poisson-UCB (c=√2 ≈ 1.41)"),
        RandomAgent(n_arms=n_arms, name="Pure Random (Baseline)"),
    ]

    def env_factory(s: int) -> CappedPoissonBandit:
        return CappedPoissonBandit(lambdas=lambdas, max_cap=max_cap, seed=s)

    simulator = Simulator(
        env_factory=env_factory,
        agents=agents,
        case_name=f"Poisson Capped ({max_cap} Piring, {years} Tahun)",
    )

    # 1. Jalankan Simulasi Monte Carlo
    print(f"\n[1] Menjalankan {n_runs} replikasi Monte Carlo ({total_steps} langkah)...")
    start_time = time.time()
    results = simulator.run(n_steps=total_steps, n_runs=n_runs, seed=seed, verbose=True)
    elapsed = time.time() - start_time
    print(f"    Selesai dalam {elapsed:.2f} detik.")

    # 2. Jalankan Trajektori 1 Agen Riil (Tahun 1 & Tahun 2)
    print("\n[2] Menjalankan Trajektori 1 Agen Riil Selama 2 Tahun...")
    single_env = env_factory(777)
    agent_ts = GammaPoissonThompsonAgent(n_arms=n_arms, name="Thompson Sampling")
    agent_ucb = PoissonUCBAgent(n_arms=n_arms, c=1.0, name="Poisson-UCB (c=1.0)")

    plates_history_ts = []
    plates_history_ucb = []

    for step in range(total_steps):
        # TS
        arm_ts = agent_ts.select_arm()
        plates_ts = single_env.step(arm_ts)
        agent_ts.update(arm_ts, plates_ts)
        plates_history_ts.append(plates_ts)

        # UCB
        arm_ucb = agent_ucb.select_arm()
        plates_ucb = single_env.step(arm_ucb)
        agent_ucb.update(arm_ucb, plates_ucb)
        plates_history_ucb.append(plates_ucb)

    # 3. Visualisasi Grafik
    print(f"\n[3] Membuat visualisasi grafik ke folder: '{output_dir}'...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    colors = {
        "Gamma-Poisson Thompson Sampling": "#10b981",  # Hijau Emerald
        "Poisson-UCB (c=1.0)": "#3b82f6",             # Biru
        "Poisson-UCB (c=√2 ≈ 1.41)": "#8b5cf6",         # Ungu
        "Pure Random (Baseline)": "#ef4444",           # Merah
    }

    days_axis = np.arange(1, total_steps + 1) / meals_per_day  # 0.5 sampai 730 hari

    # Grafik 1: Akumulasi Piring Tambah Selama 2 Tahun
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, a_res in results.agent_results.items():
        c = colors.get(name, "#333333")
        mean = a_res.mean_cumulative_rewards
        std = a_res.std_cumulative_rewards / np.sqrt(n_runs)
        ax.plot(days_axis, mean, label=name, color=c, linewidth=2.2)
        ax.fill_between(days_axis, mean - 1.96 * std, mean + 1.96 * std, color=c, alpha=0.12)

    oracle_line = np.arange(1, total_steps + 1) * optimal_reward
    ax.plot(days_axis, oracle_line, label=f"Oracle / RM Padang Terbaik (Maksimal Teoritis)", color="black", linestyle="--", linewidth=1.8, alpha=0.7)
    ax.set_title(f"Total Akumulasi Piring Tambah yang Dihabiskan Selama 2 Tahun (Maks {max_cap} Piring/Makan)", fontweight="bold", pad=12)
    ax.set_xlabel("Hari ke- (1 sampai 730 Hari)")
    ax.set_ylabel("Total Akumulasi Piring Tambah (Plates)")
    ax.set_xlim(0, 730)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    path_cum_plates = os.path.join(output_dir, "poisson_cumulative_plates_2yr.png")
    fig.tight_layout()
    fig.savefig(path_cum_plates, dpi=300)
    plt.close(fig)

    # Grafik 2: Akumulasi Penyesalan (Cumulative Regret Piring yang Hilang)
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, a_res in results.agent_results.items():
        c = colors.get(name, "#333333")
        mean = a_res.mean_cumulative_regrets
        std = a_res.std_cumulative_regrets / np.sqrt(n_runs)
        ax.plot(days_axis, mean, label=name, color=c, linewidth=2.2)
        ax.fill_between(days_axis, mean - 1.96 * std, mean + 1.96 * std, color=c, alpha=0.12)

    ax.set_title("Akumulasi Penyesalan (Cumulative Regret): Kehilangan Porsi Piring Nikmat", fontweight="bold", pad=12)
    ax.set_xlabel("Hari ke- (1 sampai 730 Hari)")
    ax.set_ylabel("Cumulative Expected Regret (Piring)")
    ax.set_xlim(0, 730)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    path_cum_reg = os.path.join(output_dir, "poisson_cumulative_regret_2yr.png")
    fig.tight_layout()
    fig.savefig(path_cum_reg, dpi=300)
    plt.close(fig)

    # Grafik 3: Persentase Makan di RM Padang (% Optimal Choice)
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, a_res in results.agent_results.items():
        c = colors.get(name, "#333333")
        rate = a_res.optimal_action_rate * 100.0
        kernel = np.ones(28) / 28.0
        rate_smooth = np.convolve(rate, kernel, mode="same")
        rate_smooth[:28] = rate[:28]
        ax.plot(days_axis, rate_smooth, label=name, color=c, linewidth=2.2)

    ax.axhline(10.0, color="gray", linestyle=":", label="Peluang Acak (10%)")
    ax.set_title("Tingkat Frekuensi Makan di RM Padang Terbaik (%) Selama 2 Tahun", fontweight="bold", pad=12)
    ax.set_xlabel("Hari ke- (1 sampai 730 Hari)")
    ax.set_ylabel("Persentase Kunjungan ke RM Padang (%)")
    ax.set_ylim(-2, 102)
    ax.set_xlim(0, 730)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    path_opt_rate = os.path.join(output_dir, "poisson_optimal_choice_rate_2yr.png")
    fig.tight_layout()
    fig.savefig(path_opt_rate, dpi=300)
    plt.close(fig)

    # Grafik 4: Distribusi Kunjungan ke 10 Restoran
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n_arms)
    width = 0.22

    ts_pulls = results.agent_results["Gamma-Poisson Thompson Sampling"].mean_arm_pulls
    ucb1_pulls = results.agent_results["Poisson-UCB (c=1.0)"].mean_arm_pulls
    ucb_sqrt2_pulls = results.agent_results["Poisson-UCB (c=√2 ≈ 1.41)"].mean_arm_pulls
    rand_pulls = results.agent_results["Pure Random (Baseline)"].mean_arm_pulls

    ax.bar(x - 1.5 * width, ts_pulls, width, label="Gamma-Poisson TS", color="#10b981", alpha=0.9)
    ax.bar(x - 0.5 * width, ucb1_pulls, width, label="Poisson-UCB (c=1.0)", color="#3b82f6", alpha=0.9)
    ax.bar(x + 0.5 * width, ucb_sqrt2_pulls, width, label="Poisson-UCB (c=√2)", color="#8b5cf6", alpha=0.9)
    ax.bar(x + 1.5 * width, rand_pulls, width, label="Pure Random", color="#ef4444", alpha=0.6)

    ax.set_title(f"Distribusi Total Kunjungan ke 10 Restoran Selama 2 Tahun (Total 1.460 Sesi)", fontweight="bold", pad=12)
    ax.set_xlabel("Pilihan Restoran")
    ax.set_ylabel("Rata-rata Jumlah Kunjungan Makan")
    ax.set_xticks(x)
    labels = [f"{r['name']}\n(λ={r['lambda']}, μ_cap={expected_rewards[i]:.2f})" for i, r in enumerate(RESTAURANTS_POISSON)]
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    path_visits = os.path.join(output_dir, "poisson_restaurant_visits_2yr.png")
    fig.tight_layout()
    fig.savefig(path_visits, dpi=300)
    plt.close(fig)

    # Grafik 5: Dashboard 4-in-1 Lengkap
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # A: Cumulative Plates
    for name, a_res in results.agent_results.items():
        c = colors.get(name, "#333333")
        ax1.plot(days_axis, a_res.mean_cumulative_rewards, label=name, color=c, linewidth=2.0)
    ax1.plot(days_axis, oracle_line, label="Oracle (RM Padang)", color="black", linestyle="--", alpha=0.7)
    ax1.set_title("(A) Total Piring Tambah yang Dihabiskan (2 Tahun)", fontweight="bold")
    ax1.set_xlabel("Hari (1-730)")
    ax1.set_ylabel("Total Piring Tambah")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # B: Cumulative Regrets
    for name, a_res in results.agent_results.items():
        c = colors.get(name, "#333333")
        ax2.plot(days_axis, a_res.mean_cumulative_regrets, label=name, color=c, linewidth=2.0)
    ax2.set_title("(B) Akumulasi Penyesalan Piring (Cumulative Regret)", fontweight="bold")
    ax2.set_xlabel("Hari (1-730)")
    ax2.set_ylabel("Piring Nikmat yang Hilang")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # C: Optimal Choice Rate
    for name, a_res in results.agent_results.items():
        c = colors.get(name, "#333333")
        rate = a_res.optimal_action_rate * 100.0
        kernel = np.ones(28) / 28.0
        smooth = np.convolve(rate, kernel, mode="same")
        smooth[:28] = rate[:28]
        ax3.plot(days_axis, smooth, label=name, color=c, linewidth=2.0)
    ax3.axhline(10.0, color="gray", linestyle=":", label="Random (10%)")
    ax3.set_title("(C) % Mengunjungi Restoran Terbaik (RM Padang)", fontweight="bold")
    ax3.set_xlabel("Hari (1-730)")
    ax3.set_ylabel("% Pilihan Restoran Terbaik")
    ax3.set_ylim(-2, 102)
    ax3.legend(loc="lower right", fontsize=8)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # D: Distribusi Piring Nyata yang Dimakan 1 Agen (Histogram 0 s/d 7 Piring)
    bins = np.arange(-0.5, 8.5, 1)
    ax4.hist(plates_history_ts, bins=bins, color="#10b981", alpha=0.7, rwidth=0.8, density=True)
    ax4.set_title(f"(D) Distribusi Riil Piring Tambah per Sesi (Maks {max_cap} Piring)", fontweight="bold")
    ax4.set_xlabel("Piring Tambah yang Dihabiskan per Sesi Makan")
    ax4.set_ylabel("Frekuensi Probabilitas Riil")
    ax4.set_xticks(range(8))
    ax4.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle("Hasil Evaluasi Multi-Armed Bandit: Kasus Pecinta Kuliner Porsi Besar (Poisson Capped 7 Piring, 2 Tahun)", fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout()
    path_dashboard = os.path.join(output_dir, "poisson_dashboard_2yr.png")
    fig.savefig(path_dashboard, dpi=300)
    plt.close(fig)

    # 4. Ringkasan & Ekspor Data
    summary_data = {
        "case": f"Kasus Porsi Besar Poisson (Maks {max_cap} Piring, {years} Tahun)",
        "years": years,
        "total_days": total_days,
        "total_meals": total_steps,
        "max_cap": max_cap,
        "n_runs": n_runs,
        "optimal_restaurant": RESTAURANTS_POISSON[optimal_arm],
        "expected_capped_rewards": expected_rewards.tolist(),
        "algorithms": {}
    }

    print("\n" + "=" * 84)
    print(f"{'Algoritma':<32} | {'Total Piring':<16} | {'Cumulative Regret':<18} | {'% RM Padang':<12}")
    print("-" * 84)

    md_lines = []
    md_lines.append(f"# Ringkasan Hasil Simulasi Poisson Porsi Besar (Maksimal 7 Piring, 2 Tahun)\n")
    md_lines.append(f"- **Durasi:** {years} Tahun ({total_days:,} Hari $\\times$ 2 makan/hari = {total_steps:,} Sesi Makan)")
    md_lines.append(f"- **Kapasitas Perut Maksimal:** {max_cap} Piring Tambah per Sesi Makan")
    md_lines.append(f"- **Restoran Terbaik:** {RESTAURANTS_POISSON[optimal_arm]['name']} (E[Piring] = {optimal_reward:.4f} piring/makan)\n")
    md_lines.append("| Algoritma | Total Piring Tambah (Mean ± Std) | Cumulative Regret (Piring) | % Kunjungan RM Padang | Kunjungan RM Padang (dari 1.460) |")
    md_lines.append("| :--- | :---: | :---: | :---: | :---: |")

    sorted_agents = sorted(results.agent_results.items(), key=lambda x: x[1].final_mean_regret)

    for name, a_res in sorted_agents:
        rew = a_res.final_mean_reward
        rew_std = a_res.std_cumulative_rewards[-1]
        reg = a_res.final_mean_regret
        rate = a_res.final_optimal_rate
        best_pulls = a_res.mean_arm_pulls[optimal_arm]

        print(f"{name:<32} | {rew:>10.1f} ± {rew_std:<4.1f} | {reg:>12.2f}       | {rate:>8.1f}%   ")
        md_lines.append(f"| **{name}** | {rew:.1f} ± {rew_std:.1f} | {reg:.2f} | {rate:.1f}% | {best_pulls:.1f} / {total_steps} ({best_pulls/total_steps*100:.1f}%) |")

        summary_data["algorithms"][name] = {
            "total_plates_mean": float(rew),
            "total_plates_std": float(rew_std),
            "cumulative_regret": float(reg),
            "optimal_restaurant_percent": float(rate),
            "mean_visits_per_restaurant": [float(p) for p in a_res.mean_arm_pulls],
        }

    print("=" * 84)

    # Simpan JSON dan Markdown
    with open(os.path.join(output_dir, "poisson_summary_2yr.json"), "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    with open(os.path.join(output_dir, "poisson_summary_2yr.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\n[✓] Simulasi Poisson selesai! File tersimpan di: '{output_dir}'")
    return summary_data


if __name__ == "__main__":
    run_poisson_simulation()
