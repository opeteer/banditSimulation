"""
Script Eksekusi Utama Simulasi Multi-Armed Bandit
=================================================
Menyediakan antarmuka CLI yang kaya untuk menjalankan simulasi eksperimen bandit,
menghasilkan laporan metrik lengkap, dan memvisualisasikan grafik komparatif.

Contoh Penggunaan:
    python3 main.py --case ad-optimization --steps 2000 --runs 100
    python3 main.py --case clinical-trial --steps 1500 --runs 100
    python3 main.py --case non-stationary --steps 2000 --runs 100
    python3 main.py --case custom --custom-probs 0.05 0.15 0.08 0.22 0.03
"""

import argparse
import json
import os
import sys
from typing import Callable, List, Tuple
import numpy as np

from bandit.environment import BernoulliBandit, GaussianBandit, NonStationaryBandit, BanditEnvironment
from bandit.agents import (
    BanditAgent,
    RandomAgent,
    EpsilonGreedyAgent,
    DecayingEpsilonAgent,
    UCB1Agent,
    ThompsonSamplingAgent,
    OptimisticGreedyAgent,
)
from bandit.simulator import Simulator, SimulationResult
from bandit.visualizer import BanditVisualizer


def setup_ad_optimization_case() -> Tuple[str, Callable[[int], BanditEnvironment], List[str]]:
    """
    Kasus 1: Optimasi Kampanye Iklan Digital (Ad Creative CTR Optimization).
    5 variasi materi iklan (Banner A - E) dengan Click-Through Rate (CTR) tersembunyi.
    """
    case_name = "Optimasi CTR Iklan Digital (Ad Creative Optimization)"
    # Banner A (3.5%), Banner B (4.2%), Banner C (8.5%), Banner D (14.5% - Optimal), Banner E (5.0%)
    true_ctrs = [0.035, 0.042, 0.085, 0.145, 0.050]
    arm_labels = [
        "Banner A (CTR 3.5%)",
        "Banner B (CTR 4.2%)",
        "Banner C (CTR 8.5%)",
        "Banner D (CTR 14.5% [Optimal])",
        "Banner E (CTR 5.0%)",
    ]

    def factory(seed: int) -> BanditEnvironment:
        return BernoulliBandit(true_ctrs, seed=seed)

    return case_name, factory, arm_labels


def setup_clinical_trial_case() -> Tuple[str, Callable[[int], BanditEnvironment], List[str]]:
    """
    Kasus 2: Uji Klinis Terapi Medis (Adaptive Clinical Trials).
    4 protokol pengobatan dengan tingkat kesembuhan pasien yang belum diketahui.
    """
    case_name = "Uji Klinis Efektivitas Terapi Medis (Adaptive Clinical Trial)"
    # Treatment 1 (30%), Treatment 2 (45%), Treatment 3 (72% - Optimal), Treatment 4 (52%)
    efficacy_rates = [0.30, 0.45, 0.72, 0.52]
    arm_labels = [
        "Treatment Alpha (30%)",
        "Treatment Beta (45%)",
        "Treatment Gamma (72% [Optimal])",
        "Treatment Delta (52%)",
    ]

    def factory(seed: int) -> BanditEnvironment:
        return BernoulliBandit(efficacy_rates, seed=seed)

    return case_name, factory, arm_labels


def setup_non_stationary_case() -> Tuple[str, Callable[[int], BanditEnvironment], List[str]]:
    """
    Kasus 3: Non-Stationary Bandit (Kejenuhan Iklan / Ad Fatigue).
    Fase 1 (0-1000 langkah): Banner A menjadi primadona (12% vs 4%).
    Fase 2 (1001-2000 langkah): Terjadi ad fatigue, Banner C melonjak jadi primadona (16% vs 2%).
    """
    case_name = "Non-Stationary Bandit (Ad Fatigue & Pergeseran Tren Pasar)"
    phases = [
        {
            "step_until": 1000,
            "probabilities": [0.12, 0.05, 0.04, 0.02],
        },
        {
            "step_until": None,
            "probabilities": [0.03, 0.04, 0.16, 0.02],
        },
    ]
    arm_labels = [
        "Varian 1 (Fase 1: 12% -> Fase 2: 3%)",
        "Varian 2 (Fase 1: 5%  -> Fase 2: 4%)",
        "Varian 3 (Fase 1: 4%  -> Fase 2: 16% [Optimal Fase 2])",
        "Varian 4 (Fase 1: 2%  -> Fase 2: 2%)",
    ]

    def factory(seed: int) -> BanditEnvironment:
        return NonStationaryBandit(phases, seed=seed)

    return case_name, factory, arm_labels


def setup_custom_case(probs: List[float]) -> Tuple[str, Callable[[int], BanditEnvironment], List[str]]:
    """Kasus kustom dengan probabilitas yang ditentukan pengguna."""
    case_name = f"Kasus Kustom Bernoulli ({len(probs)} Arm)"
    arm_labels = [f"Arm #{i} (Prob {p:.3f})" for i, p in enumerate(probs)]

    def factory(seed: int) -> BanditEnvironment:
        return BernoulliBandit(probs, seed=seed)

    return case_name, factory, arm_labels


def create_agent_suite(n_arms: int) -> List[BanditAgent]:
    """Menginisialisasi seluruh suite algoritma bandit standar dan adaptif."""
    return [
        RandomAgent(n_arms=n_arms, name="Pure Random (Baseline)"),
        EpsilonGreedyAgent(n_arms=n_arms, epsilon=0.10, name="ε-Greedy (ε=0.10)"),
        EpsilonGreedyAgent(n_arms=n_arms, epsilon=0.05, name="ε-Greedy (ε=0.05)"),
        DecayingEpsilonAgent(
            n_arms=n_arms,
            initial_epsilon=1.0,
            decay_rate=0.005,
            min_epsilon=0.01,
            name="Decaying ε-Greedy",
        ),
        UCB1Agent(n_arms=n_arms, c=1.0, name="UCB1 (c=1.0)"),
        ThompsonSamplingAgent(n_arms=n_arms, name="Thompson Sampling (Bayesian)"),
        OptimisticGreedyAgent(n_arms=n_arms, initial_value=1.0, name="Optimistic Greedy (Q0=1.0)"),
    ]


def export_markdown_summary(result: SimulationResult, output_path: str) -> None:
    """Menulis ringkasan laporan hasil simulasi dalam format Markdown."""
    lines = []
    lines.append(f"# Laporan Hasil Simulasi Multi-Armed Bandit")
    lines.append(f"\n**Studi Kasus:** {result.case_name}")
    lines.append(f"- **Jumlah Arm (Pilihan):** {len(result.expected_rewards)}")
    lines.append(f"- **Langkah per Iterasi (Steps):** {result.n_steps:,}")
    lines.append(f"- **Jumlah Replikasi Monte Carlo (Runs):** {result.n_runs}")
    lines.append(f"- **Arm Optimal:** Arm #{result.optimal_arm} (Nilai Harapan: {result.optimal_reward:.4f})\n")

    lines.append("## Tabel Ringkasan Kinerja Algoritma\n")
    lines.append("| Peringkat | Algoritma | Total Reward (Mean ± Std) | Cumulative Regret | % Arm Optimal Akhir |")
    lines.append("| :---: | :--- | :---: | :---: | :---: |")

    sorted_agents = sorted(
        result.agent_results.items(),
        key=lambda item: item[1].final_mean_regret
    )

    for rank, (name, res) in enumerate(sorted_agents, 1):
        mean_rew = res.final_mean_reward
        std_rew = res.std_cumulative_rewards[-1]
        reg = res.final_mean_regret
        opt_rate = res.final_optimal_rate
        lines.append(f"| {rank} | **{name}** | {mean_rew:.2f} ± {std_rew:.2f} | {reg:.2f} | {opt_rate:.1f}% |")

    lines.append("\n## Detail Alokasi Penarikan Tuas (Arm Distribution)\n")
    lines.append("| Algoritma | " + " | ".join([f"Arm #{i}" for i in range(len(result.expected_rewards))]) + " |")
    lines.append("| :--- | " + " | ".join([":---:" for _ in range(len(result.expected_rewards))]) + " |")

    for name, res in sorted_agents:
        pull_strs = [f"{p:.1f}" for p in res.mean_arm_pulls]
        lines.append(f"| {name} | " + " | ".join(pull_strs) + " |")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def export_json_summary(result: SimulationResult, output_path: str) -> None:
    """Mengekspor metrik ringkasan ke JSON untuk kebutuhan visualisasi dan dashboard."""
    data = {
        "case_name": result.case_name,
        "n_steps": result.n_steps,
        "n_runs": result.n_runs,
        "expected_rewards": result.expected_rewards.tolist(),
        "optimal_arm": int(result.optimal_arm),
        "optimal_reward": float(result.optimal_reward),
        "algorithms": {},
    }

    # Urutkan berdasarkan regret terkecil
    sorted_agents = sorted(
        result.agent_results.items(),
        key=lambda item: item[1].final_mean_regret
    )

    # Simpan subsampel kurva waktu (misal 50 titik) agar ukuran JSON efisien
    sample_indices = np.linspace(0, result.n_steps - 1, 50, dtype=int)
    data["sampled_steps"] = (sample_indices + 1).tolist()

    for rank, (name, res) in enumerate(sorted_agents, 1):
        data["algorithms"][name] = {
            "rank": rank,
            "final_reward_mean": float(res.final_mean_reward),
            "final_reward_std": float(res.std_cumulative_rewards[-1]),
            "final_regret_mean": float(res.final_mean_regret),
            "final_regret_std": float(res.std_cumulative_regrets[-1]),
            "final_optimal_rate": float(res.final_optimal_rate),
            "mean_arm_pulls": [float(x) for x in res.mean_arm_pulls],
            "curve_cumulative_rewards": [float(res.mean_cumulative_rewards[i]) for i in sample_indices],
            "curve_cumulative_regrets": [float(res.mean_cumulative_regrets[i]) for i in sample_indices],
            "curve_optimal_rates": [float(res.optimal_action_rate[i] * 100.0) for i in sample_indices],
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Armed Bandit Simulation Engine & Visualizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--case",
        choices=["ad-optimization", "clinical-trial", "non-stationary", "custom"],
        default="ad-optimization",
        help="Skenario studi kasus yang disimulasikan",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2000,
        help="Jumlah langkah / ronde pengambilan keputusan (rounds)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Jumlah replikasi Monte Carlo untuk estimasi statistik yang stabil",
    )
    parser.add_argument(
        "--custom-probs",
        type=float,
        nargs="+",
        default=None,
        help="Daftar probabilitas arm untuk skenario custom (contoh: 0.1 0.4 0.8)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Direktori penyimpanan grafik dan file laporan",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed untuk reprodusibilitas",
    )

    args = parser.parse_args()

    # 1. Tentukan skenario kasus
    if args.case == "ad-optimization":
        case_name, env_factory, arm_labels = setup_ad_optimization_case()
    elif args.case == "clinical-trial":
        case_name, env_factory, arm_labels = setup_clinical_trial_case()
    elif args.case == "non-stationary":
        case_name, env_factory, arm_labels = setup_non_stationary_case()
    elif args.case == "custom":
        if not args.custom_probs:
            print("Kesalahan: Harap sediakan minimal 2 nilai probabilitas pada --custom-probs (contoh: --custom-probs 0.05 0.15 0.08)")
            sys.exit(1)
        case_name, env_factory, arm_labels = setup_custom_case(args.custom_probs)
    else:
        raise ValueError(f"Kasus tidak dikenali: {args.case}")

    sample_env = env_factory(args.seed)
    n_arms = sample_env.n_arms

    # 2. Siapkan algoritma agen yang akan diuji
    agents = create_agent_suite(n_arms)

    # 3. Jalankan simulasi Monte Carlo
    simulator = Simulator(
        env_factory=env_factory,
        agents=agents,
        case_name=case_name,
    )

    simulation_result = simulator.run(
        n_steps=args.steps,
        n_runs=args.runs,
        seed=args.seed,
        verbose=True,
    )

    # 4. Generate Visualisasi Grafik
    print(f"\n[+] Membuat grafik visualisasi ke folder: '{args.output_dir}'...")
    visualizer = BanditVisualizer(simulation_result, output_dir=args.output_dir)

    dashboard_file = visualizer.plot_full_dashboard("bandit_comparison_dashboard.png")
    rewards_file = visualizer.plot_cumulative_rewards(filename="cumulative_rewards.png")
    regrets_file = visualizer.plot_cumulative_regrets(filename="cumulative_regrets.png")
    optimal_rate_file = visualizer.plot_optimal_action_rate(filename="optimal_action_rate.png")
    allocations_file = visualizer.plot_arm_allocations(filename="arm_allocation_distribution.png")

    # 5. Export Ringkasan Laporan
    md_summary_file = os.path.join(args.output_dir, "simulation_summary.md")
    export_markdown_summary(simulation_result, md_summary_file)

    json_summary_file = os.path.join(args.output_dir, "simulation_summary.json")
    export_json_summary(simulation_result, json_summary_file)

    # 6. Cetak ringkasan ke konsol
    print("\n" + "=" * 76)
    print("RINGKASAN HASIL PERFORMA ALGORITMA MULTI-ARMED BANDIT")
    print("=" * 76)
    print(visualizer.get_summary_table_text())
    print("=" * 76)
    print(f"[✓] Semua grafik berhasil disimpan:")
    print(f"    - Dashboard Lengkap : {dashboard_file}")
    print(f"    - Akumulasi Reward  : {rewards_file}")
    print(f"    - Akumulasi Regret  : {regrets_file}")
    print(f"    - % Arm Optimal     : {optimal_rate_file}")
    print(f"    - Distribusi Arm    : {allocations_file}")
    print(f"[✓] Laporan teks disimpan ke: {md_summary_file}")
    print(f"[✓] Data JSON disimpan ke   : {json_summary_file}")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()
