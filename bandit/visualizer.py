"""
Modul Visualisasi Kinerja Multi-Armed Bandit
===========================================
Menghasilkan grafik komparatif berkualitas tinggi menggunakan Matplotlib:
1. Cumulative Reward over Time
2. Cumulative Regret over Time (Linear & Log scale)
3. Optimal Arm Selection Rate (%) over Time
4. Arm Pull Distribution per Algoritma
5. Dashboard Komprehensif 4-in-1
"""

import os
from typing import Optional, List
import numpy as np
import matplotlib.pyplot as plt

from .simulator import SimulationResult


class BanditVisualizer:
    """Kelas untuk membuat grafik analisis dan laporan visual simulasi bandit."""

    # Palet warna modern & berdaya kontras tinggi
    PALETTE = [
        "#1f77b4",  # Biru
        "#ff7f0e",  # Oranye
        "#2ca02c",  # Hijau
        "#d62728",  # Merah
        "#9467bd",  # Ungu
        "#8c564b",  # Cokelat
        "#e377c2",  # Pink
        "#7f7f7f",  # Abu-abu
    ]

    def __init__(self, result: SimulationResult, output_dir: str = "results"):
        self.result = result
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Konfigurasi style matplotlib
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 14,
            "figure.dpi": 150,
        })

    def plot_cumulative_rewards(
        self,
        show_confidence: bool = True,
        filename: str = "cumulative_rewards.png",
    ) -> str:
        """Plot akumulasi reward rata-rata seiring bertambahnya langkah (time steps)."""
        fig, ax = plt.subplots(figsize=(9, 5.5))
        steps = np.arange(1, self.result.n_steps + 1)

        for idx, (name, agent_res) in enumerate(self.result.agent_results.items()):
            color = self.PALETTE[idx % len(self.PALETTE)]
            mean = agent_res.mean_cumulative_rewards
            ax.plot(steps, mean, label=name, color=color, linewidth=2.0)

            if show_confidence and self.result.n_runs > 1:
                stderr = agent_res.std_cumulative_rewards / np.sqrt(self.result.n_runs)
                ax.fill_between(steps, mean - 1.96 * stderr, mean + 1.96 * stderr, color=color, alpha=0.15)

        # Plot garis teoritis optimal (Oracle / Omniscient agent)
        optimal_line = steps * self.result.optimal_reward
        ax.plot(steps, optimal_line, label=f"Oracle (Arm #{self.result.optimal_arm})", color="black", linestyle="--", linewidth=1.8, alpha=0.7)

        ax.set_title(f"Perbandingan Akumulasi Reward: {self.result.case_name}", fontweight="bold", pad=12)
        ax.set_xlabel("Langkah Waktu (Steps / Rounds)")
        ax.set_ylabel("Total Akumulasi Reward")
        ax.legend(loc="upper left", frameon=True, framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.6)

        filepath = os.path.join(self.output_dir, filename)
        fig.tight_layout()
        fig.savefig(filepath, dpi=300)
        plt.close(fig)
        return filepath

    def plot_cumulative_regrets(
        self,
        log_scale: bool = False,
        show_confidence: bool = True,
        filename: str = "cumulative_regrets.png",
    ) -> str:
        """Plot akumulasi penyesalan (cumulative regret)."""
        fig, ax = plt.subplots(figsize=(9, 5.5))
        steps = np.arange(1, self.result.n_steps + 1)

        for idx, (name, agent_res) in enumerate(self.result.agent_results.items()):
            color = self.PALETTE[idx % len(self.PALETTE)]
            mean = agent_res.mean_cumulative_regrets
            ax.plot(steps, mean, label=name, color=color, linewidth=2.0)

            if show_confidence and self.result.n_runs > 1:
                stderr = agent_res.std_cumulative_regrets / np.sqrt(self.result.n_runs)
                ax.fill_between(steps, mean - 1.96 * stderr, mean + 1.96 * stderr, color=color, alpha=0.15)

        scale_label = "(Log Scale)" if log_scale else "(Linear Scale)"
        ax.set_title(f"Perbandingan Akumulasi Regret {scale_label}: {self.result.case_name}", fontweight="bold", pad=12)
        ax.set_xlabel("Langkah Waktu (Steps / Rounds)")
        ax.set_ylabel("Cumulative Expected Regret")
        if log_scale:
            ax.set_yscale("log")
        ax.legend(loc="upper left", frameon=True, framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.6)

        filepath = os.path.join(self.output_dir, filename)
        fig.tight_layout()
        fig.savefig(filepath, dpi=300)
        plt.close(fig)
        return filepath

    def plot_optimal_action_rate(
        self,
        smoothing_window: int = 25,
        filename: str = "optimal_action_rate.png",
    ) -> str:
        """Plot persentase pemilihan arm optimal seiring waktu."""
        fig, ax = plt.subplots(figsize=(9, 5.5))
        steps = np.arange(1, self.result.n_steps + 1)

        for idx, (name, agent_res) in enumerate(self.result.agent_results.items()):
            color = self.PALETTE[idx % len(self.PALETTE)]
            rate = agent_res.optimal_action_rate * 100.0

            # Smoothing menggunakan moving average jika window > 1
            if smoothing_window > 1 and len(rate) > smoothing_window:
                kernel = np.ones(smoothing_window) / smoothing_window
                rate_smooth = np.convolve(rate, kernel, mode="same")
                # Jaga batas awal dan akhir
                rate_smooth[:smoothing_window] = rate[:smoothing_window]
                rate_smooth[-smoothing_window:] = rate[-smoothing_window:]
                ax.plot(steps, rate_smooth, label=name, color=color, linewidth=2.0)
            else:
                ax.plot(steps, rate, label=name, color=color, linewidth=1.8, alpha=0.85)

        # Baseline probabilitas acak seragam
        random_prob = (1.0 / len(self.result.expected_rewards)) * 100.0
        ax.axhline(random_prob, color="gray", linestyle=":", label=f"Random Chance ({random_prob:.1f}%)")

        ax.set_title(f"Tingkat Pemilihan Arm Optimal (%) Seiring Waktu", fontweight="bold", pad=12)
        ax.set_xlabel("Langkah Waktu (Steps / Rounds)")
        ax.set_ylabel("Persentase Pemilihan Arm Terbaik (%)")
        ax.set_ylim(-2, 102)
        ax.legend(loc="lower right", frameon=True, framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.6)

        filepath = os.path.join(self.output_dir, filename)
        fig.tight_layout()
        fig.savefig(filepath, dpi=300)
        plt.close(fig)
        return filepath

    def plot_arm_allocations(
        self,
        filename: str = "arm_allocation_distribution.png",
    ) -> str:
        """Plot distribusi alokasi tarikan arm (pull count) per algoritma."""
        n_arms = len(self.result.expected_rewards)
        agents = list(self.result.agent_results.keys())
        n_agents = len(agents)

        fig, ax = plt.subplots(figsize=(10, 5.5))
        x = np.arange(n_agents)
        width = 0.8 / n_arms

        for arm_idx in range(n_arms):
            arm_pulls = [
                self.result.agent_results[ag].mean_arm_pulls[arm_idx]
                for ag in agents
            ]
            exp_val = self.result.expected_rewards[arm_idx]
            is_best = (arm_idx == self.result.optimal_arm)
            label = f"Arm #{arm_idx} (μ={exp_val:.3f}){' [TERBAIK]' if is_best else ''}"
            offset = (arm_idx - (n_arms - 1) / 2) * width
            ax.bar(x + offset, arm_pulls, width, label=label, alpha=0.9 if is_best else 0.75)

        ax.set_title(f"Distribusi Total Alokasi Penarikan Arm per Algoritma", fontweight="bold", pad=12)
        ax.set_xlabel("Algoritma Bandit")
        ax.set_ylabel("Rata-rata Jumlah Penarikan (Pulls)")
        ax.set_xticks(x)
        ax.set_xticklabels(agents, rotation=15, ha="right")
        ax.legend(title="Pilihan Arm", loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        filepath = os.path.join(self.output_dir, filename)
        fig.tight_layout()
        fig.savefig(filepath, dpi=300)
        plt.close(fig)
        return filepath

    def plot_full_dashboard(
        self,
        filename: str = "bandit_comparison_dashboard.png",
    ) -> str:
        """Membuat visualisasi dashboard komprehensif 4 panel dalam satu gambar."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        steps = np.arange(1, self.result.n_steps + 1)
        agents = list(self.result.agent_results.keys())

        # 1. Panel: Cumulative Rewards
        for idx, (name, agent_res) in enumerate(self.result.agent_results.items()):
            color = self.PALETTE[idx % len(self.PALETTE)]
            ax1.plot(steps, agent_res.mean_cumulative_rewards, label=name, color=color, linewidth=2.0)
        optimal_line = steps * self.result.optimal_reward
        ax1.plot(steps, optimal_line, label="Oracle (Ideal)", color="black", linestyle="--", alpha=0.7)
        ax1.set_title("(A) Akumulasi Reward Rata-rata", fontweight="bold")
        ax1.set_xlabel("Langkah (Steps)")
        ax1.set_ylabel("Total Reward")
        ax1.legend(loc="upper left", fontsize=8)
        ax1.grid(True, linestyle="--", alpha=0.5)

        # 2. Panel: Cumulative Regrets
        for idx, (name, agent_res) in enumerate(self.result.agent_results.items()):
            color = self.PALETTE[idx % len(self.PALETTE)]
            ax2.plot(steps, agent_res.mean_cumulative_regrets, label=name, color=color, linewidth=2.0)
        ax2.set_title("(B) Akumulasi Expected Regret (Penyesalan)", fontweight="bold")
        ax2.set_xlabel("Langkah (Steps)")
        ax2.set_ylabel("Total Regret")
        ax2.legend(loc="upper left", fontsize=8)
        ax2.grid(True, linestyle="--", alpha=0.5)

        # 3. Panel: Optimal Arm Selection Rate
        for idx, (name, agent_res) in enumerate(self.result.agent_results.items()):
            color = self.PALETTE[idx % len(self.PALETTE)]
            rate = agent_res.optimal_action_rate * 100.0
            # Rolling smooth
            kernel = np.ones(15) / 15.0
            rate_smooth = np.convolve(rate, kernel, mode="same")
            rate_smooth[:15] = rate[:15]
            ax3.plot(steps, rate_smooth, label=name, color=color, linewidth=2.0)
        random_prob = (1.0 / len(self.result.expected_rewards)) * 100.0
        ax3.axhline(random_prob, color="gray", linestyle=":", label=f"Random ({random_prob:.1f}%)")
        ax3.set_title("(C) % Pemilihan Arm Terbaik Seiring Waktu", fontweight="bold")
        ax3.set_xlabel("Langkah (Steps)")
        ax3.set_ylabel("% Arm Optimal")
        ax3.set_ylim(-2, 102)
        ax3.legend(loc="lower right", fontsize=8)
        ax3.grid(True, linestyle="--", alpha=0.5)

        # 4. Panel: Arm Allocation
        n_arms = len(self.result.expected_rewards)
        n_agents = len(agents)
        x = np.arange(n_agents)
        width = 0.8 / n_arms
        for arm_idx in range(n_arms):
            arm_pulls = [
                self.result.agent_results[ag].mean_arm_pulls[arm_idx]
                for ag in agents
            ]
            exp_val = self.result.expected_rewards[arm_idx]
            is_best = (arm_idx == self.result.optimal_arm)
            label = f"Arm #{arm_idx} (μ={exp_val:.3f}){' [BEST]' if is_best else ''}"
            offset = (arm_idx - (n_arms - 1) / 2) * width
            ax4.bar(x + offset, arm_pulls, width, label=label, alpha=0.9 if is_best else 0.7)
        ax4.set_title("(D) Distribusi Penarikan Tuas (Arm Pulls)", fontweight="bold")
        ax4.set_xlabel("Algoritma")
        ax4.set_ylabel("Rata-rata Tarikan")
        ax4.set_xticks(x)
        ax4.set_xticklabels(agents, rotation=20, ha="right", fontsize=8)
        ax4.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
        ax4.grid(axis="y", linestyle="--", alpha=0.5)

        fig.suptitle(f"Hasil Evaluasi Multi-Armed Bandit: {self.result.case_name} ({self.result.n_runs} Monte Carlo Runs)", fontsize=15, fontweight="bold", y=0.995)
        fig.tight_layout()

        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=300)
        plt.close(fig)
        return filepath

    def get_summary_table_text(self) -> str:
        """Menghasilkan teks tabel ringkasan perbandingan metrik akhir."""
        lines = []
        lines.append(f"{'Algoritma':<28} | {'Total Reward':<14} | {'Total Regret':<14} | {'% Arm Optimal':<14}")
        lines.append("-" * 76)

        sorted_agents = sorted(
            self.result.agent_results.items(),
            key=lambda item: item[1].final_mean_regret
        )

        for name, res in sorted_agents:
            lines.append(
                f"{name:<28} | "
                f"{res.final_mean_reward:>10.2f}     | "
                f"{res.final_mean_regret:>10.2f}     | "
                f"{res.final_optimal_rate:>10.1f}%    "
            )
        return "\n".join(lines)
