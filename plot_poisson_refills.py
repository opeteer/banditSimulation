"""
Visualisasi Distribusi Poisson untuk Kasus Porsi Besar (Piring Tambah / Refills)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

RESTAURANTS_POISSON = [
    {"name": "RM Padang Minang Asli [TERBAIK]", "lambda": 6.0, "color": "#10b981"},
    {"name": "Restoran Seafood Pesisir", "lambda": 5.0, "color": "#3b82f6"},
    {"name": "Bebek & Ayam Madura", "lambda": 4.0, "color": "#8b5cf6"},
    {"name": "Kedai Ramen & Gyoza", "lambda": 3.0, "color": "#f59e0b"},
    {"name": "Warung Makan Barokah", "lambda": 2.0, "color": "#64748b"},
    {"name": "Kantin Kampus Murah [TERBURUK]", "lambda": 1.0, "color": "#ef4444"},
]

def generate_poisson_plot(output_dir: str = "results"):
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "figure.titlesize": 14,
        "figure.dpi": 150,
    })

    k_values = np.arange(0, 15)  # 0 sampai 14 piring tambah

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: PMF Kurva Garis & Titik
    for r in RESTAURANTS_POISSON:
        lam = r["lambda"]
        pmf_vals = poisson.pmf(k_values, lam)
        is_best = (lam == 6.0)
        is_worst = (lam == 1.0)
        lw = 2.8 if (is_best or is_worst) else 1.8
        alpha = 1.0 if (is_best or is_worst) else 0.8
        ax1.plot(k_values, pmf_vals, marker="o", markersize=5, label=f"{r['name']} (λ={lam:.1f})", color=r["color"], linewidth=lw, alpha=alpha)

    ax1.set_title("Probabilitas Jumlah Piring Tambah: P(X = k)", fontweight="bold")
    ax1.set_xlabel("Jumlah Piring Tambah yang Dihabiskan (k)")
    ax1.set_ylabel("Probabilitas P(X = k)")
    ax1.set_xticks(k_values)
    ax1.legend(loc="upper right", fontsize=8.5)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Plot 2: Perbandingan Batang Diskret RM Padang (λ=6.0) vs Kantin Kampus (λ=1.0)
    width = 0.38
    pmf_padang = poisson.pmf(k_values, 6.0)
    pmf_kantin = poisson.pmf(k_values, 1.0)

    ax2.bar(k_values - width/2, pmf_padang, width, label="RM Padang (λ=6.0 piring) [Makan Lahap]", color="#10b981", alpha=0.9)
    ax2.bar(k_values + width/2, pmf_kantin, width, label="Kantin Kampus (λ=1.0 piring) [Sering Enek]", color="#ef4444", alpha=0.85)

    # Anotasi titik puncak (modus)
    ax2.annotate("Puncak Kantin (k=1)\nP=36.8%", xy=(1, pmf_kantin[1]), xytext=(1.5, 0.40),
                 arrowprops=dict(facecolor="#ef4444", shrink=0.08, width=1.5, headwidth=6),
                 fontsize=9, fontweight="bold", color="#ef4444")

    ax2.annotate("Puncak RM Padang (k=6)\nP=16.1%", xy=(6, pmf_padang[6]), xytext=(7.5, 0.22),
                 arrowprops=dict(facecolor="#10b981", shrink=0.08, width=1.5, headwidth=6),
                 fontsize=9, fontweight="bold", color="#10b981")

    ax2.set_title("Perbandingan Distribusi Probabilitas: RM Padang vs Kantin Kampus", fontweight="bold")
    ax2.set_xlabel("Jumlah Piring Tambah (k)")
    ax2.set_ylabel("Peluang Terjadi P(X = k)")
    ax2.set_xticks(k_values)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("Detail Distribusi Poisson Kasus Pecinta Kuliner Porsi Besar (Jumlah Piring Tambah)", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()

    out_file = os.path.join(output_dir, "poisson_refills_pmf.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"[✓] Grafik Poisson berhasil disimpan: {out_file}")
    return out_file

if __name__ == "__main__":
    generate_poisson_plot()
