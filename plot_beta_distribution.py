"""
Script Visualisasi Distribusi Beta Kontinu untuk Kasus Restoran
==============================================================
Menghasilkan visualisasi:
1. Kurva Probability Density Function (PDF) untuk ke-10 Restoran di [0, 1].
2. Perbandingan Pengaruh Presisi Koki (kappa = 5, 25, 100).
3. Histogram Sampel Acak Riil Sesi Makan vs Kurva Teoretis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist

RESTAURANTS = [
    {"name": "Warung Barokah", "mean": 0.48, "color": "#64748b"},
    {"name": "Kantin Kampus", "mean": 0.35, "color": "#ef4444"},
    {"name": "Mie Ayam Mas Giri", "mean": 0.62, "color": "#f59e0b"},
    {"name": "RM Padang Minang [BEST]", "mean": 0.88, "color": "#10b981"},
    {"name": "Bebek Madura", "mean": 0.74, "color": "#8b5cf6"},
    {"name": "Bistro Western", "mean": 0.70, "color": "#06b6d4"},
    {"name": "Fast Food Burger", "mean": 0.52, "color": "#a855f7"},
    {"name": "Kedai Ramen", "mean": 0.65, "color": "#f97316"},
    {"name": "Pecel Lele", "mean": 0.58, "color": "#84cc16"},
    {"name": "Seafood Pesisir", "mean": 0.78, "color": "#3b82f6"},
]

def generate_beta_visualizations(output_dir: str = "results"):
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

    x = np.linspace(0.001, 0.999, 1000)

    # -------------------------------------------------------------
    # Gambar 1: Kurva PDF dari 10 Restoran (Presisi kappa = 25)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    precision = 25.0

    for r in RESTAURANTS:
        mu = r["mean"]
        a = mu * precision
        b = (1.0 - mu) * precision
        y = beta_dist.pdf(x, a, b)
        is_best = (mu == 0.88)
        is_worst = (mu == 0.35)
        lw = 3.0 if (is_best or is_worst) else 1.8
        alpha = 1.0 if (is_best or is_worst) else 0.75
        label = f"{r['name']} (μ={mu:.2f}, α={a:.1f}, β={b:.1f})"
        ax.plot(x, y, label=label, color=r["color"], linewidth=lw, alpha=alpha)
        if is_best or is_worst:
            ax.fill_between(x, y, color=r["color"], alpha=0.15)

    ax.set_title("Kurva Kepadatan Probabilitas (PDF) Distribusi Beta Kontinu untuk 10 Restoran", fontweight="bold", pad=12)
    ax.set_xlabel("Tingkat Kepuasan Rasa (0.0 = Sangat Tidak Enak s/d 1.0 = Sangatlah Enak)")
    ax.set_ylabel("Kepadatan Probabilitas (Density)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True, fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.6)

    path_10resto = os.path.join(output_dir, "beta_distribution_10_restaurants.png")
    fig.tight_layout()
    fig.savefig(path_10resto, dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # Gambar 2: Panel 4-in-1 Lengkap Analisis Distribusi Beta
    # -------------------------------------------------------------
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 11))

    # Panel A: Spektrum Cita Rasa Utama (Buruk, Sedang, Enak, Terbaik)
    selected = [
        {"name": "Kantin Kampus (Buruk, Positively Skewed)", "mu": 0.35, "color": "#ef4444"},
        {"name": "Warung Barokah (Sedang, Simetris)", "mu": 0.50, "color": "#64748b"},
        {"name": "Bebek Madura (Enak, Moderat)", "mu": 0.74, "color": "#8b5cf6"},
        {"name": "RM Padang (Terbaik, Negatively Skewed)", "mu": 0.88, "color": "#10b981"},
    ]
    for s in selected:
        a = s["mu"] * precision
        b = (1.0 - s["mu"]) * precision
        y = beta_dist.pdf(x, a, b)
        ax1.plot(x, y, label=f"{s['name']}\nα={a:.1f}, β={b:.1f}", color=s["color"], linewidth=2.5)
        ax1.fill_between(x, y, color=s["color"], alpha=0.15)
    ax1.set_title("(A) Ragam Asimetri Cita Rasa: Buruk vs Menengah vs Terbaik", fontweight="bold")
    ax1.set_xlabel("Tingkat Kepuasan [0.0 - 1.0]")
    ax1.set_ylabel("Kepadatan Probabilitas")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel B: Pengaruh Parameter Presisi (Konsistensi Koki RM Padang, mu = 0.88)
    kappas = [
        {"k": 5.0, "label": "Presisi Rendah (κ=5): Koki Tidak Konsisten", "color": "#f59e0b", "ls": "--"},
        {"k": 15.0, "label": "Presisi Sedang (κ=15): Konsistensi Standar", "color": "#3b82f6", "ls": "-."},
        {"k": 25.0, "label": "Presisi Simulasi (κ=25): Masakan Rumah Enak", "color": "#10b981", "ls": "-"},
        {"k": 75.0, "label": "Presisi Tinggi (κ=75): Standar Resto Bintang 5", "color": "#059669", "ls": ":"},
    ]
    mu_padang = 0.88
    for kp in kappas:
        a = mu_padang * kp["k"]
        b = (1.0 - mu_padang) * kp["k"]
        y = beta_dist.pdf(x, a, b)
        ax2.plot(x, y, label=kp["label"], color=kp["color"], linestyle=kp["ls"], linewidth=2.2)
    ax2.set_title("(B) Pengaruh Presisi Koki (Konsistensi Rasa di RM Padang, μ=0.88)", fontweight="bold")
    ax2.set_xlabel("Tingkat Kepuasan [0.0 - 1.0]")
    ax2.set_ylabel("Kepadatan Probabilitas")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Panel C: Histogram 1.000 Sampel Riil Sesi Makan vs Kurva Teoretis
    rng = np.random.default_rng(42)
    samples_padang = rng.beta(0.88 * precision, (1 - 0.88) * precision, size=1000)
    samples_kantin = rng.beta(0.35 * precision, (1 - 0.35) * precision, size=1000)

    ax3.hist(samples_padang, bins=35, density=True, alpha=0.5, color="#10b981", label="1.000 Makan di RM Padang (Sampel)")
    ax3.hist(samples_kantin, bins=35, density=True, alpha=0.5, color="#ef4444", label="1.000 Makan di Kantin Kampus (Sampel)")

    y_padang = beta_dist.pdf(x, 0.88 * precision, (1 - 0.88) * precision)
    y_kantin = beta_dist.pdf(x, 0.35 * precision, (1 - 0.35) * precision)
    ax3.plot(x, y_padang, color="#047857", linewidth=2.5, label="PDF Teoretis RM Padang")
    ax3.plot(x, y_kantin, color="#b91c1c", linewidth=2.5, label="PDF Teoretis Kantin")

    ax3.set_title("(C) Sampel Stokastik Riil 1.000 Sesi Makan vs Kurva Teoretis", fontweight="bold")
    ax3.set_xlabel("Tingkat Kepuasan [0.0 - 1.0]")
    ax3.set_ylabel("Frekuensi Densitas Relatif")
    ax3.legend(loc="upper center", fontsize=8)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # Panel D: Kurva Kumulatif (CDF - Cumulative Distribution Function)
    for s in selected:
        a = s["mu"] * precision
        b = (1.0 - s["mu"]) * precision
        cdf_y = beta_dist.cdf(x, a, b)
        ax4.plot(x, cdf_y, label=s["name"].split(" (")[0], color=s["color"], linewidth=2.2)

    ax4.axvline(0.8, color="gray", linestyle=":", label="Batas Sangat Enak (x = 0.8)")
    ax4.set_title("(D) Cumulative Distribution Function (Peluang Mendapat Skor ≤ x)", fontweight="bold")
    ax4.set_xlabel("Ambang Skor Kepuasan (x)")
    ax4.set_ylabel("Peluang Kumulatif P(X ≤ x)")
    ax4.legend(loc="upper left", fontsize=8)
    ax4.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("Visualisasi Komprehensif Distribusi Beta Kontinu dalam Pemodelan Kepuasan Kuliner", fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout()
    path_dashboard = os.path.join(output_dir, "beta_distribution_comprehensive.png")
    fig.savefig(path_dashboard, dpi=300)
    plt.close(fig)

    print(f"[✓] Visualisasi berhasil dibuat:")
    print(f"    - Kurva 10 Restoran: {path_10resto}")
    print(f"    - Dashboard 4-Panel: {path_dashboard}")
    return path_10resto, path_dashboard

if __name__ == "__main__":
    generate_beta_visualizations()
