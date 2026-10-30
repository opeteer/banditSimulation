# Laporan Hasil Simulasi Multi-Armed Bandit

**Studi Kasus:** Optimasi CTR Iklan Digital (Ad Creative Optimization)
- **Jumlah Arm (Pilihan):** 5
- **Langkah per Iterasi (Steps):** 2,000
- **Jumlah Replikasi Monte Carlo (Runs):** 100
- **Arm Optimal:** Arm #3 (Nilai Harapan: 0.1450)

## Tabel Ringkasan Kinerja Algoritma

| Peringkat | Algoritma | Total Reward (Mean ± Std) | Cumulative Regret | % Arm Optimal Akhir |
| :---: | :--- | :---: | :---: | :---: |
| 1 | **Thompson Sampling (Bayesian)** | 260.25 ± 18.57 | 29.76 | 94.0% |
| 2 | **Decaying ε-Greedy** | 244.88 ± 23.84 | 44.39 | 82.0% |
| 3 | **ε-Greedy (ε=0.10)** | 242.71 ± 41.96 | 47.23 | 80.0% |
| 4 | **ε-Greedy (ε=0.05)** | 242.25 ± 52.27 | 49.26 | 82.0% |
| 5 | **Optimistic Greedy (Q0=1.0)** | 184.26 ± 89.98 | 102.73 | 41.0% |
| 6 | **UCB1 (c=1.0)** | 182.06 ± 16.44 | 106.75 | 41.0% |
| 7 | **Pure Random (Baseline)** | 143.44 ± 10.21 | 147.39 | 23.0% |

## Detail Alokasi Penarikan Tuas (Arm Distribution)

| Algoritma | Arm #0 | Arm #1 | Arm #2 | Arm #3 | Arm #4 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Thompson Sampling (Bayesian) | 63.0 | 64.5 | 143.0 | 1649.4 | 80.1 |
| Decaying ε-Greedy | 100.4 | 99.3 | 223.9 | 1474.5 | 101.9 |
| ε-Greedy (ε=0.10) | 61.1 | 60.1 | 347.9 | 1389.4 | 141.5 |
| ε-Greedy (ε=0.05) | 64.7 | 128.6 | 355.2 | 1371.6 | 79.9 |
| Optimistic Greedy (Q0=1.0) | 163.8 | 301.1 | 419.0 | 815.5 | 300.6 |
| UCB1 (c=1.0) | 256.1 | 264.9 | 391.2 | 795.0 | 292.8 |
| Pure Random (Baseline) | 400.5 | 399.5 | 398.7 | 398.5 | 402.9 |