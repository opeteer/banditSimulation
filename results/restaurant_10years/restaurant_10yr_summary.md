# Ringkasan Hasil Simulasi 10 Restoran Selama 10 Tahun (7.300 Sesi Makan)

- **Horizon:** 10 Tahun (3,650 Hari $\times$ 2 makan/hari = 7,300 Sesi Makan)
- **Restoran Terbaik:** RM Padang Minang Asli [TERBAIK] (Skor Rata-rata: 0.88)

| Algoritma | Total Skor Kepuasan (Mean ± Std) | Cumulative Regret | % Kunjungan RM Padang | Total Kunjungan RM Padang |
| :--- | :---: | :---: | :---: | :---: |
| **Thompson Sampling (Bayesian)** | 6368.33 ± 6.17 | 55.11 | 100.0% | 7013.2 / 7300 (96.1%) |
| **UCB1 (c=1.0)** | 6000.92 ± 6.91 | 422.72 | 89.0% | 5075.9 / 7300 (69.5%) |
| **UCB1 (c=√2 ≈ 1.41)** | 5764.22 ± 8.06 | 659.06 | 78.0% | 3979.1 / 7300 (54.5%) |
| **Pure Random (Baseline)** | 4599.72 ± 14.50 | 1823.82 | 11.0% | 730.1 / 7300 (10.0%) |