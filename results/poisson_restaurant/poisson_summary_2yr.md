# Ringkasan Hasil Simulasi Poisson Porsi Besar (Maksimal 7 Piring, 2 Tahun)

- **Durasi:** 2 Tahun (730 Hari $\times$ 2 makan/hari = 1,460 Sesi Makan)
- **Kapasitas Perut Maksimal:** 7 Piring Tambah per Sesi Makan
- **Restoran Terbaik:** RM Padang Minang Asli [TERBAIK] (E[Piring] = 5.4300 piring/makan)

| Algoritma | Total Piring Tambah (Mean ± Std) | Cumulative Regret (Piring) | % Kunjungan RM Padang | Kunjungan RM Padang (dari 1.460) |
| :--- | :---: | :---: | :---: | :---: |
| **Gamma-Poisson Thompson Sampling** | 7750.5 ± 188.9 | 181.42 | 95.0% | 1298.7 / 1460 (89.0%) |
| **Poisson-UCB (c=1.0)** | 7741.3 ± 152.2 | 184.21 | 97.0% | 1301.4 / 1460 (89.1%) |
| **Poisson-UCB (c=√2 ≈ 1.41)** | 7649.1 ± 107.1 | 280.48 | 93.0% | 1239.0 / 1460 (84.9%) |
| **Pure Random (Baseline)** | 4464.3 ± 77.5 | 3465.88 | 14.0% | 146.0 / 1460 (10.0%) |