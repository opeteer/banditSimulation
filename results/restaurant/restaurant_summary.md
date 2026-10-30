# Ringkasan Hasil Simulasi: Kasus 10 Restoran Makan (1 Tahun, 2x Makan/Hari)

- **Subjek:** 1 Agen yang tinggal di kota selama 365 hari.
- **Total Sesi Makan:** 730 kali makan (2x sehari: Siang & Malam).
- **Restoran Terbaik:** RM Padang Minang Asli [TERBAIK] (Skor Rata-rata: 0.88)

| Algoritma | Total Skor Kepuasan (Mean ± Std) | Cumulative Regret | % Makan di RM Padang Terbaik | Total Kunjungan RM Padang |
| :--- | :---: | :---: | :---: | :---: |
| **Thompson Sampling (Bayesian)** | 610.71 ± 3.01 | 31.92 | 96.0% | 571.7 / 730 (78.3%) |
| **UCB (c=√0.5)** | 551.75 ± 2.38 | 90.41 | 64.0% | 298.3 / 730 (40.9%) |
| **UCB1 (c=1.0)** | 528.31 ± 2.20 | 114.09 | 35.0% | 215.0 / 730 (29.4%) |
| **Pure Random (Baseline)** | 459.98 ± 5.07 | 182.43 | 12.0% | 73.2 / 730 (10.0%) |