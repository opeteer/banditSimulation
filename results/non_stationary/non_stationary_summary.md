# Ringkasan Hasil Evaluasi Multi-Armed Bandit Non-Stationer

- **Horizon:** 1500 Sesi Makan (3 Fase $\times$ 500 Sesi)
- **Pergeseran:** Fase 1 (RM Padang 0.88) $\rightarrow$ Fase 2 (Bistro Western 0.90) $\rightarrow$ Fase 3 (Seafood Pesisir 0.92)

| Peringkat | Algoritma | Total Skor Kepuasan (Mean ± Std) | Cumulative Regret | % Arm Optimal Akhir |
| :---: | :--- | :---: | :---: | :---: |
| 1 | **Sliding-Window TS (W=150) [Adaptif]** | 1200.32 ± 6.19 | 149.74 | 88.0% |
| 2 | **Standard UCB1 (Statis)** | 1153.25 ± 3.15 | 196.35 | 74.0% |
| 3 | **Standard TS (Statis / Keras Kepala)** | 1150.55 ± 13.76 | 199.38 | 100.0% |
| 4 | **Discounted TS (γ=0.97) [Adaptif]** | 1072.31 ± 6.85 | 277.27 | 67.0% |
| 5 | **Sliding-Window UCB (W=150) [Adaptif]** | 916.95 ± 2.94 | 432.75 | 26.0% |
| 6 | **Discounted UCB (γ=0.97) [Adaptif]** | 845.23 ± 2.98 | 504.55 | 15.0% |
| 7 | **Pure Random (Baseline)** | 792.20 ± 7.43 | 557.50 | 8.0% |