# Multi-Armed Bandit (MAB) Simulation Engine

Proyek simulasi dan evaluasi komprehensif untuk algoritma **Multi-Armed Bandit (MAB)** pada skenario optimasi dunia nyata (khususnya optimasi materi iklan / Click-Through Rate (CTR) dan A/B/n testing dinamis). Proyek ini dilengkapi dengan simulasi Monte Carlo multi-run, metrik penyesalan teoritis (*cumulative regret*), dan visualisasi grafik komparatif berkualitas tinggi.

---

## 🎯 Gambaran Masalah: Exploration vs. Exploitation Dilemma

Dalam skenario optimasi (misalnya menguji 5 variasi banner iklan baru di e-commerce), pengiklan menghadapi dilema klasik:
- **Eksplorasi (Exploration):** Menguji variasi banner yang belum banyak dicoba untuk mengumpulkan informasi akurat mengenai CTR aktualnya.
- **Eksploitasi (Exploitation):** Menampilkan banner yang saat ini memiliki performa CTR terbaik untuk memaksimalkan jumlah klik dan pendapatan saat ini.

Pendekatan konvensional (*fixed A/B testing*) membagi traffic secara rata hingga periode tertentu selesai, menyebabkan kerugian kesempatan (*opportunity loss*) yang besar. **Multi-Armed Bandit** menyelesaikan masalah ini dengan mengalihkan traffic secara dinamis ke variasi terbaik seraya eksperimen masih berlangsung.

---

## 🧠 Algoritma yang Diimplementasikan

| Algoritma | Mekanisme Utama | Karakteristik Performa |
| :--- | :--- | :--- |
| **Pure Random** | Pemilihan acak seragam $a_t \sim \text{Uniform}(0, K-1)$. | Baseline kontrol; regret linier $\mathcal{O}(t)$. |
| **$\epsilon$-Greedy** | Eksplorasi acak dengan probabilitas $\epsilon$, eksploitasi $Q_t(a)$ dengan probabilitas $1-\epsilon$. | Mudah diimplementasikan; regret linier akibat eksplorasi statis. |
| **Decaying $\epsilon$-Greedy** | Nilai $\epsilon_t$ meluruh seiring waktu: $\epsilon_t = \max(\epsilon_{min}, \frac{\epsilon_0}{1 + \gamma t})$. | Mengurangi eksplorasi di tahap akhir; regret sub-linier. |
| **UCB1 (Upper Confidence Bound)** | $a_t = \arg\max_k \left[ Q_k + c \sqrt{\frac{2 \ln t}{N_k}} \right]$. | Berdasarkan prinsip optimisme menghadapi ketidakpastian (*optimism in the face of uncertainty*). |
| **Thompson Sampling (Bayesian)** | Memilih arm berdasarkan sampel dari distribusi posterior Beta: $\theta_k \sim \text{Beta}(\alpha_k, \beta_k)$. | State-of-the-art untuk Bernoulli reward; adaptif dan konvergen sangat cepat. |
| **Optimistic Initial Values** | Inisialisasi $Q_0 \gg \mu^*$ pada agen serakah (greedy). | Mendorong eksplorasi agresif pada tahap awal; sensitif terhadap skala reward. |

---

## 📁 Struktur Direktori

```text
banditSimulation/
├── bandit/
│   ├── __init__.py          # Export modul
│   ├── agents.py            # Implementasi 6 agen algoritma MAB
│   ├── environment.py       # BernoulliBandit, GaussianBandit, NonStationaryBandit
│   ├── simulator.py         # Monte Carlo multi-run simulator
│   └── visualizer.py        # Generator grafik Matplotlib & laporan teks
├── results/                 # Folder output grafik & laporan (PNG, JSON, MD)
│   ├── bandit_comparison_dashboard.png
│   ├── cumulative_rewards.png
│   ├── cumulative_regrets.png
│   ├── optimal_action_rate.png
│   ├── arm_allocation_distribution.png
│   ├── simulation_summary.json
│   └── simulation_summary.md
├── tests/
│   └── test_bandit.py       # Unit tests otomatis
├── main.py                  # CLI eksekusi simulasi
└── README.md                # Dokumentasi proyek
```

---

## 🚀 Cara Menjalankan

### 1. Menjalankan Studi Kasus Default (Optimasi Iklan Digital)
```bash
python3 main.py --case ad-optimization --steps 2000 --runs 100
```

### 2. Menjalankan Kasus Uji Klinis Adaptif (*Clinical Trial*)
```bash
python3 main.py --case clinical-trial --steps 1500 --runs 100
```

### 3. Menjalankan Kasus Non-Stationary (Kejenuhan Iklan / Ad Fatigue)
```bash
python3 main.py --case non-stationary --steps 2000 --runs 100
```

### 4. Menjalankan Kasus Kustom dengan Probabilitas Tertentu
```bash
python3 main.py --case custom --custom-probs 0.04 0.12 0.05 0.25 0.08 --steps 1500 --runs 100
```

### 5. Menjalankan Unit Tests
```bash
python3 -m unittest discover tests
```

---

## 📊 Visualisasi & Laporan yang Dihasilkan

Setelah simulasi selesai, berkas visualisasi tersimpan di direktori `results/`:

1. **`bandit_comparison_dashboard.png`**
   Dashboard terpadu 4-panel yang memuat akumulasi reward, akumulasi regret, persentase arm optimal, dan distribusi alokasi tarikan arm.
2. **`cumulative_rewards.png`**
   Membandingkan total reward kumulatif yang dikumpulkan tiap algoritma terhadap garis teoritis optimal (*Oracle*).
3. **`cumulative_regrets.png`**
   Memvisualisasikan *expected regret* $R_t = \sum (\mu^* - \mu_{a_t})$. Algoritma optimal (seperti Thompson Sampling) memiliki kurva regret yang melandai (sub-linier/logaritmik).
4. **`optimal_action_rate.png`**
   Tingkat keberhasilan agen dalam memilih arm terbaik pada setiap langkah waktu $t$.
5. **`arm_allocation_distribution.png`**
   Diagram batang alokasi jumlah tarikan untuk masing-masing arm. Algoritma cerdas memusatkan mayoritas tarikan pada arm dengan probabilitas tertinggi.
6. **`simulation_summary.md` & `simulation_summary.json`**
   Ringkasan tabel metrik numerik dan data kurva untuk integrasi dashboard.

---

## 📝 Kesimpulan Temuan Eksperimen
1. **Thompson Sampling** menunjukkan kinerja paling unggul pada kasus Bernoulli CTR, menghasilkan reward tertinggi dengan regret paling rendah, serta mengalokasikan >80% traffic ke banner terbaik.
2. **Decaying $\epsilon$-Greedy** mengungguli $\epsilon$-Greedy statis karena mampu menekan eksplorasi berlebih setelah variasi terbaik teridentifikasi.
3. **Pure Random** menghasilkan regret linier terbesar, mengonfirmasi urgensi penggunaan algoritma bandit adaptif dibandingkan uji A/B statis berdurasi panjang.
