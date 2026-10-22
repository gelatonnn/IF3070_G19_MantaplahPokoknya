# 🧠 Tugas Besar 1 IF3070 - Dasar Inteligensi Artifisial

### *Optimasi Penempatan Kendaraan pada Kapal dengan Local Search*

## 📋 Deskripsi

Repositori ini berisi implementasi tiga algoritma **Local Search** untuk memilih
kendaraan serta menentukan posisi dan orientasinya pada geladak kapal. Tujuan
utamanya adalah memaksimalkan total *ShippingFee* dengan tetap memenuhi batas
ukuran geladak, kapasitas berat, dan larangan tumpang tindih kendaraan.

Implementasi juga mendukung dua spesifikasi bonus:

- Objective alternatif `eta_priority`, yang mempertimbangkan *ShippingFee* dan ETA.
- Optimasi armada dengan jumlah kapal lebih dari satu.

## 🧩 Algoritma yang Diimplementasikan

1. **Hill Climbing**
   - Steepest-Ascent Hill Climbing
   - Stochastic Hill Climbing
   - Hill Climbing with Sideways Move
   - Random-Restart Hill Climbing
2. **Simulated Annealing**
3. **Genetic Algorithm**

Setiap algoritma dapat dijalankan sedikitnya tiga kali dengan seed berbeda.
Program mencatat state awal dan akhir, nilai objective, jumlah iterasi, durasi
pencarian, serta grafik perkembangan objective terhadap iterasi.

## ⚙️ Struktur Folder

```text
├── .github/workflows/       
├── data/
│   ├── data.json            dataset utama 90 kendaraan dan satu kapal
│   └── fleet.json           konfigurasi bonus multi-kapal
├── output/
│   └── bonus/               hasil HC, SA, dan GA untuk bonus 1 dan 3
├── src/
│   ├── Algorithm/           implementasi HC, SA, dan GA
│   ├── Core/                loader, state, objective, constraint, dan tetangga
│   ├── Visualization/       ekspor visualisasi state dan riwayat objective
│   ├── experiment.py        runner bersama untuk CLI dan Streamlit
│   ├── ui_support.py        grafik interaktif dan ringkasan perbandingan
│   └── main.py              CLI utama eksperimen
├── streamlit_app.py          UI web local
├── .streamlit/config.toml    tema warna antarmuka
├── requirements.txt
└── README.md
```

## 🚀 Cara Menjalankan Program

1. Clone repositori dan masuk ke folder proyek:

   ```bash
   git clone https://github.com/gelatonnn/IF3070_G19_MantaplahPokoknya.git
   cd IF3070_G19_MantaplahPokoknya
   ```

2. Pastikan Python 3.12 telah terpasang. Buat dan aktifkan virtual environment:

   ```bash
   python -m venv .venv
   ```

   PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   Linux/macOS:

   ```bash
   source .venv/bin/activate
   ```

3. Instal dependensi:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Jalankan seluruh algoritma untuk tiga seed pada kasus satu kapal:

   ```bash
   python -m src.main --data data/data.json --seeds 1 2 3 --output output/eksperimen
   ```

5. Jalankan eksperimen gabungan bonus objective ETA dan multi-kapal:

   ```bash
   python -m src.main --data data/data.json --fleet data/fleet.json --objective eta_priority --seeds 101 202 303 --output output/bonus_baru
   ```

Folder output harus baru atau kosong karena program tidak menimpa hasil eksperimen
yang sudah ada.

## 🎛️ Menjalankan Algoritma Tertentu

```bash
python -m src.main --algorithm steepest --seeds 42 --output output/hc_steepest
python -m src.main --algorithm simulated_annealing --seeds 42 --output output/sa
python -m src.main --algorithm genetic_algorithm --seeds 42 --output output/ga
python -m src.main --help
```

Gunakan `--no-plots` jika hanya ingin menghasilkan JSON dan CSV. Objective yang
tersedia adalah `shipping_fee` dan `eta_priority`.

## 🖥️ Antarmuka Streamlit

Jalankan dari folder utama proyek setelah memasang dependensi:

```bash
python -m streamlit run streamlit_app.py
```

Pilih **Eksperimen tunggal** untuk satu algoritma dan seed, atau **Perbandingan**
untuk beberapa algoritma dan seed pada dataset, objective, dan konfigurasi kapal
yang sama. Parameter khusus algoritma muncul segera setelah algoritma dipilih.
Dataset kendaraan JSON dan konfigurasi armada JSON dapat diunggah; bila kosong,
aplikasi memakai `data/data.json` dan `data/fleet.json`. Maksimum 30 eksekusi
sekali jalan.

Hasil menampilkan metrik, tabel per kapal, denah geladak dan riwayat optimasi
interaktif (hover/zoom), serta grafik PNG asli. Unduh JSON/PNG individual,
`summary.csv`, atau seluruh hasil sebagai ZIP. Eksekusi yang tersimpan dapat
dibuka lagi dari sidebar.

## 📊 Output Program

Setiap eksekusi menghasilkan:

- `result.json` berisi parameter, state awal dan akhir, constraint, nilai
  objective, iterasi, dan durasi pencarian.
- `initial.png` dan `final.png` untuk visualisasi susunan kendaraan.
- `history.png` untuk perkembangan nilai objective terhadap iterasi.
- `summary.csv` untuk ringkasan seluruh algoritma dan seed.
- Detail setiap restart dalam `restart_XX/` untuk Random-Restart Hill Climbing.

Hasil eksperimen bonus yang sudah dibuat tersedia di `output/bonus/` dengan
struktur algoritma → varian (khusus HC) → seed.

## 👥 Pembagian Tugas

| Anggota | Bagian |
| --- | --- |
| Tiffany Viriya Chu - 18224068 | Simulated Annealing |
| Francis Galton - 18224072| Dataset dan seluruh varian Hill Climbing |
| Zahrah Nur Azizah - 18224092| Core & Genetic Algorithm |
