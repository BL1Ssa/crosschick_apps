# 01 — Training MobileNetV4-Conv-Small R224 (TensorFlow)

## 1. Persiapan Environment

- Python 3.10 – 3.12
- `pip install -r training/requirements.txt` (TensorFlow **2.17.1**)
- GPU: TensorFlow ≥ 2.11 tidak mendukung GPU di Windows native. Gunakan **WSL2 + CUDA**, Linux, atau Google Colab. CPU tetap bisa dipakai untuk dataset kecil.

> Versi TensorFlow Python harus sama dengan tag source TensorFlow Lite yang dipakai C++ (`v2.17.1`, lihat [scripts/setup_tensorflow_source.ps1](../scripts/setup_tensorflow_source.ps1)). Jika berbeda, model bisa memakai versi operator yang belum dikenal runtime C++.

## 2. Dataset

Nama folder kelas = nama label. Contoh 3 tingkat kesegaran:

```
dataset/
├── train/
│   ├── segar/          *.jpg
│   ├── kurang_segar/   *.jpg
│   └── busuk/          *.jpg
├── val/   (struktur sama)
└── test/  (struktur sama, opsional)
```

Alternatif: `dataset/<kelas>/*.jpg` saja → split train/val otomatis (`--val-split 0.2`).

Rekomendasi pengambilan data:
- ≥ 300 foto per kelas, berbagai pencahayaan, latar, jarak, dan jenis potongan (dada, paha, sayap).
- Kelas sebaiknya ditentukan berdasarkan kriteria objektif (jam simpan, uji TVB-N/pH, atau SNI 3924 tentang mutu karkas dan daging ayam) agar label konsisten.
- Pisahkan train/val/test **per sampel daging** (bukan per foto) agar tidak terjadi kebocoran data.
- Simpan foto dalam orientasi tegak; `tf.io.decode_image` mengabaikan tag EXIF orientation.

## 3. Arsitektur

[training/mobilenetv4.py](../training/mobilenetv4.py) mengimplementasikan MobileNetV4-Conv-Small (Qin et al., 2024) sesuai spesifikasi resmi:

| Stage | Resolusi output | Blok |
|---|---|---|
| Stem | 112×112 | Conv3×3 s2, 32 |
| 0 | 56×56 | Conv3×3 s2 32 → Conv1×1 32 |
| 1 | 28×28 | Conv3×3 s2 96 → Conv1×1 64 |
| 2 | 14×14 | UIB ExtraDW(5,5) s2 96 → 4× UIB IB(k3) 96 → UIB ConvNext(k3) 96 |
| 3 | 7×7 | UIB ExtraDW(3,3) s2 128 → ExtraDW(5,5) → IB(k5) ×2 → IB(k3) ×2 |
| Head | 1×1 | Conv1×1 960 → GAP → Conv1×1 1280 → Dropout → Dense(N, softmax) |

- Backbone ±2.5 juta parameter, ±0.2 GMACs pada 224×224.
- Input model: RGB float32 0..255. `Rescaling(1/255)` + `Normalization(mean/std ImageNet)` ada **di dalam** model.
- Konvolusi stride-2 memakai padding simetris (gaya PyTorch) supaya bobot timm bisa diporting identik.

Cek arsitektur: `python mobilenetv4.py`

## 4. Bobot Pretrained (sangat disarankan)

Keras tidak menyediakan bobot MobileNetV4. Google merilis checkpoint ImageNet-1k resmi lewat timm (`mobilenetv4_conv_small.e2400_r224_in1k`, top-1 ±73.8%). Script berikut menyalin bobot ke Keras dan memverifikasi output fitur identik (`max abs diff < 1e-3`):

```bash
pip install -r requirements-convert.txt
python convert_timm_weights.py --output weights/mnv4_conv_small_imagenet.weights.h5
```

Dibutuhkan koneksi internet **sekali** untuk mengunduh bobot. Torch/timm tidak dibutuhkan untuk training berikutnya.

## 5. Training

```bash
python train.py --data-dir ../dataset \
    --pretrained-weights weights/mnv4_conv_small_imagenet.weights.h5
```

Strategi:
1. **Fase 1** (`--epochs-head`, default 10): backbone dibekukan, hanya classifier dilatih (`--lr-head 1e-3`).
2. **Fase 2** (`--epochs-finetune`, default 40): seluruh backbone dilatih dengan LR kecil (`--lr-finetune 1e-4`). Layer BatchNorm tetap beku agar stabil pada dataset kecil.

Optimizer AdamW + cosine decay dengan warmup, label smoothing 0.1, class weight otomatis untuk dataset tidak seimbang, early stopping pada `val_accuracy`.

Tanpa bobot pretrained (butuh ribuan gambar per kelas):

```bash
python train.py --data-dir ../dataset --epochs-head 0 --epochs-finetune 150 --lr-head 2e-3
```

Output di `outputs/`:

| File | Isi |
|---|---|
| `best.keras` | model terbaik (val_accuracy) |
| `final.keras` | model akhir (bobot terbaik fase terakhir) |
| `labels.txt` | urutan kelas → **wajib** ikut ke aplikasi |
| `training_meta.json` | konfigurasi & metrik |
| `training_curves.png`, `history_*.csv`, `tensorboard/` | kurva training |

Augmentasi ([dataset.py](../training/dataset.py)): flip, rotasi, zoom, translasi, serta brightness/contrast **ringan**. Hue/saturation sengaja tidak diubah karena warna daging adalah fitur utama kesegaran.

## 6. Evaluasi

```bash
python evaluate.py --model outputs/best.keras --data-dir ../dataset/test
```

Menghasilkan precision/recall/F1 per kelas, `confusion_matrix_best.png`, dan `eval_best.json`.

## 7. Export TensorFlow Lite

```bash
python export_tflite.py --model outputs/best.keras --quant fp16 --data-dir ../dataset/val
```

| `--quant` | Ukuran ± | Input/Output | Keterangan |
|---|---|---|---|
| `none` | 10 MB | float32 | referensi akurasi |
| `fp16` | 5 MB | float32 | **default**, akurasi ≈ float32 |
| `dynamic` | 3 MB | float32 | bobot int8 |
| `int8` | 3 MB | uint8 | tercepat di CPU; wajib `--data-dir` (kalibrasi) |

Script otomatis membandingkan output Keras vs TFLite. Setelah export, evaluasi ulang file `.tflite`:

```bash
python evaluate.py --model exported/freshness_mnv4s_r224_fp16.tflite --data-dir ../dataset/test
```

Library C++ otomatis mendeteksi tipe input/output (float32/uint8/int8), jadi semua varian bisa langsung dipakai.
