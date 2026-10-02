# 02 — AI Inference C++ (TensorFlow Lite / LiteRT C++ API)

Library [inference_cpp/](../inference_cpp/) membungkus TensorFlow Lite C++ API Google menjadi satu kelas `crosschick::FreshnessClassifier`. Kode yang **sama** dipakai oleh CLI desktop dan aplikasi Android (via JNI), sehingga hasil uji di PC = hasil di HP.

## 1. Mengambil Source TensorFlow Lite

TensorFlow Lite C++ API tidak didistribusikan sebagai binary siap pakai, jadi dibangun dari source resmi dengan CMake:

```powershell
.\scripts\setup_tensorflow_source.ps1          # Windows  -> third_party/tensorflow (tag v2.17.1)
./scripts/setup_tensorflow_source.sh           # Linux/macOS
```

Script yang sama juga meng-clone `stb_image.h` (pembaca JPG/PNG untuk CLI desktop) ke `third_party/stb`. Lokasi TensorFlow lain dapat dipakai dengan `-DTFLITE_SOURCE_DIR=<path>`.

Include path IntelliSense VS Code sudah diatur di [.vscode/c_cpp_properties.json](../.vscode/c_cpp_properties.json).

## 2. Build Desktop (uji di PC)

Prasyarat: CMake ≥ 3.16, Git, compiler C++17 (Visual Studio 2022 / GCC ≥ 9 / Clang ≥ 10).

```powershell
cmake -S inference_cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target classify_image -j 8
```

Build pertama cukup lama karena mengompilasi TFLite, XNNPACK, Abseil, FlatBuffers, dll.

Menjalankan:

```powershell
.\build\Release\classify_image.exe model.tflite labels.txt foto_ayam.jpg --threads 4 --bench 50
```

Contoh format output (angka ilustrasi):

```
Gambar     : foto_ayam.jpg (3024x4032)
Prediksi   : segar (94.12%)
  - segar                 94.12%
  - kurang_segar           5.31%
  - busuk                  0.57%
Preprocess : 1.85 ms | Inference: 6.40 ms
Benchmark  : 50 run, rata-rata 8.10 ms
```

## 3. API

Header: [freshness_classifier.h](../inference_cpp/include/crosschick/freshness_classifier.h)

```cpp
#include "crosschick/freshness_classifier.h"

crosschick::ClassifierOptions opt;
opt.num_threads = 4;

std::string err;
auto clf = crosschick::FreshnessClassifier::CreateFromFile("model.tflite", opt, &err);
if (!clf) { /* tampilkan err */ }
clf->SetLabels(crosschick::LoadLabels("labels.txt"));

crosschick::ImageView img{pixels, width, height, /*stride*/ 0, crosschick::PixelFormat::kRGB};
crosschick::Classification res;
if (clf->Classify(img, &res, &err)) {
  printf("%s %.2f\n", res.label.c_str(), res.confidence);
}
```

| Fungsi | Keterangan |
|---|---|
| `CreateFromFile(path, opt, &err)` | memuat `.tflite` dari file |
| `CreateFromBuffer(std::vector<char>, opt, &err)` | memuat dari memori (dipakai Android: asset) |
| `SetLabels(labels)` | mengisi nama kelas sesuai `labels.txt` |
| `Classify(ImageView, &result, &err)` | preprocessing + inferensi; thread-safe |
| `input_width/height()`, `num_classes()` | info model |

`ImageView` mendukung `kRGB`, `kRGBA`, `kBGR` (OpenCV), `kBGRA`, dan `row_stride_bytes` untuk buffer ber-padding (Android Bitmap).

## 4. Alur Internal

1. `FlatBufferModel::VerifyAndBuildFromBuffer` — memverifikasi integritas file `.tflite` sebelum dipakai.
2. `InterpreterBuilder` + `BuiltinOpResolver` — delegate **XNNPACK** (kernel CPU teroptimasi NEON/AVX) aktif otomatis.
3. Preprocessing ([freshness_classifier.cpp](../inference_cpp/src/freshness_classifier.cpp)):
   center-crop persegi → resize bilinear (half-pixel center) ke 224×224 → tulis RGB 0..255. Proses ini identik dengan `image_dataset_from_directory(crop_to_aspect_ratio=True)` saat training. Untuk model int8, nilai dikuantisasi memakai `scale`/`zero_point` tensor input.
4. `Interpreter::Invoke()`.
5. Output didekuantisasi bila perlu → probabilitas, argmax, label.

Tidak ada koneksi jaringan: seluruh proses berjalan lokal.

## 5. Opsi CMake

| Opsi | Default | Keterangan |
|---|---|---|
| `TFLITE_SOURCE_DIR` | `third_party/tensorflow` | lokasi source TensorFlow |
| `CROSSCHICK_BUILD_CLI` | `ON` | build `classify_image` (desktop saja) |
| `TFLITE_ENABLE_XNNPACK` | `ON` (dipaksa) | akselerasi CPU |
| `TFLITE_ENABLE_GPU` | `OFF` (dipaksa) | GPU delegate dinonaktifkan agar build sederhana |

## 6. Troubleshooting

| Masalah | Solusi |
|---|---|
| `Source TensorFlow tidak ditemukan` | jalankan script setup di langkah 1 |
| Error path terlalu panjang di Windows | aktifkan *LongPathsEnabled* di registry, `git config --global core.longpaths true`, atau clone ke path pendek (mis. `D:\tf`) lalu set `-DTFLITE_SOURCE_DIR` |
| `Didn't find op for builtin opcode ... version N` | versi TensorFlow Python lebih baru dari source TFLite C++ → samakan versinya |
| Inferensi sangat lambat | pastikan build `Release` |
| Hasil C++ berbeda dari Python | pastikan input RGB (bukan BGR) dan model/labels berasal dari export yang sama |
