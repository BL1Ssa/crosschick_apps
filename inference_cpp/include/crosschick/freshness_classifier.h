#pragma once

// Inference offline kesegaran daging ayam: MobileNetV4-Conv-Small R224
// menggunakan TensorFlow Lite (LiteRT) C++ API dari Google.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace crosschick {

enum class PixelFormat { kRGB, kRGBA, kBGR, kBGRA };

// Tampilan (non-owning) ke buffer gambar 8-bit interleaved.
struct ImageView {
  const uint8_t* data = nullptr;
  int width = 0;
  int height = 0;
  int row_stride_bytes = 0;  // 0 = width * channel
  PixelFormat format = PixelFormat::kRGB;
};

struct ClassifierOptions {
  int num_threads = 4;
};

struct Classification {
  int class_index = -1;
  std::string label;
  float confidence = 0.f;
  std::vector<float> probabilities;
  double preprocess_ms = 0.0;
  double inference_ms = 0.0;
};

class FreshnessClassifier {
 public:
  // Mengembalikan nullptr bila gagal; detail error ditulis ke `error` (opsional).
  static std::unique_ptr<FreshnessClassifier> CreateFromFile(
      const std::string& model_path, const ClassifierOptions& options,
      std::string* error = nullptr);

  static std::unique_ptr<FreshnessClassifier> CreateFromBuffer(
      std::vector<char> model_data, const ClassifierOptions& options,
      std::string* error = nullptr);

  ~FreshnessClassifier();
  FreshnessClassifier(const FreshnessClassifier&) = delete;
  FreshnessClassifier& operator=(const FreshnessClassifier&) = delete;

  void SetLabels(std::vector<std::string> labels);

  // Center-crop 1:1 -> resize bilinear ke 224x224 -> invoke -> softmax output.
  // Thread-safe (diserialisasi dengan mutex internal).
  bool Classify(const ImageView& image, Classification* result,
                std::string* error = nullptr);

  int input_width() const;
  int input_height() const;
  int num_classes() const;

 private:
  FreshnessClassifier();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Membaca labels.txt (satu label per baris, urutan = indeks output model).
std::vector<std::string> LoadLabels(const std::string& path);

}  // namespace crosschick
