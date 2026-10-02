// CLI uji model di desktop (Windows/Linux/macOS), 100% offline.
//
//   classify_image <model.tflite> <labels.txt> <gambar.jpg> [--threads N] [--bench N]

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "crosschick/freshness_classifier.h"

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr,
                 "Usage: %s <model.tflite> <labels.txt> <image> [--threads N] [--bench N]\n",
                 argv[0]);
    return 1;
  }
  crosschick::ClassifierOptions options;
  int bench = 0;
  for (int i = 4; i + 1 < argc; i += 2) {
    if (std::strcmp(argv[i], "--threads") == 0) options.num_threads = std::atoi(argv[i + 1]);
    if (std::strcmp(argv[i], "--bench") == 0) bench = std::atoi(argv[i + 1]);
  }

  std::string error;
  auto classifier = crosschick::FreshnessClassifier::CreateFromFile(argv[1], options, &error);
  if (!classifier) {
    std::fprintf(stderr, "Error: %s\n", error.c_str());
    return 1;
  }
  classifier->SetLabels(crosschick::LoadLabels(argv[2]));

  int w = 0, h = 0, c = 0;
  stbi_uc* pixels = stbi_load(argv[3], &w, &h, &c, 3);
  if (!pixels) {
    std::fprintf(stderr, "Gagal membaca gambar: %s\n", argv[3]);
    return 1;
  }

  crosschick::ImageView view{pixels, w, h, 0, crosschick::PixelFormat::kRGB};
  crosschick::Classification result;
  if (!classifier->Classify(view, &result, &error)) {
    std::fprintf(stderr, "Error: %s\n", error.c_str());
    stbi_image_free(pixels);
    return 1;
  }

  const auto labels = crosschick::LoadLabels(argv[2]);
  std::vector<int> order(result.probabilities.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&](int a, int b) { return result.probabilities[a] > result.probabilities[b]; });

  std::printf("Gambar     : %s (%dx%d)\n", argv[3], w, h);
  std::printf("Prediksi   : %s (%.2f%%)\n", result.label.c_str(), result.confidence * 100.f);
  for (int i : order) {
    const std::string name = i < static_cast<int>(labels.size()) ? labels[i] : std::to_string(i);
    std::printf("  - %-20s %6.2f%%\n", name.c_str(), result.probabilities[i] * 100.f);
  }
  std::printf("Preprocess : %.2f ms | Inference: %.2f ms\n", result.preprocess_ms,
              result.inference_ms);

  if (bench > 0) {
    double total = 0.0;
    for (int i = 0; i < bench; ++i) {
      classifier->Classify(view, &result, nullptr);
      total += result.preprocess_ms + result.inference_ms;
    }
    std::printf("Benchmark  : %d run, rata-rata %.2f ms\n", bench, total / bench);
  }

  stbi_image_free(pixels);
  return 0;
}
