#include "crosschick/freshness_classifier.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <mutex>
#include <utility>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"

namespace crosschick {
namespace {

using Clock = std::chrono::steady_clock;

double ElapsedMs(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void SetError(std::string* error, std::string msg) {
  if (error) *error = std::move(msg);
}

int Channels(PixelFormat f) {
  return (f == PixelFormat::kRGBA || f == PixelFormat::kBGRA) ? 4 : 3;
}

bool IsBgr(PixelFormat f) {
  return f == PixelFormat::kBGR || f == PixelFormat::kBGRA;
}

template <typename T>
T Quantize(float v, float scale, int zero_point) {
  const float q = std::round(v / scale) + static_cast<float>(zero_point);
  const float lo = static_cast<float>(std::numeric_limits<T>::min());
  const float hi = static_cast<float>(std::numeric_limits<T>::max());
  return static_cast<T>(std::clamp(q, lo, hi));
}

// Center-crop persegi + resize bilinear (half-pixel centers), identik dengan
// keras.utils.image_dataset_from_directory(crop_to_aspect_ratio=True).
// `write(dst_index, r, g, b)` menerima nilai float 0..255.
template <typename Writer>
void CropResize(const ImageView& img, int out_w, int out_h, Writer write) {
  const int ch = Channels(img.format);
  const int stride = img.row_stride_bytes > 0 ? img.row_stride_bytes : img.width * ch;
  const bool bgr = IsBgr(img.format);

  const int side = std::min(img.width, img.height);
  const int x0 = (img.width - side) / 2;
  const int y0 = (img.height - side) / 2;
  const float sx = static_cast<float>(side) / out_w;
  const float sy = static_cast<float>(side) / out_h;

  for (int oy = 0; oy < out_h; ++oy) {
    const float fy = std::clamp((oy + 0.5f) * sy - 0.5f, 0.f, side - 1.f);
    const int y_lo = static_cast<int>(fy);
    const int y_hi = std::min(y_lo + 1, side - 1);
    const float wy = fy - y_lo;
    const uint8_t* row_lo = img.data + static_cast<size_t>(y0 + y_lo) * stride;
    const uint8_t* row_hi = img.data + static_cast<size_t>(y0 + y_hi) * stride;

    for (int ox = 0; ox < out_w; ++ox) {
      const float fx = std::clamp((ox + 0.5f) * sx - 0.5f, 0.f, side - 1.f);
      const int x_lo = static_cast<int>(fx);
      const int x_hi = std::min(x_lo + 1, side - 1);
      const float wx = fx - x_lo;

      const uint8_t* p00 = row_lo + (x0 + x_lo) * ch;
      const uint8_t* p01 = row_lo + (x0 + x_hi) * ch;
      const uint8_t* p10 = row_hi + (x0 + x_lo) * ch;
      const uint8_t* p11 = row_hi + (x0 + x_hi) * ch;

      float rgb[3];
      for (int c = 0; c < 3; ++c) {
        const float top = p00[c] + (p01[c] - p00[c]) * wx;
        const float bot = p10[c] + (p11[c] - p10[c]) * wx;
        rgb[c] = top + (bot - top) * wy;
      }
      if (bgr) std::swap(rgb[0], rgb[2]);
      write(oy * out_w + ox, rgb[0], rgb[1], rgb[2]);
    }
  }
}

}  // namespace

struct FreshnessClassifier::Impl {
  std::vector<char> model_data;  // harus hidup selama model dipakai
  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<tflite::Interpreter> interpreter;
  std::vector<std::string> labels;
  std::mutex mutex;
  int in_w = 0, in_h = 0, num_classes = 0;
};

FreshnessClassifier::FreshnessClassifier() : impl_(std::make_unique<Impl>()) {}
FreshnessClassifier::~FreshnessClassifier() = default;

std::unique_ptr<FreshnessClassifier> FreshnessClassifier::CreateFromFile(
    const std::string& model_path, const ClassifierOptions& options, std::string* error) {
  std::ifstream file(model_path, std::ios::binary | std::ios::ate);
  if (!file) {
    SetError(error, "Tidak dapat membuka model: " + model_path);
    return nullptr;
  }
  const std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> data(static_cast<size_t>(size));
  if (!file.read(data.data(), size)) {
    SetError(error, "Gagal membaca model: " + model_path);
    return nullptr;
  }
  return CreateFromBuffer(std::move(data), options, error);
}

std::unique_ptr<FreshnessClassifier> FreshnessClassifier::CreateFromBuffer(
    std::vector<char> model_data, const ClassifierOptions& options, std::string* error) {
  std::unique_ptr<FreshnessClassifier> self(new FreshnessClassifier());
  Impl& d = *self->impl_;
  d.model_data = std::move(model_data);

  // Verifikasi flatbuffer sebelum dipakai (model bisa berasal dari file tak tepercaya).
  d.model = tflite::FlatBufferModel::VerifyAndBuildFromBuffer(d.model_data.data(),
                                                              d.model_data.size());
  if (!d.model) {
    SetError(error, "File .tflite tidak valid / rusak");
    return nullptr;
  }

  // BuiltinOpResolver otomatis menerapkan delegate XNNPACK (CPU teroptimasi).
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*d.model, resolver);
  builder.SetNumThreads(std::max(1, options.num_threads));
  if (builder(&d.interpreter) != kTfLiteOk || !d.interpreter) {
    SetError(error, "Gagal membuat tflite::Interpreter");
    return nullptr;
  }
  if (d.interpreter->inputs().size() != 1 || d.interpreter->outputs().size() != 1) {
    SetError(error, "Model harus memiliki tepat 1 input dan 1 output");
    return nullptr;
  }

  const int input_idx = d.interpreter->inputs()[0];
  const TfLiteTensor* in = d.interpreter->tensor(input_idx);
  if (in->dims->size == 4 && in->dims->data[0] != 1) {
    d.interpreter->ResizeInputTensor(input_idx, {1, in->dims->data[1], in->dims->data[2], 3});
  }
  if (d.interpreter->AllocateTensors() != kTfLiteOk) {
    SetError(error, "AllocateTensors gagal");
    return nullptr;
  }

  in = d.interpreter->input_tensor(0);
  if (in->dims->size != 4 || in->dims->data[3] != 3) {
    SetError(error, "Input model harus berbentuk [1, H, W, 3]");
    return nullptr;
  }
  if (in->type != kTfLiteFloat32 && in->type != kTfLiteUInt8 && in->type != kTfLiteInt8) {
    SetError(error, "Tipe input model tidak didukung (float32/uint8/int8)");
    return nullptr;
  }
  d.in_h = in->dims->data[1];
  d.in_w = in->dims->data[2];

  const TfLiteTensor* out = d.interpreter->output_tensor(0);
  d.num_classes = out->dims->data[out->dims->size - 1];
  if (out->type != kTfLiteFloat32 && out->type != kTfLiteUInt8 && out->type != kTfLiteInt8) {
    SetError(error, "Tipe output model tidak didukung");
    return nullptr;
  }
  return self;
}

void FreshnessClassifier::SetLabels(std::vector<std::string> labels) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->labels = std::move(labels);
}

int FreshnessClassifier::input_width() const { return impl_->in_w; }
int FreshnessClassifier::input_height() const { return impl_->in_h; }
int FreshnessClassifier::num_classes() const { return impl_->num_classes; }

bool FreshnessClassifier::Classify(const ImageView& image, Classification* result,
                                   std::string* error) {
  if (!result) {
    SetError(error, "result null");
    return false;
  }
  const int ch = Channels(image.format);
  if (!image.data || image.width <= 0 || image.height <= 0 ||
      (image.row_stride_bytes != 0 && image.row_stride_bytes < image.width * ch)) {
    SetError(error, "ImageView tidak valid");
    return false;
  }

  std::lock_guard<std::mutex> lock(impl_->mutex);
  Impl& d = *impl_;
  TfLiteTensor* in = d.interpreter->input_tensor(0);

  // Model sudah berisi Rescaling + normalisasi ImageNet -> cukup kirim RGB 0..255.
  auto t0 = Clock::now();
  switch (in->type) {
    case kTfLiteFloat32: {
      float* dst = in->data.f;
      CropResize(image, d.in_w, d.in_h, [dst](int i, float r, float g, float b) {
        dst[i * 3 + 0] = r;
        dst[i * 3 + 1] = g;
        dst[i * 3 + 2] = b;
      });
      break;
    }
    case kTfLiteUInt8: {
      uint8_t* dst = in->data.uint8;
      const float s = in->params.scale;
      const int zp = in->params.zero_point;
      CropResize(image, d.in_w, d.in_h, [=](int i, float r, float g, float b) {
        dst[i * 3 + 0] = Quantize<uint8_t>(r, s, zp);
        dst[i * 3 + 1] = Quantize<uint8_t>(g, s, zp);
        dst[i * 3 + 2] = Quantize<uint8_t>(b, s, zp);
      });
      break;
    }
    case kTfLiteInt8: {
      int8_t* dst = in->data.int8;
      const float s = in->params.scale;
      const int zp = in->params.zero_point;
      CropResize(image, d.in_w, d.in_h, [=](int i, float r, float g, float b) {
        dst[i * 3 + 0] = Quantize<int8_t>(r, s, zp);
        dst[i * 3 + 1] = Quantize<int8_t>(g, s, zp);
        dst[i * 3 + 2] = Quantize<int8_t>(b, s, zp);
      });
      break;
    }
    default:
      SetError(error, "Tipe input tidak didukung");
      return false;
  }
  result->preprocess_ms = ElapsedMs(t0);

  t0 = Clock::now();
  if (d.interpreter->Invoke() != kTfLiteOk) {
    SetError(error, "Interpreter::Invoke gagal");
    return false;
  }
  result->inference_ms = ElapsedMs(t0);

  const TfLiteTensor* out = d.interpreter->output_tensor(0);
  result->probabilities.resize(d.num_classes);
  for (int i = 0; i < d.num_classes; ++i) {
    float v;
    switch (out->type) {
      case kTfLiteUInt8:
        v = (out->data.uint8[i] - out->params.zero_point) * out->params.scale;
        break;
      case kTfLiteInt8:
        v = (out->data.int8[i] - out->params.zero_point) * out->params.scale;
        break;
      default:
        v = out->data.f[i];
    }
    result->probabilities[i] = v;
  }

  const auto best = std::max_element(result->probabilities.begin(), result->probabilities.end());
  result->class_index = static_cast<int>(best - result->probabilities.begin());
  result->confidence = *best;
  result->label = result->class_index < static_cast<int>(d.labels.size())
                      ? d.labels[result->class_index]
                      : "class_" + std::to_string(result->class_index);
  return true;
}

std::vector<std::string> LoadLabels(const std::string& path) {
  std::vector<std::string> labels;
  std::ifstream file(path);
  std::string line;
  while (std::getline(file, line)) {
    while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
    if (!line.empty()) labels.push_back(line);
  }
  return labels;
}

}  // namespace crosschick
