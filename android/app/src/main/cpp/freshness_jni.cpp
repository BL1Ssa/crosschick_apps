// JNI bridge: Kotlin (com.crosschick.freshness.ml.FreshnessNative) <-> C++ FreshnessClassifier.

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include <jni.h>

#include <memory>
#include <string>
#include <vector>

#include "crosschick/freshness_classifier.h"

#define LOG_TAG "CrossChickJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

using crosschick::FreshnessClassifier;

namespace {

void ThrowJava(JNIEnv* env, const char* exception_class, const std::string& msg) {
  jclass cls = env->FindClass(exception_class);
  if (cls) env->ThrowNew(cls, msg.c_str());
}

FreshnessClassifier* FromHandle(JNIEnv* env, jlong handle) {
  auto* c = reinterpret_cast<FreshnessClassifier*>(handle);
  if (!c) ThrowJava(env, "java/lang/IllegalStateException", "Classifier sudah ditutup");
  return c;
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_crosschick_freshness_ml_FreshnessNative_create(JNIEnv* env, jobject /*thiz*/,
                                                       jobject asset_manager,
                                                       jstring model_asset,
                                                       jint num_threads) {
  AAssetManager* mgr = AAssetManager_fromJava(env, asset_manager);
  const char* path = env->GetStringUTFChars(model_asset, nullptr);
  const std::string asset_name(path);
  env->ReleaseStringUTFChars(model_asset, path);

  AAsset* asset = mgr ? AAssetManager_open(mgr, asset_name.c_str(), AASSET_MODE_BUFFER) : nullptr;
  if (!asset) {
    ThrowJava(env, "java/io/FileNotFoundException", "Asset tidak ditemukan: " + asset_name);
    return 0;
  }
  const off64_t size = AAsset_getLength64(asset);
  std::vector<char> data(static_cast<size_t>(size));
  const int read = AAsset_read(asset, data.data(), data.size());
  AAsset_close(asset);
  if (read != size) {
    ThrowJava(env, "java/io/IOException", "Gagal membaca asset: " + asset_name);
    return 0;
  }

  crosschick::ClassifierOptions options;
  options.num_threads = num_threads;
  std::string error;
  auto classifier = FreshnessClassifier::CreateFromBuffer(std::move(data), options, &error);
  if (!classifier) {
    ThrowJava(env, "java/lang/RuntimeException", error);
    return 0;
  }
  LOGI("Model dimuat: input %dx%d, %d kelas, %d thread", classifier->input_width(),
       classifier->input_height(), classifier->num_classes(), num_threads);
  return reinterpret_cast<jlong>(classifier.release());
}

// Return: [preprocess_ms, inference_ms, p0, p1, ..., pN-1]
extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_crosschick_freshness_ml_FreshnessNative_classify(JNIEnv* env, jobject /*thiz*/,
                                                         jlong handle, jobject bitmap) {
  FreshnessClassifier* classifier = FromHandle(env, handle);
  if (!classifier) return nullptr;

  AndroidBitmapInfo info;
  if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS ||
      info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
    ThrowJava(env, "java/lang/IllegalArgumentException", "Bitmap harus ARGB_8888");
    return nullptr;
  }
  void* pixels = nullptr;
  if (AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS) {
    ThrowJava(env, "java/lang/IllegalStateException", "Gagal mengunci pixel bitmap");
    return nullptr;
  }

  // ARGB_8888 Android tersimpan di memori sebagai byte R,G,B,A.
  crosschick::ImageView view{static_cast<const uint8_t*>(pixels), static_cast<int>(info.width),
                             static_cast<int>(info.height), static_cast<int>(info.stride),
                             crosschick::PixelFormat::kRGBA};
  crosschick::Classification result;
  std::string error;
  const bool ok = classifier->Classify(view, &result, &error);
  AndroidBitmap_unlockPixels(env, bitmap);

  if (!ok) {
    ThrowJava(env, "java/lang/RuntimeException", error);
    return nullptr;
  }

  std::vector<float> out;
  out.reserve(result.probabilities.size() + 2);
  out.push_back(static_cast<float>(result.preprocess_ms));
  out.push_back(static_cast<float>(result.inference_ms));
  out.insert(out.end(), result.probabilities.begin(), result.probabilities.end());

  jfloatArray arr = env->NewFloatArray(static_cast<jsize>(out.size()));
  if (arr) env->SetFloatArrayRegion(arr, 0, static_cast<jsize>(out.size()), out.data());
  return arr;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_crosschick_freshness_ml_FreshnessNative_numClasses(JNIEnv* env, jobject /*thiz*/,
                                                           jlong handle) {
  FreshnessClassifier* classifier = FromHandle(env, handle);
  return classifier ? classifier->num_classes() : 0;
}

extern "C" JNIEXPORT void JNICALL
Java_com_crosschick_freshness_ml_FreshnessNative_destroy(JNIEnv* /*env*/, jobject /*thiz*/,
                                                        jlong handle) {
  delete reinterpret_cast<FreshnessClassifier*>(handle);
}
