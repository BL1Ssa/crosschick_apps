package com.crosschick.freshness.ml

import android.content.res.AssetManager
import android.graphics.Bitmap

/** Deklarasi fungsi native (implementasi di src/main/cpp/freshness_jni.cpp). */
internal object FreshnessNative {
    init {
        System.loadLibrary("freshness_jni")
    }

    external fun create(assetManager: AssetManager, modelAsset: String, numThreads: Int): Long

    /** Return: [preprocessMs, inferenceMs, p0, p1, ...]. Bitmap wajib ARGB_8888. */
    external fun classify(handle: Long, bitmap: Bitmap): FloatArray

    external fun numClasses(handle: Long): Int

    external fun destroy(handle: Long)
}
