package com.crosschick.freshness.ml

import android.content.Context
import android.graphics.Bitmap

data class FreshnessResult(
    val index: Int,
    val label: String,
    val confidence: Float,
    /** Pasangan (label, probabilitas), diurutkan dari yang tertinggi. */
    val probabilities: List<Pair<String, Float>>,
    val preprocessMs: Float,
    val inferenceMs: Float,
)

/**
 * Wrapper Kotlin untuk inference C++ (TensorFlow Lite) MobileNetV4-Conv-Small R224.
 * Semua proses berjalan offline di perangkat. Panggil [close] saat tidak dipakai.
 */
class FreshnessClassifier private constructor(
    private var handle: Long,
    val labels: List<String>,
) : AutoCloseable {

    @Synchronized
    fun classify(bitmap: Bitmap): FreshnessResult {
        check(handle != 0L) { "Classifier sudah ditutup" }
        val argb = if (bitmap.config == Bitmap.Config.ARGB_8888) bitmap
        else bitmap.copy(Bitmap.Config.ARGB_8888, false)

        val raw = FreshnessNative.classify(handle, argb)
        val probs = raw.copyOfRange(2, raw.size)
        val best = probs.indices.maxBy { probs[it] }
        return FreshnessResult(
            index = best,
            label = labelAt(best),
            confidence = probs[best],
            probabilities = probs.indices.map { labelAt(it) to probs[it] }.sortedByDescending { it.second },
            preprocessMs = raw[0],
            inferenceMs = raw[1],
        )
    }

    private fun labelAt(i: Int) = labels.getOrElse(i) { "class_$i" }

    @Synchronized
    override fun close() {
        if (handle != 0L) {
            FreshnessNative.destroy(handle)
            handle = 0L
        }
    }

    companion object {
        const val MODEL_ASSET = "freshness_model.tflite"
        const val LABELS_ASSET = "labels.txt"

        /** Operasi berat (membaca & menyiapkan model) -> panggil dari background thread. */
        fun create(
            context: Context,
            modelAsset: String = MODEL_ASSET,
            labelsAsset: String = LABELS_ASSET,
            numThreads: Int = 4,
        ): FreshnessClassifier {
            val labels = context.assets.open(labelsAsset).bufferedReader().useLines { lines ->
                lines.map { it.trim() }.filter { it.isNotEmpty() }.toList()
            }
            val handle = FreshnessNative.create(context.assets, modelAsset, numThreads)
            val numClasses = FreshnessNative.numClasses(handle)
            if (numClasses != labels.size) {
                FreshnessNative.destroy(handle)
                throw IllegalStateException(
                    "Jumlah label ($labelsAsset: ${labels.size}) tidak sama dengan output model ($numClasses)"
                )
            }
            return FreshnessClassifier(handle, labels)
        }
    }
}
