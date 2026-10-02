package com.crosschick.freshness

import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.view.View
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.lifecycle.lifecycleScope
import com.crosschick.freshness.databinding.ActivityMainBinding
import com.crosschick.freshness.ml.FreshnessClassifier
import com.crosschick.freshness.ml.FreshnessResult
import com.crosschick.freshness.util.ImageUtils
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.Locale

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var classifier: FreshnessClassifier? = null
    private var pendingCaptureUri: Uri? = null

    private val pickImage = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        uri?.let(::analyze)
    }

    private val takePicture = registerForActivityResult(ActivityResultContracts.TakePicture()) { saved ->
        val uri = pendingCaptureUri
        if (saved && uri != null) analyze(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.btnGallery.setOnClickListener {
            pickImage.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }
        binding.btnCamera.setOnClickListener { launchCamera() }

        loadModel()
    }

    private fun loadModel() {
        lifecycleScope.launch {
            try {
                val c = withContext(Dispatchers.IO) { FreshnessClassifier.create(applicationContext) }
                classifier = c
                binding.txtStatus.text = getString(R.string.model_ready, c.labels.size)
                binding.btnGallery.isEnabled = true
                binding.btnCamera.isEnabled = true
            } catch (e: Exception) {
                binding.txtStatus.text = getString(R.string.model_failed, e.message)
            }
        }
    }

    private fun launchCamera() {
        val dir = File(cacheDir, "captures").apply { mkdirs() }
        val file = File(dir, "capture.jpg")
        val uri = FileProvider.getUriForFile(this, "$packageName.fileprovider", file)
        pendingCaptureUri = uri
        takePicture.launch(uri)
    }

    private fun analyze(uri: Uri) {
        val c = classifier ?: return
        binding.txtStatus.text = getString(R.string.analyzing)
        setButtonsEnabled(false)

        lifecycleScope.launch {
            try {
                val (bitmap, result) = withContext(Dispatchers.Default) {
                    val bmp: Bitmap = ImageUtils.loadBitmap(applicationContext, uri)
                    bmp to c.classify(bmp)
                }
                binding.imagePreview.setImageBitmap(bitmap)
                showResult(result)
            } catch (e: Exception) {
                binding.txtStatus.text = getString(R.string.classify_failed, e.message)
            } finally {
                setButtonsEnabled(true)
            }
        }
    }

    private fun showResult(r: FreshnessResult) {
        binding.txtStatus.text = getString(R.string.hint_pick_image)
        binding.txtLabel.text = prettyLabel(r.label)
        binding.txtLabel.setTextColor(ContextCompat.getColor(this, colorFor(r.label)))
        binding.txtConfidence.text = getString(R.string.result_confidence, r.confidence * 100f)
        binding.txtDetails.text = r.probabilities.joinToString("\n") { (label, p) ->
            String.format(Locale.US, "%-16s %6.2f%%", prettyLabel(label), p * 100f)
        }
        binding.txtTiming.text = getString(R.string.result_timing, r.preprocessMs, r.inferenceMs)
        binding.txtWarning.visibility = if (r.confidence < LOW_CONFIDENCE) View.VISIBLE else View.GONE
    }

    private fun setButtonsEnabled(enabled: Boolean) {
        binding.btnGallery.isEnabled = enabled
        binding.btnCamera.isEnabled = enabled
    }

    private fun prettyLabel(label: String) =
        label.replace('_', ' ').replaceFirstChar { it.titlecase(Locale.getDefault()) }

    private fun colorFor(label: String): Int {
        val l = label.lowercase(Locale.ROOT)
        return when {
            listOf("busuk", "spoil", "rotten", "tidak").any { it in l } -> R.color.spoiled
            listOf("kurang", "half", "medium", "sedang").any { it in l } -> R.color.medium
            listOf("segar", "fresh").any { it in l } -> R.color.fresh
            else -> R.color.neutral
        }
    }

    override fun onDestroy() {
        classifier?.close()
        classifier = null
        super.onDestroy()
    }

    private companion object {
        const val LOW_CONFIDENCE = 0.6f
    }
}
