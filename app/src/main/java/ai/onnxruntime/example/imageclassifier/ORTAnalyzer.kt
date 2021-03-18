package ai.onnxruntime.example.imageclassifier

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import java.util.*
import kotlin.math.exp


internal data class Result(
        var detectedIndices: List<Int> = emptyList(),
        var detectedScore: MutableList<Float> = mutableListOf<Float>(),
        var processTimeMs: Long = 0
) {}

internal class ORTAnalyzer(
        private val ortSession: OrtSession?,
        private val callBack: (Result) -> Unit
) : ImageAnalysis.Analyzer {

    // Get index of top 3 values
    private fun argMax(labelVals: FloatArray): List<Int> {
        var indices = mutableListOf<Int>()
        for (k in 0..2) {
            var max: Float = 0.0f
            var idx: Int = 0
            for (i in 0..labelVals.size - 1) {
                val label_val = labelVals[i]
                if (label_val > max && !indices.contains(i)) {
                    max = label_val
                    idx = i
                }
            }

            indices.add(idx)
        }

        return indices.toList()
    }

    private fun softMax(modelResult: FloatArray): FloatArray {
        var labelVals = modelResult.copyOf()
        val max = labelVals.max()
        var sum = 0.0f

        // Get the reduced sum
        for (i in labelVals.indices) {
            labelVals[i] = exp(labelVals[i] - max!!)
            sum += labelVals[i]
        }

        if (sum != 0.0f) {
            for (i in labelVals.indices) {
                labelVals[i] /= sum
            }
        }

        return labelVals
    }

    // Rotate the image of the input bitmap
    fun Bitmap.rotate(degrees: Float): Bitmap {
        val matrix = Matrix().apply { postRotate(degrees) }
        return Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
    }

    override fun analyze(image: ImageProxy) {
        // Convert the input image to bitmap and resize to 224x224 for model input
        val imgBitmap = image.toBitmap()
        val rawBitmap = imgBitmap?.let { Bitmap.createScaledBitmap(it, 224, 224, false) }
        val bitmap = rawBitmap?.rotate(image.imageInfo.rotationDegrees.toFloat())

        if (bitmap != null) {
            val imgData = preprocess(bitmap)
            val inputName = ortSession?.inputNames?.iterator()?.next()
            var result = Result()
            val shape = longArrayOf(1, 224, 224, 3)
            val tensor = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), imgData, shape)
            val startTime = SystemClock.uptimeMillis()
            try {
                val output = ortSession?.run(Collections.singletonMap(inputName, tensor))
                result.processTimeMs = SystemClock.uptimeMillis() - startTime
                @Suppress("UNCHECKED_CAST")
                val labelVals = ((output?.get(0)?.value) as Array<FloatArray>)[0]
                result.detectedIndices = argMax(labelVals)
                for (idx in result.detectedIndices) {
                    result.detectedScore.add(labelVals[idx])
                }
                output.close()
            } finally {
                tensor.close()
            }

            callBack(result)
        }

        image.close()
    }

    protected fun finalize() {
        ortSession?.close()
    }
}