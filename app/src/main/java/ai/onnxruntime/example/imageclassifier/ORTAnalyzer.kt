package ai.onnxruntime.example.imageclassifier

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import java.nio.ByteBuffer
import java.util.*


internal data class Result(
    var detectedIndices: List<Int> = emptyList(),
    var detectedScore: MutableList<Float> = mutableListOf<Float>(),
    var processTimeMs: Long = 0
) {}

internal class ORTAnalyzer(
        private val ortSession: OrtSession?,
        private val callBack: (Result) -> Unit
) : ImageAnalysis.Analyzer {

    private fun argmMax(label_vals: FloatArray): List<Int>{
        var indices = mutableListOf<Int>()
        for( k in 0..2) {
            var max: Float = 0.0f
            var idx: Int = 0
            for (i in 0..label_vals.size - 1) {
                val label_val = label_vals[i]
                if (label_val > max && !indices.contains(i)) {
                    max = label_val
                    idx = i
                }
            }

            indices.add(idx)
        }

        return indices.toList()
    }

    fun Bitmap.rotate(degrees: Float): Bitmap {
        val matrix = Matrix().apply { postRotate(degrees) }
        return Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
    }

    override fun analyze(image: ImageProxy) {
        val imgBitmap = image.toBitmap()
        val rawBitmap = imgBitmap?.let { Bitmap.createScaledBitmap(it, 224, 224, false) }
        val bitmap = rawBitmap?.rotate(image.imageInfo.rotationDegrees.toFloat())

        if (bitmap != null) {
            val imgData = Preprocess(bitmap)
            val inputName = ortSession?.inputNames?.iterator()?.next()
            var result = Result()
            val shape = longArrayOf(1, 224, 224, 3)
            val tensor = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), imgData, shape)
            val startTime = SystemClock.uptimeMillis()
            try {
                val output = ortSession?.run(Collections.singletonMap(inputName, tensor))
                result.processTimeMs = SystemClock.uptimeMillis() - startTime
                @Suppress("UNCHECKED_CAST")
                val label_vals = ((output?.get(0)?.value) as Array<FloatArray>)[0]
                result.detectedIndices = argmMax(label_vals)
                for( idx in result.detectedIndices) {
                    result.detectedScore.add(label_vals[idx])
                }
                output.close()
            } finally {
                tensor.close()
            }

            callBack(result)
        }

        image.close()
    }
}