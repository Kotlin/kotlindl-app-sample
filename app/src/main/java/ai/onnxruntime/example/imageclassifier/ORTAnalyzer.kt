package ai.onnxruntime.example.imageclassifier

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Build
import android.os.SystemClock
import androidx.annotation.RequiresApi
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import java.util.*

private const val OUTPUT_BOXES = "TFLite_Detection_PostProcess"
private const val OUTPUT_LABELS = "TFLite_Detection_PostProcess:1"
private const val OUTPUT_SCORES = "TFLite_Detection_PostProcess:2"

internal data class Result(
    var label: Int = -1,
    var score: Float = -1f,
    var processTimeMs: Long = 0,
    var xmin: Float = 0f,
    var ymin: Float = 0f,
    var xmax: Float = 0f,
    var ymax: Float = 0f
)

internal class ORTAnalyzer(
        private val ortSession: OrtSession?,
        private val uiUpdateCallBack: (Result) -> Unit
) : ImageAnalysis.Analyzer {
    // Rotate the image of the input bitmap
    fun Bitmap.rotate(degrees: Float): Bitmap {
        val matrix = Matrix().apply { postRotate(degrees) }
        return Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
    }

    @RequiresApi(Build.VERSION_CODES.R)
    override fun analyze(image: ImageProxy) {
        // Convert the input image to bitmap and resize to 224x224 for model input
        val imgBitmap = image.toBitmap()
        val rawBitmap = imgBitmap?.let { Bitmap.createScaledBitmap(it, IMAGE_SIZE_X, IMAGE_SIZE_Y, false) }
        val bitmap = rawBitmap?.rotate(image.imageInfo.rotationDegrees.toFloat())

        if (bitmap != null) {
            val imgData = preprocess(bitmap)
            val inputName = ortSession?.inputNames?.iterator()?.next()
            val result = Result()
            val shape = longArrayOf(1, IMAGE_SIZE_X.toLong(), IMAGE_SIZE_Y.toLong(), 3)
            val ortEnv = OrtEnvironment.getEnvironment()
            ortEnv.use {
                // Create input tensor
                val input_tensor = OnnxTensor.createTensor(ortEnv, imgData, shape)
                val startTime = SystemClock.uptimeMillis()
                input_tensor.use {
                    // Run the inference and get the output tensor
                    val output = ortSession?.run(Collections.singletonMap(inputName, input_tensor))
                    if (output!![OUTPUT_BOXES].isPresent) {
                        output.use {
                            // Populate the result
                            result.processTimeMs = SystemClock.uptimeMillis() - startTime
                            @Suppress("UNCHECKED_CAST")
                            val boxes = (output!![OUTPUT_BOXES].get().value as Array<Array<FloatArray>>)[0]
                            @Suppress("UNCHECKED_CAST")
                            val probabilities = (output[OUTPUT_SCORES].get().value as Array<FloatArray>)[0]
                            @Suppress("UNCHECKED_CAST")
                            val labels = (output[OUTPUT_LABELS].get().value as Array<FloatArray>)[0]

                            result.ymin = boxes[0][0]
                            result.xmin = boxes[0][1]
                            result.ymax = boxes[0][2]
                            result.xmax = boxes[0][3]

                            result.label = labels[0].toInt()
                            result.score = probabilities[0]
                            output.close()
                        }
                    } else {
                        output.close()
                    }
                }
            }

            // Update the UI
            uiUpdateCallBack(result)
        }

        image.close()
    }

    // We can switch analyzer in the app, need to make sure the native resources are freed
    protected fun finalize() {
        ortSession?.close()
    }
}