package ai.onnxruntime.example.imageclassifier

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Build
import android.os.Environment
import android.os.SystemClock
import androidx.annotation.RequiresApi
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.*
import kotlin.math.exp

//private const val OUTPUT_BOXES = "TFLite_Detection_PostProcess"
private const val OUTPUT_BOXES = "detection_boxes:0"
private const val OUTPUT_LABELS = "detection_classes:0"
private const val OUTPUT_SCORES = "detection_scores:0"

internal data class Result(
        var detectedIndices: List<Int> = emptyList(),
        var detectedScore: MutableList<Float> = mutableListOf<Float>(),
        var processTimeMs: Long = 0,
        var xmin: Float = 0f,
        var ymin: Float = 0f,
        var xmax: Float = 0f,
        var ymax: Float = 0f
) {}

internal class ORTAnalyzer(
        private val ortSession: OrtSession?,
        private val uiUpdateCallBack: (Result) -> Unit
) : ImageAnalysis.Analyzer {
    private val sz = 1000;

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

    @RequiresApi(Build.VERSION_CODES.R)
    override fun analyze(image: ImageProxy) {
        // Convert the input image to bitmap and resize to 224x224 for model input
        val imgBitmap = image.toBitmap()
        val rawBitmap = imgBitmap?.let { Bitmap.createScaledBitmap(it, sz, sz, false) }
        val bitmap = rawBitmap?.rotate(image.imageInfo.rotationDegrees.toFloat())


//        try {
//            FileOutputStream(File(Environment.getStorageDirectory().absolutePath + "govn.png")).use({ out ->
//                bitmap?.compress(
//                    Bitmap.CompressFormat.PNG,
//                    100,
//                    out
//                ) // bmp is your Bitmap instance
//            })
//        } catch (e: IOException) {
//            e.printStackTrace()
//        }

        if (bitmap != null) {
//            val pixelCount = image.cropRect.width() * image.cropRect.height()
//            val pixelSizeBits = ImageFormat.getBitsPerPixel(ImageFormat.YUV_420_888)
//            val outputBuffer = ByteArray(pixelCount * pixelSizeBits / 8)
//            imageToByteBuffer(image, outputBuffer, pixelCount)
            val imgData = preProcess(bitmap)
            val inputName = ortSession?.inputNames?.iterator()?.next()
            var result = Result()
            val shape = longArrayOf(1, sz.toLong(), sz.toLong(), 3)
            val ortEnv = OrtEnvironment.getEnvironment()
            ortEnv.use {
                // Create input tensor
                val input_tensor = OnnxTensor.createTensor(ortEnv,imgData, shape)
                val startTime = SystemClock.uptimeMillis()
                input_tensor.use {
                    // Run the inference and get the output tensor
                    val output = ortSession?.run(Collections.singletonMap(inputName, input_tensor))
                    output.use {
                        // Populate the result
                        result.processTimeMs = SystemClock.uptimeMillis() - startTime
                        val boxes = (output!![OUTPUT_BOXES].get().value as Array<Array<FloatArray>>)[0]
                        val probabilities = (output[OUTPUT_SCORES].get().value as Array<FloatArray>)[0]
                        val labels = (output[OUTPUT_LABELS].get().value as Array<FloatArray>)[0]
                        val m = probabilities.max()
                        val numberOfFoundObjects = boxes.size
                        for (i in 0 until numberOfFoundObjects) {
                                // left, bot, right, top
                            val yMin = boxes[i][0]
                            val xMin = boxes[i][1]
                            val yMax = boxes[i][2]
                            val xMax = boxes[i][3]

                            result.xmin = xMin
                            result.ymin = yMin
                            result.xmax = xMax
                            result.ymax = yMax

                            println("xmin: ${xMin}, ymin: ${yMin}, xmax: ${xMax}, ymax: ${yMax} score ${probabilities[i]} label ${labels[i]}")
                            break
                        }




//                        @Suppress("UNCHECKED_CAST")
//                        val labelVals = ((output?.get(0)?.value) as Array<FloatArray>)[0]
//                        result.detectedIndices = argMax(labelVals)
//                        for (idx in result.detectedIndices) {
//                            result.detectedScore.add(labelVals[idx])
//                        }
                        output?.close()
                    }
                }
            }

            // Update the UI
            uiUpdateCallBack(result)
        }

        image.close()
    }

    fun govnanalyze(bitmap: Bitmap) {
        val szz =1000
        val rawBitmap = bitmap?.let { Bitmap.createScaledBitmap(it, szz, szz, false) }
            val imgData = preProcess(rawBitmap)
            val inputName = ortSession?.inputNames?.iterator()?.next()
            var result = Result()
            val shape = longArrayOf(1, szz.toLong(), szz.toLong(), 3)
            val ortEnv = OrtEnvironment.getEnvironment()
            ortEnv.use {
                // Create input tensor
                val input_tensor = OnnxTensor.createTensor(ortEnv,imgData, shape)
                val startTime = SystemClock.uptimeMillis()
                input_tensor.use {
                    // Run the inference and get the output tensor
                    val output = ortSession?.run(Collections.singletonMap(inputName, input_tensor))
                    output.use {
                        // Populate the result
                        result.processTimeMs = SystemClock.uptimeMillis() - startTime
                        val boxes = (output!![OUTPUT_BOXES].get().value as Array<Array<FloatArray>>)[0]
                        val probabilities = (output[OUTPUT_SCORES].get().value as Array<FloatArray>)[0]
                        val labels = (output[OUTPUT_LABELS].get().value as Array<FloatArray>)[0]
                        val m = probabilities.max()
                        val numberOfFoundObjects = boxes.size
                        for (i in 0 until numberOfFoundObjects) {
                            // left, bot, right, top
                            val yMin = boxes[i][0]
                            val xMin = boxes[i][1]
                            val yMax = boxes[i][2]
                            val xMax = boxes[i][3]

                            result.xmin = xMin
                            result.ymin = yMin
                            result.xmax = xMax
                            result.ymax = yMax

                            println("xmin: ${xMin}, ymin: ${yMin}, xmax: ${xMax}, ymax: ${yMax} score ${probabilities[i]} label ${labels[i]}")
                            break
                        }




//                        @Suppress("UNCHECKED_CAST")
//                        val labelVals = ((output?.get(0)?.value) as Array<FloatArray>)[0]
//                        result.detectedIndices = argMax(labelVals)
//                        for (idx in result.detectedIndices) {
//                            result.detectedScore.add(labelVals[idx])
//                        }
                        output?.close()
                    }
                }
            }

    }

    // We can switch analyzer in the app, need to make sure the native resources are freed
    protected fun finalize() {
        ortSession?.close()
    }
}