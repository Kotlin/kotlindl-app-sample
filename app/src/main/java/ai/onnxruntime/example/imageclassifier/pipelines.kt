package ai.onnxruntime.example.imageclassifier

import android.os.Build
import android.os.SystemClock
import androidx.annotation.RequiresApi
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import org.jetbrains.kotlinx.dl.api.inference.onnx.classification.ImageRecognitionModel
import org.jetbrains.kotlinx.dl.api.inference.onnx.executionproviders.ExecutionProvider
import org.jetbrains.kotlinx.dl.api.inference.onnx.executionproviders.ExecutionProvider.CPU
import org.jetbrains.kotlinx.dl.api.inference.onnx.executionproviders.ExecutionProvider.NNAPI
import org.jetbrains.kotlinx.dl.api.inference.onnx.inferUsing
import org.jetbrains.kotlinx.dl.api.inference.onnx.objectdetection.SSDMobileNetObjectDetectionModel

internal class DetectionPipeline (
    private val model: SSDMobileNetObjectDetectionModel,
    private val uiUpdateCallBack: (Result) -> Unit
) : ImageAnalysis.Analyzer {

    @RequiresApi(Build.VERSION_CODES.R)
    override fun analyze(image: ImageProxy) {
        val imgBitmap = image.toBitmap()

        model.setTargetRotation(image.imageInfo.rotationDegrees.toFloat())

        val start = SystemClock.uptimeMillis()
        val detections = model.inferUsing(CPU()) {
            it.detectObjects(imgBitmap!!, 1)
        }
        val end = SystemClock.uptimeMillis()

        when {
            detections.isNotEmpty() -> {
                val detection = detections[0]
                uiUpdateCallBack(DetectionResult(end - start, detection.probability, detection))
            }
        }

        image.close()
    }

    // We can switch analyzer in the app, need to make sure the native resources are freed
    protected fun finalize() {
        model.close()
    }
}

internal class ClassificationPipeline (
    private val model: ImageRecognitionModel,
    private val uiUpdateCallBack: (Result) -> Unit
) : ImageAnalysis.Analyzer {

    @RequiresApi(Build.VERSION_CODES.R)
    override fun analyze(image: ImageProxy) {
        val imgBitmap = image.toBitmap()

        model.setTargetRotation(image.imageInfo.rotationDegrees.toFloat())

        val start = SystemClock.uptimeMillis()
        val predictions = model.inferUsing(NNAPI()) {
            it.predictTopKObjects(imgBitmap!!, 1)
        }
        val end = SystemClock.uptimeMillis()

        when {
            predictions.isNotEmpty() -> {
                val (label, confidence) = predictions[0]
                uiUpdateCallBack(ClassificationResult(end - start, confidence, label))
            }
        }

        image.close()
    }

    // We can switch analyzer in the app, need to make sure the native resources are freed
    protected fun finalize() {
        model.close()
    }
}
