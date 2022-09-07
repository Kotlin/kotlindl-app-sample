package ai.onnxruntime.example.imageclassifier

import android.os.Build
import android.os.SystemClock
import androidx.annotation.RequiresApi
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import org.jetbrains.kotlinx.dl.api.inference.onnx.executionproviders.ExecutionProvider.CPU
import org.jetbrains.kotlinx.dl.api.inference.onnx.inferUsing
import org.jetbrains.kotlinx.dl.api.inference.onnx.objectdetection.SSDMobileNetObjectDetectionModel


internal class KotlinDlAnalyzer (
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
            detections.isNotEmpty() -> uiUpdateCallBack(Result(end - start, detections[0]))
        }

        image.close()
    }

    // We can switch analyzer in the app, need to make sure the native resources are freed
    protected fun finalize() {
        model.close()
    }
}
