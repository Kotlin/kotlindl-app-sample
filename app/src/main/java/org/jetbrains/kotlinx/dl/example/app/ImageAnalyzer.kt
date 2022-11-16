package org.jetbrains.kotlinx.dl.example.app

import android.content.Context
import android.content.res.Resources
import android.os.SystemClock
import androidx.camera.core.ImageProxy
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModelHub

internal class ImageAnalyzer(
    context: Context,
    private val resources: Resources,
    private val uiUpdateCallBack: (AnalysisResult?) -> Unit,
    initialPipelineIndex: Int = 0
) {
    private val hub = ONNXModelHub(context)

    val pipelinesList = Pipelines.values().sortedWith(Comparator { o1, o2 ->
        if (o1.task != o2.task) return@Comparator o1.task.ordinal - o2.task.ordinal
        o1.ordinal - o2.ordinal
    })
    private val pipelines = pipelinesList.map { it.createPipeline(hub, resources) }

    @Volatile
    var currentPipelineIndex: Int = initialPipelineIndex
        private set
    private val currentPipeline: InferencePipeline? get() = pipelines.getOrNull(currentPipelineIndex)

    fun analyze(image: ImageProxy, isImageFlipped: Boolean) {
        val start = SystemClock.uptimeMillis()
        val result = currentPipeline?.analyze(image)
        val end = SystemClock.uptimeMillis()

        val rotationDegrees = image.imageInfo.rotationDegrees
        image.close()

        if (result == null) {
            uiUpdateCallBack(null)
        } else {
            val (prediction, confidence) = result
            val (width, height) = if (rotationDegrees == 0 || rotationDegrees == 180) image.width to image.height
            else image.height to image.width
            uiUpdateCallBack(
                AnalysisResult(
                    prediction, confidence, end - start, width, height,
                    isImageFlipped
                )
            )
        }
    }

    fun setPipeline(index: Int) {
        currentPipelineIndex = index
    }

    fun clear() {
        currentPipelineIndex = -1
    }

    fun close() {
        clear()
        pipelines.forEach(InferencePipeline::close)
    }
}

data class AnalysisResult(
    val prediction: Any,
    val confidence: Float,
    val processTimeMs: Long,
    val width: Int,
    val height: Int,
    val isImageFlipped: Boolean
)
