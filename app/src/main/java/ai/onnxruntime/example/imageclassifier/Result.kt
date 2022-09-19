package ai.onnxruntime.example.imageclassifier

import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.api.inference.posedetection.DetectedPose

interface Result {
    var processTimeMs: Long
    val confidence: Float
    val text: String
}

internal data class DetectionResult(
    override var processTimeMs: Long,
    val detection: DetectedObject
) : Result {
    override val text get() = detection.classLabel
    override val confidence get() = detection.probability
}

internal data class ClassificationResult(
    override var processTimeMs: Long,
    override val confidence: Float,
    val prediction: String
) : Result {
    override val text get() = prediction
}

internal data class PoseDetectionResult(
    override var processTimeMs: Long,
    val detection: DetectedPose
) : Result {
    override val text get() = "human pose"
    override val confidence get() = 1f
}