package org.jetbrains.kotlinx.dl.example.app

import android.content.res.Resources
import android.graphics.Bitmap
import androidx.camera.core.ImageProxy
import org.jetbrains.kotlinx.dl.api.extension.argmax
import org.jetbrains.kotlinx.dl.api.inference.imagerecognition.InputType
import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModelHub
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.api.inference.onnx.classification.ImageRecognitionModel
import org.jetbrains.kotlinx.dl.api.inference.onnx.classification.predictTopKObjects
import org.jetbrains.kotlinx.dl.api.inference.onnx.executionproviders.ExecutionProvider.CPU
import org.jetbrains.kotlinx.dl.api.inference.onnx.executionproviders.ExecutionProvider.NNAPI
import org.jetbrains.kotlinx.dl.api.inference.onnx.inferUsing
import org.jetbrains.kotlinx.dl.api.inference.onnx.objectdetection.SSDLikeModel
import org.jetbrains.kotlinx.dl.api.inference.onnx.objectdetection.detectObjects
import org.jetbrains.kotlinx.dl.api.inference.onnx.posedetection.SinglePoseDetectionModel
import org.jetbrains.kotlinx.dl.api.inference.onnx.posedetection.detectPose
import org.jetbrains.kotlinx.dl.api.inference.posedetection.DetectedPose
import org.jetbrains.kotlinx.dl.dataset.Imagenet
import org.jetbrains.kotlinx.dl.dataset.preprocessing.*
import org.jetbrains.kotlinx.dl.dataset.preprocessing.camerax.toBitmap


interface InferencePipeline {
    fun analyze(image: ImageProxy): Pair<Any, Float>?
    fun close()
}

enum class Pipelines {
    SSDMobilenetV1 {
        override fun createPipeline(hub: ONNXModelHub, resources: Resources): InferencePipeline {
            return DetectionPipeline(ONNXModels.ObjectDetection.SSDMobileNetV1.pretrainedModel(hub))
        }
    },
    EfficientNetLite4 {
        override fun createPipeline(hub: ONNXModelHub, resources: Resources): InferencePipeline {
            return ClassificationPipeline(ONNXModels.CV.EfficientNet4Lite().pretrainedModel(hub))
        }
    },
    MobilenetV1 {
        override fun createPipeline(hub: ONNXModelHub, resources: Resources): InferencePipeline {
            return ClassificationPipeline(ONNXModels.CV.MobilenetV1().pretrainedModel(hub))
        }
    },
    Shufflenet {
        override fun createPipeline(hub: ONNXModelHub, resources: Resources): InferencePipeline {
            return ShufflenetPipeline(
                OnnxInferenceModel(resources.openRawResource(R.raw.shufflenet).readBytes())
            )
        }
    },
    EfficientDetLite0 {
        override fun createPipeline(hub: ONNXModelHub, resources: Resources): InferencePipeline {
            return DetectionPipeline(ONNXModels.ObjectDetection.EfficientDetLite0.pretrainedModel(hub))
        }
    },
    MoveNetSinglePoseLighting {
        override fun createPipeline(hub: ONNXModelHub, resources: Resources): InferencePipeline {
            return PoseDetectionPipeline(ONNXModels.PoseDetection.MoveNetSinglePoseLighting.pretrainedModel(hub))
        }
    };

    abstract fun createPipeline(hub: ONNXModelHub, resources: Resources): InferencePipeline

}

internal class DetectionPipeline(private val model: SSDLikeModel) : InferencePipeline {
    override fun analyze(image: ImageProxy): Pair<DetectedObject, Float>? {
        val detections = model.inferUsing(CPU()) {
            it.detectObjects(image, 1)
        }
        if (detections.isEmpty()) return null

        val detection = detections.single()
        return detection to detection.probability
    }

    override fun close() {
        model.close()
    }
}

internal class ClassificationPipeline(private val model: ImageRecognitionModel) : InferencePipeline {

    override fun analyze(image: ImageProxy): Pair<String, Float>? {
        val predictions = model.inferUsing(NNAPI()) {
            it.predictTopKObjects(image, 1)
        }
        if (predictions.isEmpty()) return null
        return predictions.single()
    }

    override fun close() {
        model.close()
    }
}

internal class ShufflenetPipeline(
    private val model: OnnxInferenceModel
) : InferencePipeline {
    private val labels = Imagenet.V1k.labels()

    override fun analyze(image: ImageProxy): Pair<String, Float> {
        val bitmap = image.toBitmap()
        val rotation = image.imageInfo.rotationDegrees.toFloat()

        val preprocessing = pipeline<Bitmap>()
            .resize {
                outputHeight = 224
                outputWidth = 224
            }
            .rotate { degrees = rotation }
            .toFloatArray { layout = TensorLayout.NCHW }
            .call(InputType.TORCH.preprocessing(channelsLast = false))

        val (label, confidence) = model.inferUsing(CPU()) {
            val (tensor, shape) = preprocessing.apply(bitmap)
            val logits = model.predictSoftly(tensor)
            val (confidence, _) = Softmax().apply(logits to shape)
            val labelId = confidence.argmax()
            labels[labelId]!! to confidence[labelId]
        }

        return label to confidence
    }

    override fun close() {
        model.close()
    }
}

class PoseDetectionPipeline(private val model: SinglePoseDetectionModel) : InferencePipeline {
    override fun analyze(image: ImageProxy): Pair<DetectedPose, Float>? {
        val detectedPose = model.inferUsing(CPU()) {
            it.detectPose(image)
        }

        if (detectedPose.landmarks.isEmpty()) return null

        return detectedPose to 1f
    }

    override fun close() = model.close()
}
