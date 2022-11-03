package org.jetbrains.kotlinx.dl.example.app

import android.content.res.Resources
import android.graphics.Bitmap
import android.graphics.Rect
import androidx.camera.core.ImageProxy
import org.jetbrains.kotlinx.dl.api.inference.facealignment.Landmark
import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.api.inference.posedetection.DetectedPose
import org.jetbrains.kotlinx.dl.api.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.impl.dataset.Imagenet
import org.jetbrains.kotlinx.dl.impl.inference.imagerecognition.InputType
import org.jetbrains.kotlinx.dl.impl.preprocessing.*
import org.jetbrains.kotlinx.dl.impl.preprocessing.camerax.toBitmap
import org.jetbrains.kotlinx.dl.impl.util.argmax
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModelHub
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModels
import org.jetbrains.kotlinx.dl.onnx.inference.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.onnx.inference.classification.ImageRecognitionModel
import org.jetbrains.kotlinx.dl.onnx.inference.classification.predictTopKObjects
import org.jetbrains.kotlinx.dl.onnx.inference.executionproviders.ExecutionProvider.CPU
import org.jetbrains.kotlinx.dl.onnx.inference.executionproviders.ExecutionProvider.NNAPI
import org.jetbrains.kotlinx.dl.onnx.inference.facealignment.FaceDetectionModel
import org.jetbrains.kotlinx.dl.onnx.inference.facealignment.Fan2D106FaceAlignmentModel
import org.jetbrains.kotlinx.dl.onnx.inference.inferUsing
import org.jetbrains.kotlinx.dl.onnx.inference.objectdetection.SSDLikeModel
import org.jetbrains.kotlinx.dl.onnx.inference.objectdetection.detectObjects
import org.jetbrains.kotlinx.dl.onnx.inference.posedetection.SinglePoseDetectionModel
import org.jetbrains.kotlinx.dl.onnx.inference.posedetection.detectPose


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
    },
    FaceAlignment {
        override fun createPipeline(hub: ONNXModelHub, resources: Resources): InferencePipeline {
            val detectionModel = ONNXModels.FaceDetection.UltraFace320.pretrainedModel(hub)
            val alignmentModel = ONNXModels.FaceAlignment.Fan2d106.pretrainedModel(hub)
            return FaceAlignmentPipeline(detectionModel, alignmentModel)
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

class FaceAlignmentPipeline(
    private val detectionModel: FaceDetectionModel,
    private val alignmentModel: Fan2D106FaceAlignmentModel
) : InferencePipeline {
    override fun analyze(image: ImageProxy): Pair<FaceAlignmentResult, Float>? {
        val bitmap = image.toBitmap(applyRotation = true)

        val detectedObjects = detectionModel.detectFaces(bitmap, 1)
        if (detectedObjects.isEmpty()) {
            return null
        }

        val face = detectedObjects.first()
        val faceRect = Rect(
            (face.xMin * 0.9f * bitmap.width).toInt().coerceAtLeast(0),
            (face.yMin * 0.9f * bitmap.height).toInt().coerceAtLeast(0),
            (face.xMax * 1.1f * bitmap.width).toInt().coerceAtMost(bitmap.width),
            (face.yMax * 1.1f * bitmap.height).toInt().coerceAtMost(bitmap.height)
        )
        val faceCrop = pipeline<Bitmap>().crop {
            x = faceRect.left
            y = faceRect.top
            width = faceRect.width()
            height = faceRect.height()
        }.apply(bitmap)

        val landmarks = alignmentModel.predict(faceCrop)
            .map { landmark ->
                Landmark(
                    (faceRect.left + landmark.x * faceRect.width()) / bitmap.width,
                    (faceRect.top + landmark.y * faceRect.height()) / bitmap.height
                )
            }
        return FaceAlignmentResult(face, landmarks) to 1f
    }

    override fun close() {
        detectionModel.close()
        alignmentModel.close()
    }
}

data class FaceAlignmentResult(val face: DetectedObject, val landmarks: List<Landmark>)
