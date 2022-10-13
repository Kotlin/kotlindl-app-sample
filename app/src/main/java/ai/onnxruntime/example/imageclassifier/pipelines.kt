package ai.onnxruntime.example.imageclassifier

import android.content.Context
import android.content.res.Resources
import android.graphics.Bitmap
import android.os.Build
import android.os.SystemClock
import android.util.Log
import androidx.annotation.RequiresApi
import androidx.camera.core.ImageAnalysis
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
import org.jetbrains.kotlinx.dl.dataset.shape.TensorShape
import kotlin.math.exp

data class Result(
    val prediction: Any,
    val confidence: Float,
    val processTimeMs: Long,
    val width: Int,
    val height: Int,
)

internal class PipelineAnalyzer(
    context: Context,
    private val resources: Resources,
    private val uiUpdateCallBack: (Result?) -> Unit
) : ImageAnalysis.Analyzer {
    private val hub = ONNXModelHub(context)
    private val pipelines = Pipelines.values().map { it.createPipeline(hub, resources) }

    @Volatile
    private var currentPipeline: Pipeline? = null

    fun setPipeline(index: Int) {
        currentPipeline = pipelines[index]
    }

    fun clear() {
        currentPipeline = null
    }

    override fun analyze(image: ImageProxy) {
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
            uiUpdateCallBack(Result(prediction, confidence, end - start, width, height))
        }
    }

    fun close() {
        clear()
        pipelines.forEach(Pipeline::close)
    }
}

interface Pipeline {
    fun analyze(image: ImageProxy): Pair<Any, Float>?
    fun close()
}

enum class Pipelines {
    SSDMobilenetV1 {
        override fun createPipeline(hub: ONNXModelHub, resources: Resources): Pipeline {
            return DetectionPipeline(ONNXModels.ObjectDetection.SSDMobileNetV1.pretrainedModel(hub))
        }
    },
    EfficientNetLite4 {
        override fun createPipeline(hub: ONNXModelHub, resources: Resources): Pipeline {
            return ClassificationPipeline(ONNXModels.CV.EfficientNet4Lite().pretrainedModel(hub))
        }
    },
    MobilenetV1 {
        override fun createPipeline(hub: ONNXModelHub, resources: Resources): Pipeline {
            return ClassificationPipeline(ONNXModels.CV.MobilenetV1().pretrainedModel(hub))
        }
    },
    Shufflenet {
        override fun createPipeline(hub: ONNXModelHub, resources: Resources): Pipeline {
            return ShufflenetPipeline(
                OnnxInferenceModel(resources.openRawResource(R.raw.shufflenet).readBytes())
            )
        }
    },
    EfficientDetLite0 {
        override fun createPipeline(hub: ONNXModelHub, resources: Resources): Pipeline {
            return DetectionPipeline(ONNXModels.ObjectDetection.EfficientDetLite0.pretrainedModel(hub))
        }
    },
    MoveNetSinglePoseLighting {
        override fun createPipeline(hub: ONNXModelHub, resources: Resources): Pipeline {
            return PoseDetectionPipeline(ONNXModels.PoseDetection.MoveNetSinglePoseLighting.pretrainedModel(hub))
        }
    };

    abstract fun createPipeline(hub: ONNXModelHub, resources: Resources): Pipeline

}

internal class DetectionPipeline(private val model: SSDLikeModel) : Pipeline {
    override fun analyze(image: ImageProxy): Pair<DetectedObject, Float>? {
        Log.i("DetectionPipeline", "image size ${image.width} x ${image.height}")

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

internal class ClassificationPipeline(private val model: ImageRecognitionModel) : Pipeline {

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
) : Pipeline {
    private val labels = Imagenet.V1k.labels()

    @RequiresApi(Build.VERSION_CODES.R)
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

    internal class Softmax : FloatArrayOperation() {
        override fun applyImpl(data: FloatArray, shape: TensorShape): FloatArray {
            val logits = data.copyOf()
            val max = logits[logits.argmax()]
            var sum = 0.0f

            for (i in logits.indices) {
                logits[i] = exp(logits[i] - max)
                sum += logits[i]
            }

            if (sum != 0.0f) {
                for (i in logits.indices) {
                    logits[i] /= sum
                }
            }

            return logits
        }
    }
}

class PoseDetectionPipeline(private val model: SinglePoseDetectionModel) : Pipeline {
    override fun analyze(image: ImageProxy): Pair<DetectedPose, Float>? {
        val detectedPose = model.inferUsing(CPU()) {
            it.detectPose(image)
        }

        if (detectedPose.landmarks.isEmpty()) return null

        return detectedPose to 1f
    }

    override fun close() = model.close()
}
