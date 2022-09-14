package ai.onnxruntime.example.imageclassifier

import android.graphics.Bitmap
import android.os.Build
import android.os.SystemClock
import androidx.annotation.RequiresApi
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import org.jetbrains.kotlinx.dl.api.core.util.predictTop5Labels
import org.jetbrains.kotlinx.dl.api.extension.argmax
import org.jetbrains.kotlinx.dl.api.inference.imagerecognition.InputType
import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.api.inference.onnx.classification.ImageRecognitionModel
import org.jetbrains.kotlinx.dl.api.inference.onnx.executionproviders.ExecutionProvider.CPU
import org.jetbrains.kotlinx.dl.api.inference.onnx.executionproviders.ExecutionProvider.NNAPI
import org.jetbrains.kotlinx.dl.api.inference.onnx.inferUsing
import org.jetbrains.kotlinx.dl.api.inference.onnx.objectdetection.SSDMobileNetObjectDetectionModel
import org.jetbrains.kotlinx.dl.dataset.Imagenet
import org.jetbrains.kotlinx.dl.dataset.preprocessing.*
import org.jetbrains.kotlinx.dl.dataset.shape.TensorShape
import kotlin.math.exp
import kotlin.math.log

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

internal class ShufflenetPipeline (
    private val model: OnnxInferenceModel,
    private val uiUpdateCallBack: (Result) -> Unit
) : ImageAnalysis.Analyzer {
    private val labels = Imagenet.V1k.labels()

    @RequiresApi(Build.VERSION_CODES.R)
    override fun analyze(image: ImageProxy) {
        val imgBitmap = image.toBitmap()

        val preprocessing = pipeline<Bitmap>()
            .resize {
                outputHeight = 224
                outputWidth = 224
            }
            .rotate { degrees = image.imageInfo.rotationDegrees.toFloat() }
            .toFloatArray { layout = TensorLayout.NCHW }
            .call(InputType.TORCH.preprocessing(channelsLast = false))

        val start = SystemClock.uptimeMillis()
        val (label, confidence) = model.inferUsing(CPU()) {
            val (tensor, shape) = preprocessing.apply(imgBitmap!!)
            val logits = model.predictSoftly(tensor)
            val (confidence, _) = Softmax().apply(logits to shape)
            val labelId = confidence.argmax()
            labels[labelId]!! to confidence[labelId]
        }
        val end = SystemClock.uptimeMillis()

        uiUpdateCallBack(ClassificationResult(end - start, confidence, label))

        image.close()
    }

    // We can switch analyzer in the app, need to make sure the native resources are freed
    protected fun finalize() {
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

