package ai.onnxruntime.example.imageclassifier

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.RectF
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.AdapterView.OnItemSelectedListener
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import kotlinx.android.synthetic.main.activity_main.*
import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModelHub
import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import java.lang.Integer.min
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors


class MainActivity : AppCompatActivity(), OnItemSelectedListener  {
    private val backgroundExecutor: ExecutorService by lazy { Executors.newSingleThreadExecutor() }

    private var imageCapture: ImageCapture? = null
    private var imageAnalysis: ImageAnalysis? = null

    private var modelNames = arrayOf(
        "SSDMobilenetV1",
        "EfficientNet-Lite4",
        "MobilenetV1",
        "Shufflenet",
        "EfficientDetLite0"
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val modelsSpinnerAdapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, modelNames)
        modelsSpinnerAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)

        with (models) {
            adapter = modelsSpinnerAdapter
            setSelection(0, false)
            onItemSelectedListener = this@MainActivity
            prompt = "Select model"
            gravity = Gravity.CENTER

        }

        pose_button.setOnClickListener {
            val intent = Intent(this, PoseDetectionActivity::class.java)
            startActivity(intent)
        }
        // Request Camera permission
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                    this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener(Runnable {
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Preview
            val preview = Preview.Builder()
                    .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                    .build()
                    .also {
                        it.setSurfaceProvider(viewFinder.surfaceProvider)
                    }

            imageCapture = ImageCapture.Builder()
                    .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                    .build()

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            val hub = ONNXModelHub(applicationContext)
            val model = ONNXModels.ObjectDetection.SSDMobileNetV1.pretrainedModel(hub)
            imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(backgroundExecutor, DetectionPipeline(model, ::updateUI))
                }

            try {
                cameraProvider.unbindAll()

                cameraProvider.bindToLifecycle(
                        this, cameraSelector, preview, imageCapture, imageAnalysis)
            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onDestroy() {
        super.onDestroy()
        backgroundExecutor.shutdown()
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(this,
                        "Permissions not granted by the user.",
                        Toast.LENGTH_SHORT).show()
                finish()
            }

        }
    }

    private fun updateUI(result: Result?) {
        if (result == null)
            return

        if (result.confidence < 0.5f)
            return

        runOnUiThread {
            clearUi()
            when (result) {
                is DetectionResult -> visualizeDetection(result.detection)
                is ClassificationResult -> visualizeClassification(result.prediction to result.confidence)
            }
            inference_time_value.text = result.processTimeMs.toString() + "ms"
        }
    }

    private fun clearUi() {
        box_prediction.visibility = View.GONE
    }

    private fun visualizeDetection(detection: DetectedObject) {
        percentMeter.progress = (detection.probability * 100).toInt()
        detected_item_1.text = detection.classLabel
        detected_item_value_1.text = "%.2f%%".format(detection.probability * 100)

        val rect = mapOutputCoordinates(detection)

        (box_prediction.layoutParams as ViewGroup.MarginLayoutParams).apply {
            topMargin = rect.top.toInt()
            leftMargin = rect.left.toInt()
            width = min(viewFinder.width, rect.right.toInt() - rect.left.toInt())
            height = min(viewFinder.height, rect.bottom.toInt() - rect.top.toInt())
        }

        box_prediction.visibility = View.VISIBLE
    }

    private fun visualizeClassification(prediction: Pair<String, Float>) {
        val (label, confidence) = prediction

        percentMeter.progress = (confidence * 100).toInt()
        detected_item_1.text = label
        detected_item_value_1.text = "%.2f%%".format(confidence * 100)
    }

    private fun mapOutputCoordinates(detection: DetectedObject): RectF {
        // Step 1: map location to the preview coordinates
        val previewLocation = RectF(
            detection.xMin * viewFinder.width,
            detection.yMin * viewFinder.height - 350,
            detection.xMax * viewFinder.width,
            detection.yMax * viewFinder.height - 350
        )


        // Step 2: compensate for camera sensor orientation and mirroring
        val isFrontFacing = false
        val correctedLocation = if (isFrontFacing) {
            RectF(
                viewFinder.width - previewLocation.right,
                previewLocation.top,
                viewFinder.width - previewLocation.left,
                previewLocation.bottom)
        } else {
            previewLocation
        }

        // Step 3: compensate for 1:1 to 4:3 aspect ratio conversion + small margin
        val margin = 0.00f
        val requestedRatio = 4f / 3f
        val midX = (correctedLocation.left + correctedLocation.right) / 2f
        val midY = (correctedLocation.top + correctedLocation.bottom) / 2f
        return if (viewFinder.width < viewFinder.height) {
            RectF(
                midX - (1f + margin) * requestedRatio * correctedLocation.width() / 2f,
                midY - (1f - margin) * correctedLocation.height() / 2f,
                midX + (1f + margin) * requestedRatio * correctedLocation.width() / 2f,
                midY + (1f - margin) * correctedLocation.height() / 2f
            )
        } else {
            RectF(
                midX - (1f - margin) * correctedLocation.width() / 2f,
                midY - (1f + margin) * requestedRatio * correctedLocation.height() / 2f,
                midX + (1f - margin) * correctedLocation.width() / 2f,
                midY + (1f + margin) * requestedRatio * correctedLocation.height() / 2f
            )
        }
    }

    companion object {
        const val TAG = "ORTImageClassifier"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }

    override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
        imageAnalysis?.clearAnalyzer()

        val hub = ONNXModelHub(applicationContext)

        val pipeline  = when (modelNames[position]) {
            "SSDMobilenetV1" -> {
                val model = ONNXModels.ObjectDetection.SSDMobileNetV1.pretrainedModel(hub)
                DetectionPipeline(model, ::updateUI)
            }
            "EfficientNet-Lite4" -> {
                val model = ONNXModels.CV.EfficientNet4Lite().pretrainedModel(hub)
                ClassificationPipeline(model, ::updateUI)
            }
            "MobilenetV1" -> {
                val model = ONNXModels.CV.MobilenetV1().pretrainedModel(hub)
                ClassificationPipeline(model, ::updateUI)
            }
            "Shufflenet" -> {
                val modelBytes = resources.openRawResource(R.raw.shufflenet).readBytes()
                val model = OnnxInferenceModel(modelBytes)
                ShufflenetPipeline(model, ::updateUI)
            }
            "EfficientDetLite0" -> {
                val model = ONNXModels.ObjectDetection.EfficientDetLite0.pretrainedModel(hub)
                DetectionPipeline(model, ::updateUI)
            }
            else -> throw NotImplementedError()
        }

        imageAnalysis?.setAnalyzer(backgroundExecutor, pipeline)
    }

    override fun onNothingSelected(parent: AdapterView<*>?) {
        TODO("Not yet implemented")
    }
}

internal interface Result {
    var processTimeMs: Long
    val confidence: Float
}

internal data class DetectionResult(
    override var processTimeMs: Long,
    override val confidence: Float,
    val detection: DetectedObject
) : Result


internal data class ClassificationResult(
    override var processTimeMs: Long,
    override val confidence: Float,
    val prediction: String
) : Result
