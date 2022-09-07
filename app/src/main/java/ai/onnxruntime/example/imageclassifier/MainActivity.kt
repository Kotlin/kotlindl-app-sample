package ai.onnxruntime.example.imageclassifier

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.RectF
import android.os.Bundle
import android.util.Log
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import kotlinx.android.synthetic.main.activity_main.*
import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.api.inference.onnx.objectdetection.SSDMobileNetObjectDetectionModel
import java.lang.Integer.min
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors


class MainActivity : AppCompatActivity() {
    private val backgroundExecutor: ExecutorService by lazy { Executors.newSingleThreadExecutor() }

    private var imageCapture: ImageCapture? = null
    private var imageAnalysis: ImageAnalysis? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
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


            val model = SSDMobileNetObjectDetectionModel(readModelBytes())
            imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(backgroundExecutor, KotlinDlAnalyzer(model, ::updateUI))
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

        if (result.detection.probability < 0.5f)
            return

        runOnUiThread {
            box_prediction.visibility = View.GONE

            percentMeter.progress = (result.detection.probability * 100).toInt()
            detected_item_1.text = result.detection.classLabel
            detected_item_value_1.text = "%.2f%%".format(result.detection.probability * 100)

            val rect = mapOutputCoordinates(result)

            (box_prediction.layoutParams as ViewGroup.MarginLayoutParams).apply {
                topMargin = rect.top.toInt()
                leftMargin = rect.left.toInt()
                width = min(viewFinder.width, rect.right.toInt() - rect.left.toInt())
                height = min(viewFinder.height, rect.bottom.toInt() - rect.top.toInt())
            }

            box_prediction.visibility = View.VISIBLE

            inference_time_value.text = result.processTimeMs.toString() + "ms"
        }
    }

    private fun mapOutputCoordinates(result: Result): RectF {
        // Step 1: map location to the preview coordinates
        val previewLocation = RectF(
            result.detection.xMin * viewFinder.width,
            result.detection.yMin * viewFinder.height - 350,
            result.detection.xMax * viewFinder.width,
            result.detection.yMax * viewFinder.height - 350
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

//         Step 3: compensate for 1:1 to 4:3 aspect ratio conversion + small margin
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

    private fun readModelBytes(): ByteArray {
        return resources.openRawResource(R.raw.ssd_onnx_300_with_runtime_opt).readBytes();
    }

    companion object {
        const val TAG = "ORTImageClassifier"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}


internal data class Result(
    var processTimeMs: Long = 0,
    var detection: DetectedObject
)
