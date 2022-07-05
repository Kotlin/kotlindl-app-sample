package ai.onnxruntime.example.imageclassifier

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OrtSession.SessionOptions
import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.BitmapFactory
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
import java.io.File
import java.io.IOException
import java.lang.Integer.min
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors


class MainActivity : AppCompatActivity() {
    private val backgroundExecutor: ExecutorService by lazy { Executors.newSingleThreadExecutor() }
    private val labelData: List<String> by lazy { readLabels() }

    private var imageCapture: ImageCapture? = null
    private var imageAnalysis: ImageAnalysis? = null
    private var enableNNAPI: Boolean = false
    private var ortEnv: OrtEnvironment? = null

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

        enable_nnapi_toggle.setOnCheckedChangeListener { _, isChecked ->
            enableNNAPI = isChecked
            imageAnalysis?.clearAnalyzer()
            imageAnalysis?.setAnalyzer(backgroundExecutor, ORTAnalyzer(CreateOrtSession(), ::updateUI))
        }
    }

    @Throws(IOException::class)
    fun getFileFromAssets(context: Context, fileName: String): File = File(context.cacheDir, fileName)
        .also {
            if (!it.exists()) {
                it.outputStream().use { cache ->
                    context.assets.open(fileName).use { inputStream ->
                        inputStream.copyTo(cache)
                    }
                }
            }
        }

    private fun startCamera() {
        // Initialize ortEnv8
        ortEnv = OrtEnvironment.getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL)

//        val bitmap = BitmapFactory.decodeResource(resources, R.raw.imggg)
//
//        val ort = ORTAnalyzer(CreateOrtSession(), ::updateUI)
//
//        ort.govnanalyze(bitmap)

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

            imageAnalysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also {
                        it.setAnalyzer(backgroundExecutor, ORTAnalyzer(CreateOrtSession(), ::updateUI))
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
        ortEnv?.close()
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

    private fun updateUI(result: Result) {
//        if (result.detectedScore.isEmpty())
//            return

        runOnUiThread {
            box_prediction.visibility = View.GONE
//            percentMeter.progress = (result.detectedScore[0] * 100).toInt()
//            detected_item_1.text = labelData[result.detectedIndices[0]]
//            detected_item_value_1.text = "%.2f%%".format(result.detectedScore[0] * 100)
//
//            if (result.detectedIndices.size > 1) {
//                detected_item_2.text = labelData[result.detectedIndices[1]]
//                detected_item_value_2.text = "%.2f%%".format(result.detectedScore[1] * 100)
//            }
//
//            if (result.detectedIndices.size > 2) {
//                detected_item_3.text = labelData[result.detectedIndices[2]]
//                detected_item_value_3.text = "%.2f%%".format(result.detectedScore[2] * 100)
//            }
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
            result.xmin * viewFinder.width,
            result.ymin * viewFinder.height - 350,
            result.xmax * viewFinder.width,
            result.ymax * viewFinder.height - 350
        )
//        val correctedLocation = previewLocation
//        return previewLocation
//
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

    private fun readModel(): ByteArray {
        return resources.openRawResource(R.raw.mobilenet_v1_float).readBytes();
    }

    private fun readSsd(): ByteArray {
        return resources.openRawResource(R.raw.modified3).readBytes();
    }

    private fun readLabels(): List<String> {
        return resources.openRawResource(R.raw.labels).bufferedReader().readLines()
    }

    private fun CreateOrtSession(): OrtSession? {
        val so = SessionOptions()
        so.use {
            // Set to use 2 intraOp threads for CPU EP
            so.setIntraOpNumThreads(2)

            if (enableNNAPI)
                so.addNnapi()

            return ortEnv?.createSession(readSsd(), so)
        }
    }

    companion object {
        public const val TAG = "ORTImageClassifier"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE)
    }
}
