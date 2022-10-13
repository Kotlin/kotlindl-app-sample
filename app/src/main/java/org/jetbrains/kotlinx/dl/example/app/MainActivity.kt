package org.jetbrains.kotlinx.dl.example.app

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.view.View
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
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors


class MainActivity : AppCompatActivity(), OnItemSelectedListener {
    private val backgroundExecutor: ExecutorService by lazy { Executors.newSingleThreadExecutor() }

    private var imageCapture: ImageCapture? = null
    private var imageAnalysis: ImageAnalysis? = null

    @Volatile
    private var pipelineAnalyzer: PipelineAnalyzer? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val modelsSpinnerAdapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_item,
            Pipelines.values().map { it.name })
        modelsSpinnerAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)

        with(models) {
            adapter = modelsSpinnerAdapter
            onItemSelectedListener = object : OnItemSelectedListener {
                override fun onItemSelected(
                    parent: AdapterView<*>?,
                    view: View?,
                    position: Int,
                    id: Long
                ) {
                    pipelineAnalyzer?.setPipeline(position)
                }

                override fun onNothingSelected(parent: AdapterView<*>?) {
                    pipelineAnalyzer?.clear()
                }
            }
            prompt = "Select model"
            gravity = Gravity.CENTER

        }
        // Request Camera permission
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }

        detector_view.scaleType = viewFinder.scaleType
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

            pipelineAnalyzer = PipelineAnalyzer(applicationContext, resources, ::updateUI)
            imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(backgroundExecutor, pipelineAnalyzer!!)
                }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture, imageAnalysis
                )
                runOnUiThread {
                    models.setSelection(0, false)
                }
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
        pipelineAnalyzer?.close()
        backgroundExecutor.shutdown()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(
                    this,
                    "Permissions not granted by the user.",
                    Toast.LENGTH_SHORT
                ).show()
                finish()
            }

        }
    }

    private fun updateUI(result: Result?) {
        runOnUiThread {
            clearUi()
            if (result == null || result.confidence < 0.5f) {
                detector_view.setDetection(null)
                return@runOnUiThread
            }
            detector_view.setDetection(result)
            percentMeter.progress = (result.confidence * 100).toInt()
            val (item, value) = when (val prediction = result.prediction) {
                is String -> prediction to "%.2f%%".format(result.confidence * 100)
                is DetectedObject -> prediction.label to "%.2f%%".format(result.confidence * 100)
                else -> "" to ""
            }
            detected_item_1.text = item
            detected_item_value_1.text = value
            inference_time_value.text = result.processTimeMs.toString() + "ms"
        }
    }

    private fun clearUi() {
        detected_item_1.text = ""
        detected_item_value_1.text = ""
        inference_time_value.text = ""
        percentMeter.progress = 0
    }

    companion object {
        const val TAG = "ORTImageClassifier"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }

    override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
        pipelineAnalyzer?.setPipeline(position)
    }

    override fun onNothingSelected(parent: AdapterView<*>?) {
        TODO("Not yet implemented")
    }
}