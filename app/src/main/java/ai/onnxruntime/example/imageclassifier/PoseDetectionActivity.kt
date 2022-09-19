package ai.onnxruntime.example.imageclassifier

import android.graphics.BitmapFactory
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import kotlinx.android.synthetic.main.activity_pose.*
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModelHub
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.api.inference.onnx.executionproviders.ExecutionProvider.CPU
import org.jetbrains.kotlinx.dl.api.inference.onnx.inferUsing

class PoseDetectionActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_pose)

        val bitmap = BitmapFactory.decodeStream(resources.openRawResource(R.raw.pose))

        val hub = ONNXModelHub(applicationContext)
        val model = ONNXModels.PoseDetection.MoveNetSinglePoseLighting.pretrainedModel(hub)

        image.setImageBitmap(bitmap)

        val pose = model.inferUsing(CPU()) {
            it.detectPose(bitmap)
        }

        detections_view.setPose(pose)
    }
}
