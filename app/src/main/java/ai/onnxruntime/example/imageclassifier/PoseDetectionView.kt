package ai.onnxruntime.example.imageclassifier

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color.red
import android.graphics.Paint
import android.text.TextPaint
import android.util.AttributeSet
import android.view.View
import org.jetbrains.kotlinx.dl.api.inference.facealignment.Landmark
import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.api.inference.posedetection.DetectedPose
import org.jetbrains.kotlinx.dl.visualization.drawObjects
import org.jetbrains.kotlinx.dl.visualization.drawPose

class PoseDetectionView(context: Context, attrs: AttributeSet) : View(context, attrs) {
    private val landmarkPaint = Paint().apply {
        color = resources.getColor(R.color.white)
        style = Paint.Style.STROKE
    }
    private val edgePaint = Paint().apply {
        color = resources.getColor(R.color.purple_500)
        strokeWidth = 4f
        style = Paint.Style.STROKE
    }
    private val detections = mutableListOf<DetectedPose>()

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        synchronized(this) {
            canvas.drawPose(detections[0], landmarkPaint, edgePaint, 4f)
        }
    }

    fun setPose(pose: DetectedPose) {
        synchronized(this) {
            detections.clear()
            detections.add(pose)
            this.postInvalidate()
        }
    }
}
