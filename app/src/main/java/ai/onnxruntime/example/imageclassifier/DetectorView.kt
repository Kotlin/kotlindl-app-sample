package ai.onnxruntime.example.imageclassifier

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.text.TextPaint
import android.util.AttributeSet
import androidx.camera.view.PreviewView.ScaleType
import org.jetbrains.kotlinx.dl.visualization.*

class DetectorView(context: Context, attrs: AttributeSet) :
    DetectorViewBase<Result>(context, attrs) {
    private val objectPaint = Paint().apply {
        color = resources.getColor(R.color.white)
        style = Paint.Style.STROKE
        strokeWidth = resources.getDimensionPixelSize(R.dimen.object_stroke_width).toFloat()
    }
    private val textPaint = TextPaint().apply {
        textSize = resources.getDimensionPixelSize(R.dimen.label_font_size).toFloat()
        color = resources.getColor(R.color.white)
    }
    private val landmarkPaint = Paint().apply {
        color = resources.getColor(R.color.white)
        style = Paint.Style.FILL
        strokeWidth = resources.getDimensionPixelSize(R.dimen.object_stroke_width).toFloat()
    }
    private val radius = resources.getDimensionPixelSize(R.dimen.object_stroke_width).toFloat()
    private var bounds: PreviewImageBounds? = null

    var scaleType: ScaleType = ScaleType.FILL_CENTER

    override fun onDetectionSet(detection: Result?) {
        bounds = detection?.let {
            getPreviewImageBounds(it.width, it.height, width, height, scaleType)
        }
    }

    override fun Canvas.drawDetection(detection: Result) {
        val currentBounds = bounds ?: bounds()
        when (detection) {
            is DetectionResult -> drawObject(
                detection.detection,
                objectPaint, textPaint,
                currentBounds
            )

            is PoseDetectionResult -> drawPose(
                detection.detection,
                landmarkPaint, objectPaint, radius,
                currentBounds
            )
        }
    }
}