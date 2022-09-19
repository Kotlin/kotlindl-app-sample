package ai.onnxruntime.example.imageclassifier

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.text.TextPaint
import android.util.AttributeSet
import android.view.View
import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
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

    override fun Canvas.drawDetection(detection: Result) {
        when (detection) {
            is DetectionResult -> drawObject(detection.detection, objectPaint, textPaint)
        }
    }
}