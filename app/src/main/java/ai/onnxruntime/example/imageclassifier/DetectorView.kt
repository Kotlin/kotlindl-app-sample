package ai.onnxruntime.example.imageclassifier

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.text.TextPaint
import android.util.AttributeSet
import android.util.Log
import android.view.View
import androidx.camera.core.AspectRatio
import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.api.inference.posedetection.DetectedPose
import org.jetbrains.kotlinx.dl.visualization.*
import kotlin.math.max
import kotlin.math.min
import kotlin.properties.Delegates

abstract class DetectorViewBase<T>(context: Context, attrs: AttributeSet) : View(context, attrs) {
    /**
     * Detection result to visualize
     */
    private var _detection: T? = null

    fun setDetection(detection: T?) {
        synchronized(this) {
            _detection = detection

            postInvalidate()
        }
    }

    /**
     * Draw given detection result on the [Canvas].
     */
    abstract fun Canvas.drawDetection(detection: T)

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        synchronized(this) {
            val detection = _detection
            if (detection != null) {
                canvas.drawDetection(detection)
            }
        }
    }
}

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

    override fun Canvas.drawDetection(detection: Result) {
        // only for scaleType="fillCenter"
        val scale = max(width.toFloat() / detection.width, height.toFloat() / detection.height)
        val previewWidth = detection.width * scale
        val previewHeight = detection.height * scale
        val previewX = width / 2 - previewWidth / 2
        val previewY = height / 2 - previewHeight / 2

        when (detection) {
            is DetectionResult -> drawObject(
                detection.detection,
                previewX, previewY, previewWidth, previewHeight,
                objectPaint, textPaint
            )
            is PoseDetectionResult -> drawPose(
                detection.detection,
                previewX, previewY, previewWidth, previewHeight,
                landmarkPaint, objectPaint, radius
            )
        }
    }
}

fun Canvas.drawObject(
    detectedObject: DetectedObject,
    x: Float, y: Float,
    width: Float, height: Float,
    paint: Paint,
    labelPaint: TextPaint
) {
    val x0 = detectedObject.xMin * width + x
    val y0 = detectedObject.yMin * height + y

    val frameWidth = paint.strokeWidth * detectedObject.probability

    val objectPaint = Paint(paint).apply { strokeWidth = frameWidth }
    drawRect(
        RectF(x0, y0, detectedObject.xMax * width + x, detectedObject.yMax * height + y),
        objectPaint
    )

    val label = "${detectedObject.classLabel} : " + "%.2f".format(detectedObject.probability)
    drawText(label, x0, y0 - labelPaint.fontMetrics.descent - frameWidth / 2, labelPaint)
}

fun Canvas.drawPose(
    detectedPose: DetectedPose,
    x: Float, y: Float,
    width: Float, height: Float,
    landmarkPaint: Paint, edgePaint: Paint,
    landmarkRadius: Float
) {
    detectedPose.edges.forEach { edge ->
        drawLine(
            width * edge.start.x + x, height * edge.start.y + y,
            width * edge.end.x + x, height * edge.end.y + y,
            edgePaint
        )
    }

    detectedPose.poseLandmarks.forEach { landmark ->
        drawCircle(width * landmark.x + x, height * landmark.y + y, landmarkRadius, landmarkPaint)
    }
}