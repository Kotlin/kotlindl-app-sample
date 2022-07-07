package ai.onnxruntime.example.imageclassifier

import android.graphics.*
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.FloatBuffer

const val IMAGE_MEAN: Float = 127.5f;
const val IMAGE_STD: Float = 127.5f;
const val DIM_BATCH_SIZE = 1;
const val DIM_PIXEL_SIZE = 3;
const val IMAGE_SIZE_X = 300;
const val IMAGE_SIZE_Y = 300;

fun preprocess(bitmap: Bitmap): FloatBuffer {
    val imgData = FloatBuffer.allocate(
        DIM_BATCH_SIZE
                * DIM_PIXEL_SIZE
                * IMAGE_SIZE_X
                * IMAGE_SIZE_Y
    )
    imgData.rewind()
    val stride = IMAGE_SIZE_X * IMAGE_SIZE_Y
    val bmpData = IntArray(stride)
    bitmap.getPixels(bmpData, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
    var idx: Int = 0
    for (i in 0..IMAGE_SIZE_X - 1) {
        for (j in 0..IMAGE_SIZE_Y - 1) {
            val pixelValue = bmpData[idx++]
            imgData.put((pixelValue shr 16 and 0xFF).toFloat())
            imgData.put((pixelValue shr 8 and 0xFF).toFloat())
            imgData.put((pixelValue and 0xFF).toFloat())
        }
    }
//    for (i in 0..IMAGE_SIZE_SSD - 1) {
//        for (j in 0..IMAGE_SIZE_SSD - 1) {
//            val idx = IMAGE_SIZE_SSD * i + j
//            val pixelValue = bmpData[idx]
//            imgData.put(idx, (pixelValue shr 16 and 0xFF).toFloat())
//            imgData.put(idx + stride, ((pixelValue shr 8 and 0xFF)).toFloat())
//            imgData.put(idx + stride * 2, (pixelValue and 0xFF).toFloat())
//        }
//    }

    imgData.rewind()
    return imgData
}

fun ImageProxy.toBitmap(): Bitmap? {
    val nv21 = yuv420888ToNv21(this)
    val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
    return yuvImage.toBitmap()
}

private fun YuvImage.toBitmap(): Bitmap? {
    val out = ByteArrayOutputStream()
    if (!compressToJpeg(Rect(0, 0, width, height), 100, out))
        return null
    val imageBytes: ByteArray = out.toByteArray()
    return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
}

private fun yuv420888ToNv21(image: ImageProxy): ByteArray {
    val pixelCount = image.cropRect.width() * image.cropRect.height()
    val pixelSizeBits = ImageFormat.getBitsPerPixel(ImageFormat.YUV_420_888)
    val outputBuffer = ByteArray(pixelCount * pixelSizeBits / 8)
    imageToByteBuffer(image, outputBuffer, pixelCount)
    return outputBuffer
}

public fun imageToByteBuffer(image: ImageProxy, outputBuffer: ByteArray, pixelCount: Int) {
    assert(image.format == ImageFormat.YUV_420_888)

    val imageCrop = image.cropRect
    val imagePlanes = image.planes

    imagePlanes.forEachIndexed { planeIndex, plane ->
        // How many values are read in input for each output value written
        // Only the Y plane has a value for every pixel, U and V have half the resolution i.e.
        //
        // Y Plane            U Plane    V Plane
        // ===============    =======    =======
        // Y Y Y Y Y Y Y Y    U U U U    V V V V
        // Y Y Y Y Y Y Y Y    U U U U    V V V V
        // Y Y Y Y Y Y Y Y    U U U U    V V V V
        // Y Y Y Y Y Y Y Y    U U U U    V V V V
        // Y Y Y Y Y Y Y Y
        // Y Y Y Y Y Y Y Y
        // Y Y Y Y Y Y Y Y
        val outputStride: Int

        // The index in the output buffer the next value will be written at
        // For Y it's zero, for U and V we start at the end of Y and interleave them i.e.
        //
        // First chunk        Second chunk
        // ===============    ===============
        // Y Y Y Y Y Y Y Y    V U V U V U V U
        // Y Y Y Y Y Y Y Y    V U V U V U V U
        // Y Y Y Y Y Y Y Y    V U V U V U V U
        // Y Y Y Y Y Y Y Y    V U V U V U V U
        // Y Y Y Y Y Y Y Y
        // Y Y Y Y Y Y Y Y
        // Y Y Y Y Y Y Y Y
        var outputOffset: Int

        when (planeIndex) {
            0 -> {
                outputStride = 1
                outputOffset = 0
            }
            1 -> {
                outputStride = 2
                // For NV21 format, U is in odd-numbered indices
                outputOffset = pixelCount + 1
            }
            2 -> {
                outputStride = 2
                // For NV21 format, V is in even-numbered indices
                outputOffset = pixelCount
            }
            else -> {
                // Image contains more than 3 planes, something strange is going on
                return@forEachIndexed
            }
        }

        val planeBuffer = plane.buffer
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride

        // We have to divide the width and height by two if it's not the Y plane
        val planeCrop = if (planeIndex == 0) {
            imageCrop
        } else {
            Rect(
                    imageCrop.left / 2,
                    imageCrop.top / 2,
                    imageCrop.right / 2,
                    imageCrop.bottom / 2
            )
        }

        val planeWidth = planeCrop.width()
        val planeHeight = planeCrop.height()

        // Intermediate buffer used to store the bytes of each row
        val rowBuffer = ByteArray(plane.rowStride)

        // Size of each row in bytes
        val rowLength = if (pixelStride == 1 && outputStride == 1) {
            planeWidth
        } else {
            // Take into account that the stride may include data from pixels other than this
            // particular plane and row, and that could be between pixels and not after every
            // pixel:
            //
            // |---- Pixel stride ----|                    Row ends here --> |
            // | Pixel 1 | Other Data | Pixel 2 | Other Data | ... | Pixel N |
            //
            // We need to get (N-1) * (pixel stride bytes) per row + 1 byte for the last pixel
            (planeWidth - 1) * pixelStride + 1
        }

        for (row in 0 until planeHeight) {
            // Move buffer position to the beginning of this row
            planeBuffer.position(
                    (row + planeCrop.top) * rowStride + planeCrop.left * pixelStride)

            if (pixelStride == 1 && outputStride == 1) {
                // When there is a single stride value for pixel and output, we can just copy
                // the entire row in a single step
                planeBuffer.get(outputBuffer, outputOffset, rowLength)
                outputOffset += rowLength
            } else {
                // When either pixel or output have a stride > 1 we must copy pixel by pixel
                planeBuffer.get(rowBuffer, 0, rowLength)
                for (col in 0 until planeWidth) {
                    outputBuffer[outputOffset] = rowBuffer[col * pixelStride]
                    outputOffset += outputStride
                }
            }
        }
    }
}

val cocoCategories: Map<Int, String> = mapOf(
    1 to "person",
    2 to "bicycle",
    3 to "car",
    4 to "motorcycle",
    5 to "airplane",
    6 to "bus",
    7 to "train",
    8 to "truck",
    9 to "boat",
    10 to "traffic light",
    11 to "fire hydrant",
    13 to "stop sign",
    14 to "parking meter",
    15 to "bench",
    16 to "bird",
    17 to "cat",
    18 to "dog",
    19 to "horse",
    20 to "sheep",
    21 to "cow",
    22 to "elephant",
    23 to "bear",
    24 to "zebra",
    25 to "giraffe",
    27 to "backpack",
    28 to "umbrella",
    31 to "handbag",
    32 to "tie",
    33 to "suitcase",
    34 to "frisbee",
    35 to "skis",
    36 to "snowboard",
    37 to "sports ball",
    38 to "kite",
    39 to "baseball bat",
    40 to "baseball glove",
    41 to "skateboard",
    42 to "surfboard",
    43 to "tennis racket",
    44 to "bottle",
    46 to "wine glass",
    47 to "cup",
    48 to "fork",
    49 to "knife",
    50 to "spoon",
    51 to "bowl",
    52 to "banana",
    53 to "apple",
    54 to "sandwich",
    55 to "orange",
    56 to "broccoli",
    57 to "carrot",
    58 to "hot dog",
    59 to "pizza",
    60 to "donut",
    61 to "cake",
    62 to "chair",
    63 to "couch",
    64 to "potted plant",
    65 to "bed",
    67 to "dining table",
    70 to "toilet",
    72 to "tv",
    73 to "laptop",
    74 to "mouse",
    75 to "remote",
    76 to "keyboard",
    77 to "cell phone",
    78 to "microwave",
    79 to "oven",
    80 to "toaster",
    81 to "sink",
    82 to "refrigerator",
    84 to "book",
    85 to "clock",
    86 to "vase",
    87 to "scissors",
    88 to "teddy bear",
    89 to "hair drier",
    90 to "toothbrush"
)
