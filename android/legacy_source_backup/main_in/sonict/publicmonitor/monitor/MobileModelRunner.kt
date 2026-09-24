package com.sonict.publicmonitor.monitor

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

sealed interface ModelResult {
    data class Available(val risk: Int) : ModelResult
    data class Unavailable(val reason: String) : ModelResult
}

/**
 * Expected example contract: input [1, 80000] Float32 PCM at 16 kHz,
 * output [1, 1] deepfake probability. Adapt this class to your exported model.
 */
class MobileModelRunner(context: Context) : AutoCloseable {
    private val interpreter: Interpreter? = runCatching {
        val bytes = context.assets.open("sonict_mobile.tflite").use { it.readBytes() }
        val buffer = ByteBuffer.allocateDirect(bytes.size)
            .order(ByteOrder.nativeOrder())
            .apply {
                put(bytes)
                rewind()
            }
        Interpreter(buffer)
    }.getOrNull()

    fun analyze(pcm: FloatArray): ModelResult {
        val model = interpreter
            ?: return ModelResult.Unavailable("Add sonict_mobile.tflite to app/src/main/assets")
        if (pcm.size != 80_000) {
            return ModelResult.Unavailable("Model input requires exactly 80,000 samples")
        }

        return runCatching {
            val output = Array(1) { FloatArray(1) }
            model.run(arrayOf(pcm), output)
            ModelResult.Available((output[0][0] * 100f).toInt().coerceIn(0, 100))
        }.getOrElse {
            ModelResult.Unavailable("Model shape does not match the mobile adapter")
        }
    }

    override fun close() = interpreter?.close() ?: Unit
}
