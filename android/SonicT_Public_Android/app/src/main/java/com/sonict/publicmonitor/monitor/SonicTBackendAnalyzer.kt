package com.sonict.publicmonitor.monitor

import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.util.UUID

sealed interface BackendResult {
    data class Success(
        val classification: String,
        val confidence: Double,
        val riskScore: Double,
        val riskLevel: String,
        val recommendation: String
    ) : BackendResult {
        fun notificationText(): String {
            val risk = riskScore.toInt().coerceIn(0, 100)
            val name = classification.uppercase()
            return when {
                risk >= 80 -> "CRITICAL $name • $risk/100 — end call and verify"
                risk >= 60 -> "HIGH $name • $risk/100 — verify caller"
                risk >= 30 -> "$name • $risk/100 — verification advised"
                else -> "$name • risk $risk/100"
            }
        }
    }

    data class Unavailable(val reason: String) : BackendResult
}

/** Sends an in-memory five-second WAV chunk to SonicT's FastAPI live endpoint. */
class SonicTBackendAnalyzer(
    private val baseUrl: String,
    private val apiKey: String
) {
    fun analyze(samples: ShortArray): BackendResult {
        if (baseUrl.isBlank()) return BackendResult.Unavailable("Backend is not configured")

        return runCatching {
            val boundary = "SonicT-${UUID.randomUUID()}"
            val connection = URL("${baseUrl.trimEnd('/')}/analyze-live-chunk")
                .openConnection() as HttpURLConnection

            connection.requestMethod = "POST"
            connection.connectTimeout = 15_000
            connection.readTimeout = 120_000
            connection.doOutput = true
            connection.setRequestProperty("Accept", "application/json")
            connection.setRequestProperty("X-API-Key", apiKey)
            connection.setRequestProperty("Content-Type", "multipart/form-data; boundary=$boundary")

            val wav = WavEncoder.pcm16Mono(samples, 16_000)
            connection.outputStream.buffered().use { output ->
                output.write("--$boundary\r\n".toByteArray())
                output.write(
                    "Content-Disposition: form-data; name=\"file\"; filename=\"live_chunk.wav\"\r\n"
                        .toByteArray()
                )
                output.write("Content-Type: audio/wav\r\n\r\n".toByteArray())
                output.write(wav)
                output.write("\r\n--$boundary--\r\n".toByteArray())
            }

            val status = connection.responseCode
            val stream = if (status in 200..299) connection.inputStream else connection.errorStream
            val body = stream?.bufferedReader()?.use { it.readText() }.orEmpty()
            connection.disconnect()
            if (status !in 200..299) error("SonicT returned HTTP $status")

            val json = JSONObject(body)
            BackendResult.Success(
                classification = json.optString("classification", "UNKNOWN"),
                confidence = json.optDouble("confidence", 0.0),
                riskScore = json.optDouble("risk_score", 0.0),
                riskLevel = json.optString("risk_level", "UNKNOWN"),
                recommendation = json.optString("recommendation", "Verify the caller")
            )
        }.getOrElse { BackendResult.Unavailable(it.message ?: "Connection failed") }
    }
}

private object WavEncoder {
    fun pcm16Mono(samples: ShortArray, sampleRate: Int): ByteArray {
        val pcmSize = samples.size * 2
        val out = ByteArrayOutputStream(44 + pcmSize)
        out.write("RIFF".toByteArray())
        writeInt(out, 36 + pcmSize)
        out.write("WAVEfmt ".toByteArray())
        writeInt(out, 16)
        writeShort(out, 1)
        writeShort(out, 1)
        writeInt(out, sampleRate)
        writeInt(out, sampleRate * 2)
        writeShort(out, 2)
        writeShort(out, 16)
        out.write("data".toByteArray())
        writeInt(out, pcmSize)
        samples.forEach { writeShort(out, it.toInt()) }
        return out.toByteArray()
    }

    private fun writeInt(out: ByteArrayOutputStream, value: Int) {
        out.write(value and 0xff)
        out.write((value ushr 8) and 0xff)
        out.write((value ushr 16) and 0xff)
        out.write((value ushr 24) and 0xff)
    }

    private fun writeShort(out: ByteArrayOutputStream, value: Int) {
        out.write(value and 0xff)
        out.write((value ushr 8) and 0xff)
    }
}
