package com.sonict.publicmonitor.monitor

import android.Manifest
import android.app.Notification
import android.app.Service
import android.content.Intent
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.IBinder
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat
import com.sonict.publicmonitor.notification.RiskNotification
import kotlin.concurrent.thread

class SpeakerMonitorService : Service() {
    companion object {
        const val ACTION_START = "sonict.monitor.START"
        const val ACTION_STOP = "sonict.monitor.STOP"
        private const val NOTIFICATION_ID = 7001
    }

    @Volatile private var running = false
    private var recorder: AudioRecord? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            stopCapture()
            return START_NOT_STICKY
        }
        if (!running) startCapture()
        return START_NOT_STICKY
    }

    private fun startCapture() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            stopSelf()
            return
        }

        RiskNotification.createChannels(this)
        startForeground(NOTIFICATION_ID, monitorNotification("Starting offline monitor…"))
        running = true

        thread(name = "SonicT-Audio-Monitor") {
            val sampleRate = 16_000
            val chunkSamples = sampleRate * 5
            val audio = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                maxOf(chunkSamples * 2, AudioRecord.getMinBufferSize(
                    sampleRate,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT
                ))
            )
            recorder = audio
            val shorts = ShortArray(chunkSamples)
            val model = MobileModelRunner(this)

            try {
                audio.startRecording()
                while (running) {
                    var offset = 0
                    while (running && offset < shorts.size) {
                        val read = audio.read(shorts, offset, shorts.size - offset)
                        if (read <= 0) break
                        offset += read
                    }
                    if (offset == shorts.size) {
                        val pcm = FloatArray(shorts.size) { shorts[it] / 32768f }
                        val message = when (val result = model.analyze(pcm)) {
                            is ModelResult.Available -> when {
                                result.risk >= 80 -> "Critical audio risk ${result.risk}/100 — end and verify"
                                result.risk >= 60 -> "High audio risk ${result.risk}/100 — verify caller"
                                else -> "Dynamic audio risk ${result.risk}/100"
                            }
                            is ModelResult.Unavailable -> "Model unavailable — metadata screening remains active"
                        }
                        NotificationManagerCompat.from(this)
                            .notify(NOTIFICATION_ID, monitorNotification(message))
                    }
                }
            } catch (_: Exception) {
                // The UI continues to show metadata screening even if a device blocks mic access.
            } finally {
                runCatching { audio.stop() }
                audio.release()
                model.close()
                recorder = null
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf()
            }
        }
    }

    private fun monitorNotification(message: String): Notification =
        NotificationCompat.Builder(this, RiskNotification.CHANNEL_MONITOR)
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setContentTitle("SonicT speaker-mode monitoring")
            .setContentText(message)
            .setOngoing(true)
            .build()

    private fun stopCapture() {
        running = false
        runCatching { recorder?.stop() }
    }

    override fun onDestroy() {
        stopCapture()
        super.onDestroy()
    }
}
