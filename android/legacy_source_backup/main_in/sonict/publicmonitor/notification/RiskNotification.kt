package com.sonict.publicmonitor.notification

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import com.sonict.publicmonitor.MainActivity
import com.sonict.publicmonitor.data.CallerDetails
import com.sonict.publicmonitor.risk.RiskAssessment

object RiskNotification {
    const val CHANNEL_RISK = "sonict_risk"
    const val CHANNEL_MONITOR = "sonict_monitor"
    const val ACTION_BLOCK = "com.sonict.publicmonitor.BLOCK"
    const val ACTION_REPORT = "com.sonict.publicmonitor.REPORT"
    const val EXTRA_NUMBER = "number"

    fun createChannels(context: Context) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val manager = context.getSystemService(NotificationManager::class.java)
        manager.createNotificationChannels(
            listOf(
                NotificationChannel(CHANNEL_RISK, "Call risk alerts", NotificationManager.IMPORTANCE_HIGH),
                NotificationChannel(CHANNEL_MONITOR, "Live monitoring", NotificationManager.IMPORTANCE_LOW)
            )
        )
    }

    fun showIncoming(context: Context, caller: CallerDetails, assessment: RiskAssessment) {
        createChannels(context)
        val openIntent = PendingIntent.getActivity(
            context,
            10,
            Intent(context, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        val blockIntent = PendingIntent.getBroadcast(
            context,
            caller.number.hashCode(),
            Intent(context, CallActionReceiver::class.java)
                .setAction(ACTION_BLOCK)
                .putExtra(EXTRA_NUMBER, caller.number),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        val reportIntent = PendingIntent.getBroadcast(
            context,
            caller.number.hashCode() + 1,
            Intent(context, CallActionReceiver::class.java)
                .setAction(ACTION_REPORT)
                .putExtra(EXTRA_NUMBER, caller.number),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val notification = NotificationCompat.Builder(context, CHANNEL_RISK)
            .setSmallIcon(android.R.drawable.stat_sys_warning)
            .setContentTitle("${assessment.level}: Dynamic risk ${assessment.score}/100")
            .setContentText("${caller.displayName} • ${caller.approximateRegion}")
            .setStyle(
                NotificationCompat.BigTextStyle().bigText(
                    "${caller.number}\n${caller.type}\n${assessment.recommendation}\n" +
                        assessment.reasons.joinToString(" • ")
                )
            )
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setContentIntent(openIntent)
            .setAutoCancel(true)
            .addAction(0, "Report", reportIntent)
            .addAction(0, "Block future calls", blockIntent)
            .build()

        runCatching { NotificationManagerCompat.from(context).notify(caller.number.hashCode(), notification) }
    }
}
