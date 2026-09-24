package com.sonict.publicmonitor.notification

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.widget.Toast
import com.sonict.publicmonitor.data.LocalSecurityStore

class CallActionReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        val number = intent.getStringExtra(RiskNotification.EXTRA_NUMBER) ?: return
        val store = LocalSecurityStore(context)
        when (intent.action) {
            RiskNotification.ACTION_BLOCK -> {
                store.block(number)
                Toast.makeText(context, "Future calls blocked in SonicT", Toast.LENGTH_LONG).show()
            }
            RiskNotification.ACTION_REPORT -> {
                store.reportIncident(number)
                Toast.makeText(context, "Incident saved locally", Toast.LENGTH_LONG).show()
            }
        }
    }
}
