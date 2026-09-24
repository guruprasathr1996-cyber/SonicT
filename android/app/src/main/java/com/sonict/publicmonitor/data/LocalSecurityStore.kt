package com.sonict.publicmonitor.data

import android.content.Context
import java.security.MessageDigest

class LocalSecurityStore(context: Context) {
    private val preferences = context.getSharedPreferences("sonict_security", Context.MODE_PRIVATE)

    var autoBlockCritical: Boolean
        get() = preferences.getBoolean("auto_block_critical", false)
        set(value) = preferences.edit().putBoolean("auto_block_critical", value).apply()

    var backendUrl: String
        get() = preferences.getString("backend_url", "") ?: ""
        set(value) = preferences.edit()
            .putString("backend_url", value.trim().trimEnd('/'))
            .apply()

    var apiKey: String
        get() = preferences.getString("api_key", "sonict-demo-2026") ?: "sonict-demo-2026"
        set(value) = preferences.edit().putString("api_key", value.trim()).apply()

    fun block(number: String) = preferences.edit()
        .putBoolean("blocked_${hash(number)}", true)
        .apply()

    fun isBlocked(number: String): Boolean =
        preferences.getBoolean("blocked_${hash(number)}", false)

    fun reportIncident(number: String) {
        val key = "incident_${hash(number)}"
        preferences.edit().putInt(key, incidentCount(number) + 1).apply()
    }

    fun incidentCount(number: String): Int =
        preferences.getInt("incident_${hash(number)}", 0)

    private fun hash(value: String): String {
        val bytes = MessageDigest.getInstance("SHA-256").digest(value.toByteArray())
        return bytes.joinToString("") { "%02x".format(it) }
    }
}
