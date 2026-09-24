package com.sonict.publicmonitor

import android.Manifest
import android.app.role.RoleManager
import android.content.Intent
import android.graphics.Color
import android.os.Build
import android.os.Bundle
import android.view.ViewGroup
import android.widget.Button
import android.widget.CheckBox
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.sonict.publicmonitor.data.LocalSecurityStore
import com.sonict.publicmonitor.monitor.SpeakerMonitorService
import com.sonict.publicmonitor.notification.RiskNotification

class MainActivity : AppCompatActivity() {
    private val permissionsLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { }

    private val roleLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { recreate() }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        RiskNotification.createChannels(this)
        requestAppPermissions()

        val store = LocalSecurityStore(this)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 64, 48, 48)
            setBackgroundColor(Color.rgb(247, 251, 250))
        }

        root.addView(label("SonicT Call Guard", 28f, true))
        root.addView(label("Offline caller risk monitoring for public users", 16f, false))
        root.addView(spacer())

        val roleManager = getSystemService(RoleManager::class.java)
        val roleHeld = roleManager.isRoleHeld(RoleManager.ROLE_CALL_SCREENING)
        root.addView(label(
            if (roleHeld) "✓ Call screening is active" else "Call screening is not enabled",
            18f,
            true
        ))

        root.addView(Button(this).apply {
            text = "Enable SonicT call screening"
            isEnabled = !roleHeld
            setOnClickListener {
                roleLauncher.launch(roleManager.createRequestRoleIntent(RoleManager.ROLE_CALL_SCREENING))
            }
        })

        root.addView(CheckBox(this).apply {
            text = "Automatically reject critical-risk calls"
            isChecked = store.autoBlockCritical
            setOnCheckedChangeListener { _, checked -> store.autoBlockCritical = checked }
        })

        root.addView(spacer())
        root.addView(label("Live speaker-mode check", 20f, true))
        root.addView(label(
            "Place the call on speaker, get consent where required, then start. " +
                "This mode cannot directly capture the protected cellular call stream.",
            14f,
            false
        ))

        root.addView(Button(this).apply {
            text = "Start 5-second offline monitoring"
            setOnClickListener {
                val intent = Intent(this@MainActivity, SpeakerMonitorService::class.java)
                    .setAction(SpeakerMonitorService.ACTION_START)
                ContextCompat.startForegroundService(this@MainActivity, intent)
            }
        })
        root.addView(Button(this).apply {
            text = "Stop monitoring"
            setOnClickListener {
                startService(
                    Intent(this@MainActivity, SpeakerMonitorService::class.java)
                        .setAction(SpeakerMonitorService.ACTION_STOP)
                )
            }
        })

        root.addView(spacer())
        root.addView(label(
            "Privacy: caller numbers are hashed before block/incident history is stored. " +
                "Raw audio is processed in memory and is not saved by this prototype.",
            13f,
            false
        ))

        setContentView(root)
    }

    private fun requestAppPermissions() {
        val permissions = mutableListOf(Manifest.permission.READ_CONTACTS, Manifest.permission.RECORD_AUDIO)
        if (Build.VERSION.SDK_INT >= 33) permissions += Manifest.permission.POST_NOTIFICATIONS
        permissionsLauncher.launch(permissions.toTypedArray())
    }

    private fun label(text: String, size: Float, bold: Boolean) = TextView(this).apply {
        this.text = text
        textSize = size
        setTextColor(Color.rgb(7, 29, 43))
        if (bold) setTypeface(typeface, android.graphics.Typeface.BOLD)
        setPadding(0, 8, 0, 8)
    }

    private fun spacer() = TextView(this).apply {
        layoutParams = ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 32)
    }
}
