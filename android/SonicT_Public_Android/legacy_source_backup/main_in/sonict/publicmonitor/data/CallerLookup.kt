package com.sonict.publicmonitor.data

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.net.Uri
import android.provider.ContactsContract
import androidx.core.content.ContextCompat

data class CallerDetails(
    val number: String,
    val displayName: String,
    val knownContact: Boolean,
    val type: String,
    val approximateRegion: String
)

class CallerLookup(private val context: Context) {
    fun find(number: String): CallerDetails {
        val region = when {
            number.startsWith("+91") -> "India (+91)"
            number.startsWith("+1") -> "North America (+1)"
            number.startsWith("+") -> "International number"
            else -> "Region unavailable"
        }

        if (ContextCompat.checkSelfPermission(context, Manifest.permission.READ_CONTACTS)
            != PackageManager.PERMISSION_GRANTED
        ) return unknown(number, region)

        val uri = Uri.withAppendedPath(
            ContactsContract.PhoneLookup.CONTENT_FILTER_URI,
            Uri.encode(number)
        )

        context.contentResolver.query(
            uri,
            arrayOf(
                ContactsContract.PhoneLookup.DISPLAY_NAME,
                ContactsContract.PhoneLookup.TYPE
            ),
            null,
            null,
            null
        )?.use { cursor ->
            if (cursor.moveToFirst()) {
                val name = cursor.getString(0) ?: "Known caller"
                val typeValue = cursor.getInt(1)
                val type = ContactsContract.CommonDataKinds.Phone
                    .getTypeLabel(context.resources, typeValue, "Phone").toString()
                return CallerDetails(number, name, true, type, region)
            }
        }
        return unknown(number, region)
    }

    private fun unknown(number: String, region: String) = CallerDetails(
        number = number,
        displayName = "Unknown caller",
        knownContact = false,
        type = "Mobile/phone type unavailable",
        approximateRegion = region
    )
}
