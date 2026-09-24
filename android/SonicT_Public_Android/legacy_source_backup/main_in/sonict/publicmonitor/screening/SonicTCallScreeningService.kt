package com.sonict.publicmonitor.screening

import android.telecom.Call
import android.telecom.CallScreeningService
import android.telecom.Connection
import com.sonict.publicmonitor.data.CallerLookup
import com.sonict.publicmonitor.data.LocalSecurityStore
import com.sonict.publicmonitor.notification.RiskNotification
import com.sonict.publicmonitor.risk.CallerRiskInput
import com.sonict.publicmonitor.risk.OfflineRiskEngine
import com.sonict.publicmonitor.risk.RiskLevel

class SonicTCallScreeningService : CallScreeningService() {
    override fun onScreenCall(callDetails: Call.Details) {
        val number = callDetails.handle?.schemeSpecificPart.orEmpty()
        val caller = CallerLookup(this).find(number)
        val store = LocalSecurityStore(this)

        val verificationFailed =
            callDetails.callerNumberVerificationStatus == Connection.VERIFICATION_STATUS_FAILED

        val assessment = OfflineRiskEngine.calculate(
            CallerRiskInput(
                knownContact = caller.knownContact,
                verificationFailed = verificationFailed,
                locallyBlocked = store.isBlocked(number),
                previousIncidentCount = store.incidentCount(number)
            )
        )

        val autoBlock = store.autoBlockCritical && assessment.level == RiskLevel.CRITICAL
        val silence = !autoBlock && assessment.level in setOf(RiskLevel.HIGH, RiskLevel.CRITICAL)

        respondToCall(
            callDetails,
            CallResponse.Builder()
                .setDisallowCall(autoBlock)
                .setRejectCall(autoBlock)
                .setSilenceCall(silence)
                .build()
        )

        RiskNotification.showIncoming(this, caller, assessment)
    }
}
