package com.sonict.publicmonitor.risk

data class CallerRiskInput(
    val knownContact: Boolean,
    val verificationFailed: Boolean,
    val locallyBlocked: Boolean,
    val previousIncidentCount: Int,
    val audioRisk: Int? = null
)

object OfflineRiskEngine {
    fun calculate(input: CallerRiskInput): RiskAssessment {
        var metadataScore = 0
        val reasons = mutableListOf<String>()

        if (!input.knownContact) {
            metadataScore += 20
            reasons += "Number is not in contacts"
        }
        if (input.verificationFailed) {
            metadataScore += 35
            reasons += "Network number verification failed"
        }
        if (input.locallyBlocked) {
            metadataScore += 50
            reasons += "Number is in the local SonicT block list"
        }
        if (input.previousIncidentCount > 0) {
            metadataScore += (input.previousIncidentCount * 10).coerceAtMost(30)
            reasons += "Previous local incident reports: ${input.previousIncidentCount}"
        }

        // Audio is used only when a real mobile model returned a result.
        val score = if (input.audioRisk == null) {
            metadataScore.coerceIn(0, 100)
        } else {
            (metadataScore * 0.35f + input.audioRisk * 0.65f).toInt().coerceIn(0, 100)
        }

        val level = when (score) {
            in 0..29 -> RiskLevel.LOW
            in 30..59 -> RiskLevel.MEDIUM
            in 60..79 -> RiskLevel.HIGH
            else -> RiskLevel.CRITICAL
        }

        if (reasons.isEmpty()) reasons += "No local warning indicators"
        return RiskAssessment(score, level, reasons)
    }
}
