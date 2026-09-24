package com.sonict.publicmonitor.risk

enum class RiskLevel { LOW, MEDIUM, HIGH, CRITICAL }

data class RiskAssessment(
    val score: Int,
    val level: RiskLevel,
    val reasons: List<String>
) {
    val recommendation: String
        get() = when (level) {
            RiskLevel.LOW -> "No major risk detected"
            RiskLevel.MEDIUM -> "Verify the caller before sharing sensitive information"
            RiskLevel.HIGH -> "Suspicious call — use trusted verification"
            RiskLevel.CRITICAL -> "Critical risk — end the call and consider blocking"
        }
}
