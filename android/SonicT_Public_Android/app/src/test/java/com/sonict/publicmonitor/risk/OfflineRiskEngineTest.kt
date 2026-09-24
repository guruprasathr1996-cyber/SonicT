package com.sonict.publicmonitor.risk

import org.junit.Assert.assertEquals
import org.junit.Test

class OfflineRiskEngineTest {
    @Test fun knownCallerWithoutWarningsIsLow() {
        val result = OfflineRiskEngine.calculate(CallerRiskInput(true, false, false, 0))
        assertEquals(0, result.score)
        assertEquals(RiskLevel.LOW, result.level)
    }

    @Test fun blockedFailedUnknownCallerIsCritical() {
        val result = OfflineRiskEngine.calculate(CallerRiskInput(false, true, true, 1))
        assertEquals(100, result.score)
        assertEquals(RiskLevel.CRITICAL, result.level)
    }
}
