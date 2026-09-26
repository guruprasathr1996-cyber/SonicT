import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

const DEMO_TOKEN = "sonict-offline-demo";

const DEMO_RESULT = {
  input_file: "ai_cloned_voice_demo.wav",
  classification: "DEEPFAKE",
  confidence: 0.91,
  class_probabilities: { genuine: 0.09, deepfake: 0.91 },
  features: {
    voice_clone_probability: 0.91,
    spectrogram_probability: 0.84,
    voice_feature_probability: 0.79,
    replay_probability: 0.72,
  },
  tampering: {
    f4_max: 0.93,
    f4_mean: 0.76,
    f4_median: 0.79,
    f4_std: 0.12,
    f4_suspicious_ratio: 0.68,
    f4_high_ratio: 0.44,
    suspicious_windows: [
      { start: 5, end: 10, score: 0.82 },
      { start: 15, end: 20, score: 0.93 },
    ],
  },
  voice_integrity_risk: {
    risk_score: 87,
    risk_level: "CRITICAL",
    explanation: "Multiple models detected synthetic voice and editing indicators.",
  },
  security_alert: {
    alert: true,
    message: "Critical voice-integrity risk. End the call and verify independently.",
  },
  operational_assessment: {
    status: "VERIFICATION_REQUIRED",
    recommended_action: "Do not authorize sensitive actions. Call back using a trusted number.",
  },
  evidence_integrity: {
    sha256: "demo-8a5d4f3b2c1e7096a814c8d9ef02a1b7",
    status: "VERIFIED",
  },
  privacy_and_retention: {
    audio_stored: false,
    message: "Demo sample only. No audio is uploaded or stored.",
  },
  experimental_f1b: {
    prediction: "AI-GENERATED",
    deepfake_probability: 0.89,
    label: "AI-GENERATED",
    probability: 0.89,
  },
  forensic_report: {
    conclusion: "Strong evidence of voice cloning and post-processing tampering.",
    recommended_action: "Use trusted-channel identity verification before continuing.",
  },
  incident: {
    incident_uuid: "DEMO-INC-001",
    case_status: "OPEN",
    verification_status: "PENDING",
  },
};

const DEMO_GENUINE_RESULT = {
  ...DEMO_RESULT,
  input_file: "genuine_voice_demo.wav",
  classification: "GENUINE",
  confidence: 0.96,
  class_probabilities: { genuine: 0.96, deepfake: 0.04 },
  features: {
    voice_clone_probability: 0.04,
    spectrogram_probability: 0.08,
    voice_feature_probability: 0.07,
    replay_probability: 0.03,
  },
  tampering: {
    f4_max: 0.11,
    f4_mean: 0.05,
    f4_median: 0.04,
    f4_std: 0.02,
    f4_suspicious_ratio: 0,
    f4_high_ratio: 0,
    suspicious_windows: [],
  },
  voice_integrity_risk: {
    risk_score: 8,
    risk_level: "LOW",
    explanation: "The forensic models found consistent natural speech characteristics.",
  },
  security_alert: { alert: false, message: "No major voice-integrity risk detected." },
  operational_assessment: {
    status: "NORMAL",
    recommended_action: "No additional action is required for this demonstration sample.",
  },
  experimental_f1b: {
    prediction: "BONAFIDE",
    deepfake_probability: 0.04,
    label: "BONAFIDE",
    probability: 0.96,
  },
  forensic_report: {
    conclusion: "The sample is consistent with genuine human speech.",
    recommended_action: "Continue normal verification procedures.",
  },
  incident: null,
};

const DEMO_TAMPERED_RESULT = {
  ...DEMO_RESULT,
  input_file: "edited_call_recording_demo.wav",
  classification: "TAMPERED",
  confidence: 0.84,
  class_probabilities: { genuine: 0.16, tampered: 0.84 },
  features: {
    voice_clone_probability: 0.29,
    spectrogram_probability: 0.81,
    voice_feature_probability: 0.62,
    replay_probability: 0.18,
  },
  tampering: {
    f4_max: 0.94,
    f4_mean: 0.71,
    f4_median: 0.75,
    f4_std: 0.15,
    f4_suspicious_ratio: 0.63,
    f4_high_ratio: 0.41,
    suspicious_windows: [
      { start: 10, end: 15, score: 0.86 },
      { start: 25, end: 30, score: 0.94 },
    ],
  },
  voice_integrity_risk: {
    risk_score: 74,
    risk_level: "HIGH",
    explanation: "Abrupt spectral changes indicate possible cutting and splicing.",
  },
  security_alert: { alert: true, message: "Possible audio editing detected. Verify the original recording." },
  operational_assessment: {
    status: "VERIFICATION_REQUIRED",
    recommended_action: "Compare the recording with its original source before using it as evidence.",
  },
  experimental_f1b: {
    prediction: "BONAFIDE",
    deepfake_probability: 0.29,
    label: "BONAFIDE",
    probability: 0.71,
  },
  forensic_report: {
    conclusion: "The recording contains likely editing or splicing regions.",
    recommended_action: "Review the highlighted time regions and obtain the original file.",
  },
  incident: {
    incident_uuid: "DEMO-INC-002",
    case_status: "OPEN",
    verification_status: "PENDING",
  },
};

const DEMO_SAMPLES = {
  genuine: { label: "Genuine Voice", description: "Natural human speech", result: DEMO_GENUINE_RESULT },
  cloned: { label: "AI-Cloned Voice", description: "Synthetic impersonation", result: DEMO_RESULT },
  tampered: { label: "Tampered Audio", description: "Edited and spliced call", result: DEMO_TAMPERED_RESULT },
};

const createDemoReport = (sampleResult) => [{
  id: `DEMO-RPT-${sampleResult.classification}`,
  filename: sampleResult.input_file,
  created_at: "2026-09-25T09:30:00Z",
  classification: sampleResult.classification,
  confidence: sampleResult.confidence,
  class_probabilities: sampleResult.class_probabilities,
  forensic_conclusion: sampleResult.forensic_report.conclusion,
  recommended_action: sampleResult.forensic_report.recommended_action,
  disclaimer: "Pre-analysed demonstration data; not a live forensic result.",
}];

const createDemoIncidents = (sampleResult) => sampleResult.incident ? [{
  incident_uuid: sampleResult.incident.incident_uuid,
  filename: sampleResult.input_file,
  timestamp: "2026-09-25T09:30:00Z",
  classification: sampleResult.classification,
  confidence: sampleResult.confidence,
  risk_score: sampleResult.voice_integrity_risk.risk_score,
  risk_level: sampleResult.voice_integrity_risk.risk_level,
  case_status: "OPEN",
  verification_status: "PENDING",
  investigator_notes: "Offline prototype demonstration incident.",
  resolution: "INCONCLUSIVE",
}] : [];

const DEMO_REPORTS = createDemoReport(DEMO_RESULT);

const DEMO_INCIDENTS = createDemoIncidents(DEMO_RESULT);

function App() {
  const [demoMode, setDemoMode] = useState(
    () => sessionStorage.getItem("sonict_demo_mode") === "true"
  );
  const [demoSampleKey, setDemoSampleKey] = useState(
    () => sessionStorage.getItem("sonict_demo_sample") || "cloned"
  );
  const [activePage, setActivePage] = useState("dashboard");
  const [authMode, setAuthMode] = useState("signin");
  const [authToken, setAuthToken] = useState(
    () => sessionStorage.getItem("sonict_demo_mode") === "true"
      ? DEMO_TOKEN
      : localStorage.getItem("sonict_access_token") || sessionStorage.getItem("sonict_access_token") || ""
  );
  const [authForm, setAuthForm] = useState({
    name: "",
    email: "",
    password: "",
    confirmPassword: "",
    remember: false,
  });
  const [authError, setAuthError] = useState("");
  const [authLoading, setAuthLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [callerDetails, setCallerDetails] = useState({
    phone: "",
    location: "",
    type: "Mobile",
    knownContact: "Unknown",
  });
  const [contactBlocked, setContactBlocked] = useState(false);
  const [riskNoticeDismissed, setRiskNoticeDismissed] = useState(false);
  const [selectedModelKey, setSelectedModelKey] = useState("F1");
  const [adminPreferences, setAdminPreferences] = useState({
    sso: false,
    twoFactor: true,
    ipWhitelist: false,
    securityEmails: true,
    highRiskNotifications: true,
    systemHealthNotifications: true,
    shaFingerprinting: true,
    automaticCleanup: true,
    f1Enabled: true,
    f2Enabled: true,
    f3Enabled: true,
    f4Enabled: true,
    f5Enabled: true,
  });
  const [adminConfig, setAdminConfig] = useState({
    organizationName: "SonicT Audio Forensics",
    supportEmail: "security@sonict.local",
    timezone: "Asia/Kolkata",
    dateFormat: "DD/MM/YYYY",
    defaultRole: "Investigator",
    sessionTimeoutMinutes: "30",
    maxFailedLogins: "5",
    passwordExpiryDays: "90",
    highRiskThreshold: "60",
    criticalRiskThreshold: "80",
    f4SuspiciousThreshold: "0.50",
    f4HighThreshold: "0.70",
    liveChunkSeconds: "5",
    maxUploadMb: "100",
    temporaryRetentionHours: "24",
    reportRetentionDays: "365",
    autoIncidentThreshold: "60",
    otpExpiryMinutes: "5",
    otpMaxAttempts: "5",
    otpResendCooldown: "60",
  });
  const [adminSaveMessage, setAdminSaveMessage] = useState("");

  const [file, setFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState("");

  const [result, setResult] = useState(() =>
    sessionStorage.getItem("sonict_demo_mode") === "true"
      ? (DEMO_SAMPLES[sessionStorage.getItem("sonict_demo_sample")] || DEMO_SAMPLES.cloned).result
      : null
  );

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // ======================================================
  // CHUNK ANALYSIS STATES
  // ======================================================

  const [chunkResult, setChunkResult] = useState(null);
  const [chunkLoading, setChunkLoading] = useState(false);
  const [chunkError, setChunkError] = useState("");

  // ======================================================
  // HUMAN MIMICRY ANALYSIS STATES
  // ======================================================

  const [mimicryResult, setMimicryResult] = useState(null);
  const [mimicryLoading, setMimicryLoading] = useState(false);
  const [mimicryError, setMimicryError] = useState("");

  // ======================================================
  // LIVE MICROPHONE MONITORING STATES
  // ======================================================

  const [liveMonitoring, setLiveMonitoring] = useState(false);
  const [liveProcessing, setLiveProcessing] = useState(false);
  const [liveResult, setLiveResult] = useState(null);
  const [liveHistory, setLiveHistory] = useState([]);
  const [liveError, setLiveError] = useState("");
  const [liveChunkCount, setLiveChunkCount] = useState(0);
  const [androidSyncEnabled, setAndroidSyncEnabled] = useState(false);
  const [androidSyncStatus, setAndroidSyncStatus] = useState("Not connected");
  const [androidLastSeen, setAndroidLastSeen] = useState("");
  const androidLatestIdRef = useRef(null);

  // ======================================================
  // CONTINUOUS LIVE THREAT STATES
  // ======================================================

  const [liveContinuousThreat, setLiveContinuousThreat] = useState(false);
  const [liveSuspiciousSegments, setLiveSuspiciousSegments] = useState([]);

  const liveProcessedHistoryRef = useRef([]);

  const mediaStreamRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const liveTimerRef = useRef(null);
  const liveMonitoringRef = useRef(false);
  const liveQueueRef = useRef([]);
  const liveQueueProcessingRef = useRef(false);
  const lastNotificationBandRef = useRef("");

  useEffect(() => {
    const score = Number(liveResult?.risk_score || 0);
    const band = score >= 80 ? "critical" : score >= 60 ? "high" : "";

    if (!band) {
      lastNotificationBandRef.current = "";
      return;
    }

    setRiskNoticeDismissed(false);

    if (
      band !== lastNotificationBandRef.current &&
      typeof Notification !== "undefined" &&
      Notification.permission === "granted"
    ) {
      new Notification(
        band === "critical"
          ? "SonicT: Suspicious call detected"
          : "SonicT: Elevated caller risk",
        {
          body:
            band === "critical"
              ? `Dynamic Risk Score ${score.toFixed(0)}/100. Block the contact and verify independently.`
              : `Dynamic Risk Score ${score.toFixed(0)}/100. Do not share sensitive information.`,
        }
      );
    }

    lastNotificationBandRef.current = band;
  }, [liveResult]);

  const [dragActive, setDragActive] = useState(false);

  const [reports, setReports] = useState(() =>
    sessionStorage.getItem("sonict_demo_mode") === "true"
      ? createDemoReport((DEMO_SAMPLES[sessionStorage.getItem("sonict_demo_sample")] || DEMO_SAMPLES.cloned).result)
      : []
  );
  const [reportsLoading, setReportsLoading] = useState(false);
  const [reportsError, setReportsError] = useState("");

  const [reportDownloading, setReportDownloading] = useState(false);
  const [reportDownloadError, setReportDownloadError] = useState("");

  // ALERTS & INCIDENTS / TRUSTED-CHANNEL VERIFICATION
const [incidents, setIncidents] = useState(() =>
  sessionStorage.getItem("sonict_demo_mode") === "true"
    ? createDemoIncidents((DEMO_SAMPLES[sessionStorage.getItem("sonict_demo_sample")] || DEMO_SAMPLES.cloned).result)
    : []
);
const [incidentsLoading, setIncidentsLoading] = useState(false);
const [incidentsError, setIncidentsError] = useState("");
const [selectedIncident, setSelectedIncident] = useState(() =>
  sessionStorage.getItem("sonict_demo_mode") === "true"
    ? createDemoIncidents((DEMO_SAMPLES[sessionStorage.getItem("sonict_demo_sample")] || DEMO_SAMPLES.cloned).result)[0] || null
    : null
);
const [verificationLoading, setVerificationLoading] = useState(false);
const [verificationError, setVerificationError] = useState("");
const [verificationMessage, setVerificationMessage] = useState("");
const [developmentOtp, setDevelopmentOtp] = useState("");
const [otpInput, setOtpInput] = useState("");
const [incidentAudit, setIncidentAudit] = useState([]);
const [incidentAuditLoading, setIncidentAuditLoading] = useState(false);
const [incidentReportLoading, setIncidentReportLoading] = useState(false);
const [incidentActionLoading, setIncidentActionLoading] = useState(false);
const [investigatorNotes, setInvestigatorNotes] = useState("");
const [resolutionChoice, setResolutionChoice] = useState("INCONCLUSIVE");

  const [systemStatus, setSystemStatus] = useState({
    api: false,
    loading: false,
    message: "Not checked",
  });

const API_BASE =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") || "";

const API_AUTH_HEADERS = {
  "ngrok-skip-browser-warning": "true",
  "X-SonicT-Client": "web",
  ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
};

useEffect(() => {
  if (!androidSyncEnabled || demoMode) {
    return undefined;
  }

  let cancelled = false;

  const loadLatestAndroidResult = async () => {
    try {
      const response = await fetch(`${API_BASE}/mobile/latest`, {
        headers: {
          "ngrok-skip-browser-warning": "true",
          "X-SonicT-Client": "web",
          ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
        },
      });
      const data = await response.json().catch(() => null);

      if (!response.ok) {
        throw new Error(data?.detail || "Unable to read the Android result.");
      }
      if (cancelled) return;

      if (!data?.available || !data?.result) {
        setAndroidSyncStatus("Connected · waiting for APK analysis");
        return;
      }

      setAndroidSyncStatus("Connected · receiving APK results");
      setAndroidLastSeen(data.received_at || "");

      if (androidLatestIdRef.current === data.id) {
        return;
      }

      androidLatestIdRef.current = data.id;
      const mobileResult = {
        ...data.result,
        source_client: "android",
        received_at: data.received_at,
      };
      setLiveResult(mobileResult);
      setLiveHistory((previous) => [
        {
          ...mobileResult,
          timestamp: new Date(data.received_at || Date.now()).toLocaleTimeString(),
          chunk_number: previous.length + 1,
        },
        ...previous,
      ].slice(0, 20));
      setLiveChunkCount((count) => count + 1);
      setLiveError("");
    } catch (err) {
      if (!cancelled) {
        setAndroidSyncStatus("Connection error");
        setLiveError(err.message || "Unable to connect to Android monitoring.");
      }
    }
  };

  setAndroidSyncStatus("Connecting to APK results...");
  loadLatestAndroidResult();
  const intervalId = window.setInterval(loadLatestAndroidResult, 3000);

  return () => {
    cancelled = true;
    window.clearInterval(intervalId);
  };
}, [androidSyncEnabled, demoMode, API_BASE, authToken]);

  /* ======================================================
     ALERTS & INCIDENTS API
  ====================================================== */

const loadIncidents = async () => {
  if (demoMode) {
    setIncidents(createDemoIncidents(DEMO_SAMPLES[demoSampleKey]?.result || DEMO_RESULT));
    return;
  }
  setIncidentsLoading(true);
  setIncidentsError("");

  try {
    const response = await fetch(`${API_BASE}/incidents`, {
      headers: API_AUTH_HEADERS,
    });

    const data = await response.json().catch(() => null);

    if (!response.ok) {
      throw new Error(data?.detail || "Unable to load incidents.");
    }

    const list = Array.isArray(data)
      ? data
      : Array.isArray(data?.incidents)
      ? data.incidents
      : [];

    setIncidents(list);

    setSelectedIncident((current) => {
      if (!list.length) return null;
      if (!current) return list[0];

      return (
        list.find(
          (item) => item.incident_uuid === current.incident_uuid
        ) || list[0]
      );
    });
  } catch (err) {
    console.error(err);
    setIncidentsError(err.message || "Unable to load incidents.");
  } finally {
    setIncidentsLoading(false);
  }
};

const loadIncidentDetails = async (incidentUuid) => {
  if (demoMode) {
    const demoItems = createDemoIncidents(DEMO_SAMPLES[demoSampleKey]?.result || DEMO_RESULT);
    setSelectedIncident(demoItems.find((item) => item.incident_uuid === incidentUuid) || demoItems[0] || null);
    return;
  }
  if (!incidentUuid) return;

  try {
    const response = await fetch(
      `${API_BASE}/incidents/${incidentUuid}`,
      { headers: API_AUTH_HEADERS }
    );

    const data = await response.json().catch(() => null);

    if (!response.ok) {
      throw new Error(data?.detail || "Unable to load incident.");
    }

    const incident = data?.incident || data;
    setSelectedIncident(incident);

    setIncidents((previous) =>
      previous.map((item) =>
        item.incident_uuid === incident.incident_uuid
          ? incident
          : item
      )
    );

    await loadIncidentAudit(incidentUuid);
  } catch (err) {
    console.error(err);
    setIncidentsError(err.message || "Unable to load incident.");
  }
};

const loadIncidentAudit = async (incidentUuid) => {
  if (demoMode) {
    setIncidentAudit([]);
    return;
  }
  if (!incidentUuid) return;
  setIncidentAuditLoading(true);

  try {
    const response = await fetch(
      `${API_BASE}/incidents/${incidentUuid}/audit`,
      { headers: API_AUTH_HEADERS }
    );
    const data = await response.json().catch(() => null);
    if (!response.ok) {
      throw new Error(data?.detail || "Unable to load incident audit trail.");
    }
    setIncidentAudit(Array.isArray(data?.events) ? data.events : []);
  } catch (err) {
    console.error(err);
    setIncidentsError(err.message || "Unable to load incident audit trail.");
  } finally {
    setIncidentAuditLoading(false);
  }
};

const downloadIncidentReport = async () => {
  if (demoMode) {
    setVerificationMessage("Report download requires the live API. The on-screen incident is a pre-analysed demo sample.");
    return;
  }
  if (!selectedIncident?.incident_uuid) return;
  setIncidentReportLoading(true);
  setIncidentsError("");

  try {
    const response = await fetch(
      `${API_BASE}/incidents/${selectedIncident.incident_uuid}/report`,
      { headers: API_AUTH_HEADERS }
    );
    if (!response.ok) {
      const data = await response.json().catch(() => null);
      throw new Error(data?.detail || "Unable to download incident report.");
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `sonict_incident_${selectedIncident.incident_uuid}.json`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  } catch (err) {
    console.error(err);
    setIncidentsError(err.message || "Unable to download incident report.");
  } finally {
    setIncidentReportLoading(false);
  }
};

const updateIncidentLifecycle = async (action) => {
  if (demoMode) {
    setVerificationMessage(`Demo preview: incident action “${action}” was not saved.`);
    return;
  }
  if (!selectedIncident?.incident_uuid) return;
  setIncidentActionLoading(true);
  setIncidentsError("");

  try {
    const response = await fetch(
      `${API_BASE}/incidents/${selectedIncident.incident_uuid}/lifecycle`,
      {
        method: "POST",
        headers: {
          ...API_AUTH_HEADERS,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          action,
          investigator_notes: investigatorNotes,
          resolution: action === "RESOLVE" ? resolutionChoice : null,
        }),
      }
    );
    const data = await response.json().catch(() => null);
    if (!response.ok) {
      throw new Error(data?.detail || "Unable to update incident lifecycle.");
    }
    if (data?.incident) {
      setSelectedIncident(data.incident);
      setInvestigatorNotes(data.incident.investigator_notes || "");
      setIncidents((previous) =>
        previous.map((item) =>
          item.incident_uuid === data.incident.incident_uuid
            ? data.incident
            : item
        )
      );
    }
    await loadIncidentAudit(selectedIncident.incident_uuid);
  } catch (err) {
    console.error(err);
    setIncidentsError(err.message || "Unable to update incident lifecycle.");
  } finally {
    setIncidentActionLoading(false);
  }
};

const requestTrustedVerification = async () => {
  if (demoMode) {
    setVerificationMessage("Demo preview: trusted-channel verification requires the live API.");
    return;
  }
  if (!selectedIncident?.incident_uuid) return;

  setVerificationLoading(true);
  setVerificationError("");
  setVerificationMessage("");
  setDevelopmentOtp("");
  setOtpInput("");

  try {
    const response = await fetch(
      `${API_BASE}/verification/request`,
      {
        method: "POST",
        headers: {
          ...API_AUTH_HEADERS,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          incident_uuid: selectedIncident.incident_uuid,
        }),
      }
    );

    const data = await response.json().catch(() => null);

    if (!response.ok) {
      throw new Error(
        data?.detail || data?.message || "Unable to start verification."
      );
    }

    setVerificationMessage(
      data?.message || "Trusted-channel verification started."
    );

    if (data?.development_otp) {
      setDevelopmentOtp(String(data.development_otp));
    }

    await loadIncidentDetails(selectedIncident.incident_uuid);
  } catch (err) {
    console.error(err);
    setVerificationError(
      err.message || "Unable to start trusted-channel verification."
    );
  } finally {
    setVerificationLoading(false);
  }
};

const verifyTrustedOtp = async () => {
  if (demoMode) {
    setVerificationMessage("Demo preview: OTP verification requires the live API.");
    return;
  }
  if (!selectedIncident?.incident_uuid) return;

  const cleanOtp = otpInput.trim();

  if (!/^\d{6}$/.test(cleanOtp)) {
    setVerificationError("Enter the 6-digit verification OTP.");
    return;
  }

  setVerificationLoading(true);
  setVerificationError("");
  setVerificationMessage("");

  try {
    const response = await fetch(
      `${API_BASE}/verification/verify`,
      {
        method: "POST",
        headers: {
          ...API_AUTH_HEADERS,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          incident_uuid: selectedIncident.incident_uuid,
          otp: cleanOtp,
        }),
      }
    );

    const data = await response.json().catch(() => null);

    if (!response.ok) {
      throw new Error(
        data?.detail || data?.message || "Verification failed."
      );
    }

    setVerificationMessage(
      data?.message ||
        (data?.verified
          ? "Trusted-channel verification successful."
          : "Verification was not successful.")
    );

    if (data?.incident) {
      setSelectedIncident(data.incident);
      setIncidents((previous) =>
        previous.map((item) =>
          item.incident_uuid === data.incident.incident_uuid
            ? data.incident
            : item
        )
      );
    }

    if (data?.verified) {
      setDevelopmentOtp("");
      setOtpInput("");
    }
  } catch (err) {
    console.error(err);
    setVerificationError(err.message || "Unable to verify OTP.");
  } finally {
    setVerificationLoading(false);
  }
};

useEffect(() => {
  if (activePage === "alerts" && !demoMode) {
    loadIncidents();
  }
}, [activePage, demoMode]);

useEffect(() => {
  if (activePage === "alerts" && selectedIncident?.incident_uuid && !demoMode) {
    loadIncidentAudit(selectedIncident.incident_uuid);
  }
}, [activePage, selectedIncident?.incident_uuid, demoMode]);

useEffect(() => {
  setInvestigatorNotes(selectedIncident?.investigator_notes || "");
  setResolutionChoice(selectedIncident?.resolution || "INCONCLUSIVE");
}, [selectedIncident?.incident_uuid]);

  /* ======================================================
     SUPPORTED AUDIO FORMATS
  ====================================================== */

  const allowedExtensions = [
    ".wav",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".m4a",
    ".flac",
    ".aac",
    ".ogg",
    ".wma",
    ".opus",
    ".webm",
    ".mov",
    ".mkv",
    ".avi",
    ".3gp",
    ".3g2",
    ".ts",
    ".m2ts",
    ".mka",
    ".aiff",
    ".aif",
    ".caf",
    ".amr",
  ];


  /* ======================================================
     AUDIO PREVIEW
  ====================================================== */

  useEffect(() => {
    if (!file) {
      setAudioUrl("");
      return;
    }

    const url = URL.createObjectURL(file);

    setAudioUrl(url);

    return () => {
      URL.revokeObjectURL(url);
    };
  }, [file]);


  /* ======================================================
     LIVE MICROPHONE CLEANUP
  ====================================================== */

  useEffect(() => {
    return () => {
      liveMonitoringRef.current = false;

      if (liveTimerRef.current) {
        clearTimeout(liveTimerRef.current);
      }

      if (
        mediaRecorderRef.current &&
        mediaRecorderRef.current.state !== "inactive"
      ) {
        try {
          mediaRecorderRef.current.stop();
        } catch {
          // Ignore recorder shutdown errors.
        }
      }

      if (mediaStreamRef.current) {
        mediaStreamRef.current
          .getTracks()
          .forEach((track) => track.stop());
      }

      liveQueueRef.current = [];
    };
  }, []);


  /* ======================================================
     LIVE MICROPHONE ANALYSIS
  ====================================================== */

  const updateLiveContinuousThreat = (history) => {
    const suspiciousSegments = [];
    let currentSegment = null;

    history.forEach((item) => {
      const level = String(
        item.risk_level || ""
      ).toUpperCase();

      const suspicious = [
        "HIGH",
        "CRITICAL",
      ].includes(level);

      if (suspicious) {
        if (!currentSegment) {
          currentSegment = {
            start: item.start,
            end: item.end,
            chunks: 1,
            max_risk: Number(
              item.risk_score || 0
            ),
          };
        } else {
          currentSegment.end =
            item.end;

          currentSegment.chunks += 1;

          currentSegment.max_risk =
            Math.max(
              currentSegment.max_risk,
              Number(
                item.risk_score || 0
              )
            );
        }
      } else {
        if (
          currentSegment &&
          currentSegment.chunks >= 2
        ) {
          suspiciousSegments.push({
            ...currentSegment,
          });
        }

        currentSegment = null;
      }
    });

    if (
      currentSegment &&
      currentSegment.chunks >= 2
    ) {
      suspiciousSegments.push({
        ...currentSegment,
      });
    }

    setLiveSuspiciousSegments(
      suspiciousSegments
    );

    setLiveContinuousThreat(
      suspiciousSegments.length > 0
    );
  };


  const sendLiveChunk = async (blob) => {
    if (!blob || blob.size === 0) {
      return;
    }

    setLiveProcessing(true);

    const mimeType =
      blob.type ||
      "audio/webm";

    const extension =
      mimeType.includes("ogg")
        ? "ogg"
        : mimeType.includes("wav")
        ? "wav"
        : "webm";

    const liveFile = new File(
      [blob],
      `live_chunk_${Date.now()}.${extension}`,
      {
        type: mimeType,
      }
    );

    const formData = new FormData();

    formData.append(
      "file",
      liveFile
    );

    try {
      const response = await fetch(
        `${API_BASE}/analyze-live-chunk`,
        {
          method: "POST",
          headers: API_AUTH_HEADERS,
          body: formData,
        }
      );

      const data =
        await response.json().catch(
          () => null
        );

      if (!response.ok) {
        throw new Error(
          data?.detail ||
            "Live chunk analysis failed."
        );
      }

      const timestamp =
        new Date().toLocaleTimeString();

      const chunkNumber =
        liveProcessedHistoryRef.current.length + 1;

      const startTime =
        (chunkNumber - 1) * 5;

      const endTime =
        chunkNumber * 5;

      const historyItem = {
        ...data,
        timestamp,
        chunk_number: chunkNumber,
        start: startTime,
        end: endTime,
      };

      liveProcessedHistoryRef.current = [
        ...liveProcessedHistoryRef.current,
        historyItem,
      ];

      updateLiveContinuousThreat(
        liveProcessedHistoryRef.current
      );

      setLiveResult(data);

      setLiveHistory((previous) => [
        historyItem,
        ...previous,
      ].slice(0, 20));

      setLiveChunkCount(
        liveProcessedHistoryRef.current.length
      );

      setLiveError("");

    } catch (err) {
      console.error(err);

      setLiveError(
        err.message ||
          "Unable to analyze the live microphone chunk."
      );

    } finally {
      setLiveProcessing(false);
    }
  };


  const processLiveQueue = async () => {
    if (liveQueueProcessingRef.current) {
      return;
    }

    liveQueueProcessingRef.current = true;

    try {
      while (
        liveQueueRef.current.length > 0
      ) {
        const nextBlob =
          liveQueueRef.current.shift();

        await sendLiveChunk(
          nextBlob
        );
      }

    } finally {
      liveQueueProcessingRef.current = false;
    }
  };


  const recordNextLiveChunk = () => {
    if (
      !liveMonitoringRef.current ||
      !mediaStreamRef.current
    ) {
      return;
    }

    try {
      let recorderOptions = {};

      if (
        window.MediaRecorder &&
        MediaRecorder.isTypeSupported(
          "audio/webm;codecs=opus"
        )
      ) {
        recorderOptions = {
          mimeType:
            "audio/webm;codecs=opus",
        };

      } else if (
        window.MediaRecorder &&
        MediaRecorder.isTypeSupported(
          "audio/webm"
        )
      ) {
        recorderOptions = {
          mimeType:
            "audio/webm",
        };
      }

      const recorder =
        new MediaRecorder(
          mediaStreamRef.current,
          recorderOptions
        );

      mediaRecorderRef.current =
        recorder;

      const recordedParts = [];

      recorder.ondataavailable = (
        event
      ) => {
        if (
          event.data &&
          event.data.size > 0
        ) {
          recordedParts.push(
            event.data
          );
        }
      };

      recorder.onerror = () => {
        setLiveError(
          "Microphone recording failed."
        );
      };

      recorder.onstop = () => {
        if (
          recordedParts.length > 0 &&
          liveMonitoringRef.current
        ) {
          const blob = new Blob(
            recordedParts,
            {
              type:
                recorder.mimeType ||
                "audio/webm",
            }
          );

          if (blob.size > 0) {
            liveQueueRef.current.push(
              blob
            );

            processLiveQueue();
          }
        }

        if (
          liveMonitoringRef.current
        ) {
          liveTimerRef.current =
            setTimeout(
              recordNextLiveChunk,
              100
            );
        }
      };

      recorder.start();

      liveTimerRef.current =
        setTimeout(() => {
          if (
            recorder.state ===
            "recording"
          ) {
            recorder.stop();
          }
        }, 5000);

    } catch (err) {
      console.error(err);

      setLiveError(
        "Unable to start the next microphone chunk."
      );

      liveMonitoringRef.current =
        false;

      setLiveMonitoring(false);
    }
  };


  const startLiveMonitoring = async () => {
    if (demoMode) {
      setLiveError("Live microphone analysis requires the SonicT backend. Use Dashboard, Evidence Viewer, Reports, and Analytics to explore the offline sample.");
      return;
    }
    if (liveMonitoringRef.current) {
      return;
    }

    setLiveError("");
    setLiveResult(null);
    setLiveHistory([]);
    setLiveChunkCount(0);
    setLiveContinuousThreat(false);
    setLiveSuspiciousSegments([]);
    setContactBlocked(false);

    if (
      typeof Notification !== "undefined" &&
      Notification.permission === "default"
    ) {
      Notification.requestPermission().catch(() => {});
    }

    liveProcessedHistoryRef.current = [];
    liveQueueRef.current = [];

    if (
      !navigator.mediaDevices ||
      !navigator.mediaDevices.getUserMedia
    ) {
      setLiveError(
        "This browser does not support microphone capture."
      );

      return;
    }

    if (
      typeof MediaRecorder ===
      "undefined"
    ) {
      setLiveError(
        "This browser does not support MediaRecorder."
      );

      return;
    }

    try {
      const stream =
        await navigator.mediaDevices.getUserMedia(
          {
            audio: {
              echoCancellation: true,
              noiseSuppression: true,
              autoGainControl: true,
            },
          }
        );

      mediaStreamRef.current =
        stream;

      liveMonitoringRef.current =
        true;

      setLiveMonitoring(true);

      recordNextLiveChunk();

    } catch (err) {
      console.error(err);

      setLiveError(
        "Microphone permission was denied or the microphone is unavailable."
      );
    }
  };


  const stopLiveMonitoring = () => {
    liveMonitoringRef.current =
      false;

    setLiveMonitoring(false);

    if (liveTimerRef.current) {
      clearTimeout(
        liveTimerRef.current
      );

      liveTimerRef.current =
        null;
    }

    if (
      mediaRecorderRef.current &&
      mediaRecorderRef.current.state !==
        "inactive"
    ) {
      try {
        mediaRecorderRef.current.stop();
      } catch {
        // Ignore recorder shutdown errors.
      }
    }

    if (mediaStreamRef.current) {
      mediaStreamRef.current
        .getTracks()
        .forEach(
          (track) => track.stop()
        );

      mediaStreamRef.current =
        null;
    }

    liveQueueRef.current = [];
  };


  /* ======================================================
     FILE HANDLING
  ====================================================== */

  const handleFile = (selectedFile) => {
    if (!selectedFile) {
      return;
    }

    const fileName = selectedFile.name.toLowerCase();

    const extension =
      "." +
      fileName
        .split(".")
        .pop();

    if (!allowedExtensions.includes(extension)) {
      setError(
        "Unsupported format. Supported: WAV, MP3, MP4, MPEG, MPG, M4A, FLAC, AAC, OGG, WMA, OPUS, WEBM, MOV, MKV, AVI, 3GP, 3G2, TS, M2TS, MKA, AIFF, AIF, CAF and AMR."
      );

      setFile(null);
      setResult(null);

      setChunkResult(null);
      setChunkError("");

      setMimicryResult(null);
      setMimicryError("");

      return;
    }

    setFile(selectedFile);

    setResult(null);
    setChunkResult(null);
    setMimicryResult(null);

    setError("");
    setChunkError("");
    setMimicryError("");
  };


  const handleDrop = (event) => {
    event.preventDefault();

    setDragActive(false);

    if (
      event.dataTransfer.files &&
      event.dataTransfer.files.length > 0
    ) {
      handleFile(
        event.dataTransfer.files[0]
      );
    }
  };


  const resetAnalysis = () => {
    setFile(null);

    setResult(null);
    setChunkResult(null);
    setMimicryResult(null);

    setError("");
    setChunkError("");
    setMimicryError("");

    setLoading(false);
    setChunkLoading(false);
    setMimicryLoading(false);
  };


  /* ======================================================
     NORMAL FULL AUDIO ANALYSIS
  ====================================================== */

  const analyzeAudio = async () => {
    if (demoMode) {
      setLoading(true);
      setError("");
      window.setTimeout(() => {
        setResult(DEMO_SAMPLES[demoSampleKey]?.result || DEMO_RESULT);
        setLoading(false);
      }, 500);
      return;
    }

    if (!file) {
      setError(
        "Please select an audio file."
      );

      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();

    formData.append(
      "file",
      file
    );

    try {
      const response = await fetch(
        `${API_BASE}/analyze`,
        {
          method: "POST",
          headers: API_AUTH_HEADERS,
          body: formData,
        }
      );

      const data =
        await response.json().catch(
          () => null
        );

      if (!response.ok) {
        throw new Error(
          data?.detail ||
            "Analysis failed."
        );
      }

      setResult(data);

    } catch (err) {
      console.error(err);

      setError(
        err.message ||
          "Unable to analyze audio."
      );

    } finally {
      setLoading(false);
    }
  };


  /* ======================================================
     CHUNK-BASED ANALYSIS
  ====================================================== */

  const analyzeChunks = async () => {
    if (demoMode) {
      setChunkError("Offline demo uses a pre-analysed complete result. Connect the API for live 5-second chunk analysis.");
      return;
    }

    if (!file) {
      setChunkError(
        "Please select an audio file first."
      );

      return;
    }

    setChunkLoading(true);

    setChunkError("");
    setChunkResult(null);

    const formData = new FormData();

    formData.append(
      "file",
      file
    );

    try {
      const response = await fetch(
        `${API_BASE}/analyze-chunks`,
        {
          method: "POST",
          headers: API_AUTH_HEADERS,
          body: formData,
        }
      );

      const data =
        await response.json().catch(
          () => null
        );

      if (!response.ok) {
        throw new Error(
          data?.detail ||
            "Chunk analysis failed."
        );
      }

      setChunkResult(data);

    } catch (err) {
      console.error(err);

      setChunkError(
        err.message ||
          "Unable to run chunk analysis."
      );

    } finally {
      setChunkLoading(false);
    }
  };


  /* ======================================================
     HUMAN MIMICRY / HUMAN IMPERSONATION ANALYSIS
  ====================================================== */

  const analyzeMimicry = async () => {
    if (demoMode) {
      setMimicryError("Offline demo uses stored sample results. Connect the API to analyse a new voice recording.");
      return;
    }

    if (!file) {
      setMimicryError(
        "Please select an audio file first."
      );
      return;
    }

    setMimicryLoading(true);
    setMimicryError("");
    setMimicryResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(
        `${API_BASE}/analyze-mimicry`,
        {
          method: "POST",
          headers: API_AUTH_HEADERS,
          body: formData,
        }
      );

      const data =
        await response.json().catch(
          () => null
        );

      if (!response.ok) {
        throw new Error(
          data?.detail ||
            "Human mimicry analysis failed."
        );
      }

      setMimicryResult(data);

    } catch (err) {
      console.error(err);

      setMimicryError(
        err.message ||
          "Unable to run human mimicry analysis."
      );

    } finally {
      setMimicryLoading(false);
    }
  };


  /* ======================================================
     REPORTS
  ====================================================== */

  const loadReports = async () => {
    if (demoMode) {
      setReports(createDemoReport(DEMO_SAMPLES[demoSampleKey]?.result || DEMO_RESULT));
      return;
    }
    setReportsLoading(true);
    setReportsError("");

    try {
      const response = await fetch(
        `${API_BASE}/reports`,
        {
          headers: API_AUTH_HEADERS,
        }
      );

      if (!response.ok) {
        throw new Error(
          "Unable to load reports."
        );
      }

      const data =
        await response.json();

      if (Array.isArray(data)) {
        setReports(data);

      } else if (
        Array.isArray(data.reports)
      ) {
        setReports(
          data.reports
        );

      } else {
        setReports([]);
      }

    } catch (err) {
      console.error(err);

      setReportsError(
        "Unable to load saved reports."
      );

    } finally {
      setReportsLoading(false);
    }
  };


  /* ======================================================
     SYSTEM STATUS
  ====================================================== */

  const checkSystemStatus = async () => {
    if (demoMode) {
      setSystemStatus({ api: false, loading: false, message: "Offline demo mode — backend health check is intentionally disabled." });
      return;
    }
    setSystemStatus({
      api: false,
      loading: true,
      message:
        "Checking SonicT backend...",
    });

    try {
      const response = await fetch(`${API_BASE}/health`, {
  headers: API_AUTH_HEADERS,
});

      if (!response.ok) {
        throw new Error(
          "Health check failed."
        );
      }

      const data =
        await response.json();

      setSystemStatus({
        api: true,
        loading: false,
        message:
          data?.status ||
          "Online",
        data,
      });

    } catch (err) {
      console.error(err);

      setSystemStatus({
        api: false,
        loading: false,
        message:
          "SonicT backend is offline.",
      });
    }
  };




  /* ======================================================
     DOWNLOAD FORENSIC REPORT PDF
  ====================================================== */

  const downloadForensicReport = async () => {
    if (demoMode) {
      setReportDownloadError("PDF download requires the live API. This screen shows a pre-analysed demo report.");
      return;
    }
    if (!result?.forensic_report) {
      setReportDownloadError(
        "No forensic report is available. Run a new audio analysis first."
      );
      return;
    }

    setReportDownloading(true);
    setReportDownloadError("");

    try {
      const response = await fetch(
        `${API_BASE}/download-forensic-report`,
        {
          method: "POST",
          headers: {
            ...API_AUTH_HEADERS,
            "Content-Type": "application/json",
          },
          body: JSON.stringify(
            result.forensic_report
          ),
        }
      );

      if (!response.ok) {
        const data =
          await response.json().catch(
            () => null
          );

        throw new Error(
          data?.detail ||
            "Unable to generate the forensic PDF report."
        );
      }

      const blob =
        await response.blob();

      const url =
        URL.createObjectURL(blob);

      const originalName =
        result.forensic_report
          ?.report_information
          ?.filename ||
        file?.name ||
        "sonict_evidence";

      const baseName =
        originalName.replace(
          /\.[^/.]+$/,
          ""
        );

      const anchor =
        document.createElement(
          "a"
        );

      anchor.href = url;

      anchor.download =
        `${baseName}_sonict_forensic_report.pdf`;

      document.body.appendChild(
        anchor
      );

      anchor.click();

      anchor.remove();

      URL.revokeObjectURL(
        url
      );

    } catch (err) {
      console.error(err);

      setReportDownloadError(
        err.message ||
          "Unable to download the forensic PDF report."
      );

    } finally {
      setReportDownloading(false);
    }
  };


  /* ======================================================
     PAGE NAVIGATION
  ====================================================== */

  const changePage = (page) => {
    if (
      page !== "live" &&
      liveMonitoringRef.current
    ) {
      stopLiveMonitoring();
    }

    setActivePage(page);

    if (!demoMode && (page === "reports" || page === "analytics")) {
      loadReports();
    }

    if (page === "status") {
      if (demoMode) {
        setSystemStatus({
          api: false,
          loading: false,
          message: "Offline demo mode — backend health check is intentionally disabled.",
        });
      } else {
        checkSystemStatus();
      }
    }
  };


  /* ======================================================
     HELPERS
  ====================================================== */

  const formatSize = (bytes) => {
    if (!bytes) {
      return "0 MB";
    }

    const mb =
      bytes / (1024 * 1024);

    if (mb < 1) {
      return `${(
        bytes / 1024
      ).toFixed(1)} KB`;
    }

    return `${mb.toFixed(2)} MB`;
  };


  const percent = (value = 0) => {
    return `${(
      Number(value || 0) * 100
    ).toFixed(2)}%`;
  };

  const getRiskLevel = (value = 0) => {
    const numeric = Number(value || 0);
    const score = numeric <= 1 ? numeric * 100 : numeric;
    if (score >= 80) return { label: "CRITICAL", className: "risk-critical" };
    if (score >= 60) return { label: "HIGH", className: "risk-high" };
    if (score >= 30) return { label: "MEDIUM", className: "risk-medium" };
    return { label: "LOW", className: "risk-low" };
  };


  const formatDate = (value) => {
    if (!value) {
      return "-";
    }

    try {
      return new Date(
        value
      ).toLocaleString();

    } catch {
      return value;
    }
  };


  /* ======================================================
     RISK
  ====================================================== */

  const riskInfo = useMemo(() => {
    if (!result) {
      return null;
    }

    const backendRisk =
      result.voice_integrity_risk;

    if (backendRisk) {
      const level =
        String(
          backendRisk.risk_level || ""
        ).toUpperCase();

      const className =
        level === "LOW"
          ? "risk-low"
          : level === "MEDIUM"
          ? "risk-medium"
          : "risk-high";

      return {
        label:
          `${level || "UNKNOWN"} RISK`,

        className,

        description:
          backendRisk.recommendation ||
          "Voice integrity risk calculated from the SonicT fusion result.",
      };
    }

    const classification =
      result.classification
        ?.toLowerCase() || "";

    const confidence =
      Number(
        result.confidence || 0
      );

    if (
      classification === "genuine"
    ) {
      return {
        label: "LOW RISK",
        className: "risk-low",
        description:
          "The analyzed recording shows stronger evidence of genuine speech.",
      };
    }

    if (confidence >= 0.8) {
      return {
        label: "HIGH RISK",
        className: "risk-high",
        description:
          "Strong forensic evidence indicates suspicious audio.",
      };
    }

    return {
      label: "MEDIUM RISK",
      className: "risk-medium",
      description:
        "Suspicious forensic evidence was detected and should be reviewed.",
    };

  }, [result]);


  /* ======================================================
     FEATURE CARDS
  ====================================================== */

  const featureCards =
    result
      ? [
          {
            code: "F1",
            title: "Voice Clone Detection",
            model: "WavLM + Attention",
            value:
              result.features
                ?.voice_clone_probability ||
              0,
            icon: "AI",
          },

          {
            code: "F2",
            title: "Spectrogram Analysis",
            model: "Log-Mel + CNN",
            value:
              result.features
                ?.spectrogram_probability ||
              0,
            icon: "SP",
          },

          {
            code: "F3",
            title: "Voice Feature Analysis",
            model: "MFCC + Random Forest",
            value:
              result.features
                ?.voice_feature_probability ||
              0,
            icon: "VF",
          },

          {
            code: "F4",
            title: "Tampering Detection",
            model:
              "Boundary CNN + Temporal",
            value:
              result.tampering
                ?.f4_max ||
              0,
            icon: "TM",
          },

          {
            code: "F5",
            title: "Replay Detection",
            model: "Replay CNN",
            value:
              result.features
                ?.replay_probability ||
              0,
            icon: "RP",
          },
        ]
      : [];

  const liveFeatureCards = liveResult
    ? [
        {
          code: "F1",
          title: "Voice Clone",
          model: "WavLM + Attention",
          value:
            liveResult.feature_analysis?.f1_voice_clone ??
            liveResult.features?.voice_clone_probability ??
            0,
          icon: "AI",
        },
        {
          code: "F2",
          title: "Spectrogram Artifacts",
          model: "Log-Mel + CNN",
          value:
            liveResult.feature_analysis?.f2_spectrogram_artifacts ??
            liveResult.features?.spectrogram_probability ??
            0,
          icon: "SP",
        },
        {
          code: "F3",
          title: "Voice-Feature Anomaly",
          model: "MFCC + Random Forest",
          value:
            liveResult.feature_analysis?.f3_voice_features ??
            liveResult.features?.voice_feature_probability ??
            0,
          icon: "VF",
        },
        {
          code: "F4",
          title: "Tampering / Splicing",
          model: "Boundary CNN + Temporal",
          value:
            liveResult.feature_analysis?.f4_tampering ??
            liveResult.tampering?.f4_max ??
            0,
          icon: "TM",
        },
        {
          code: "F5",
          title: "Replay Attack",
          model: "Replay CNN",
          value:
            liveResult.feature_analysis?.f5_replay_attack ??
            liveResult.features?.replay_probability ??
            liveResult.evidence_warning?.evidence?.replay_probability ??
            0,
          icon: "RP",
        },
      ]
    : [];


  /* ======================================================
     ANALYSIS PAGE
  ====================================================== */

  const renderAnalysisPage = () => (
    <>
      <header className="topbar">
        <div>
          <p className="eyebrow">
            DIGITAL AUDIO FORENSICS PLATFORM
          </p>

          <h1>
            Audio Intelligence Analysis
          </h1>

        </div>

        <div className="top-badge">
          <span className="live-dot"></span>
          SONICT ENGINE
        </div>
      </header>


      {/* ==================================================
          UPLOAD EVIDENCE
      ================================================== */}

      <section className="panel upload-panel">

        <div className="panel-heading">
          <div>
            <span className="section-number">
              01
            </span>

            <div>
              <h2>
                Upload Evidence
              </h2>

            </div>
          </div>

          {file && (
            <button
              className="clear-button"
              onClick={resetAnalysis}
            >
              Clear
            </button>
          )}
        </div>


        <div
          className={`drop-zone ${
            dragActive
              ? "drag-active"
              : ""
          } ${
            file
              ? "has-file"
              : ""
          }`}

          onDragEnter={(e) => {
            e.preventDefault();
            setDragActive(true);
          }}

          onDragOver={(e) => {
            e.preventDefault();
            setDragActive(true);
          }}

          onDragLeave={() =>
            setDragActive(false)
          }

          onDrop={handleDrop}
        >

          {!file ? (
            <>
              <div className="upload-icon">
                <span>↑</span>
              </div>

              <h3>
                Drop audio evidence here
              </h3>

              <p>
                MP4, MPEG and other common audio/video containers are supported.
                SonicT extracts the audio
                automatically and converts
                it into 16 kHz mono WAV.
              </p>


              <label className="browse-button">

                Browse Audio

                <input
                  type="file"

                  accept="
                    .wav,
                    .mp3,
                    .mp4,
                    .mpeg,
                    .mpg,
                    .m4a,
                    .flac,
                    .aac,
                    .ogg,
                    .wma,
                    .opus,
                    .webm,
                    .mov,
                    .mkv,
                    .avi,
                    .3gp,
                    .3g2,
                    .ts,
                    .m2ts,
                    .mka,
                    .aiff,
                    .aif,
                    .caf,
                    .amr,
                    audio/*,
                    video/mp4,
                    video/mpeg,
                    video/quicktime,
                    video/x-matroska
                  "

                  hidden

                  onChange={(e) =>
                    handleFile(
                      e.target.files[0]
                    )
                  }
                />

              </label>


              <span className="formats">
                WAV • MP3 • MP4 • MPEG • MPG • M4A •
                FLAC • AAC • OGG • WMA • OPUS • WEBM •
                MOV • MKV • AVI • 3GP • AIFF • CAF • AMR
              </span>
            </>

          ) : (

            <div className="selected-file">

              <div className="audio-file-icon">
                ♪
              </div>


              <div className="file-information">

                <span className="file-label">
                  AUDIO EVIDENCE
                </span>

                <h3>
                  {file.name}
                </h3>


                <div className="file-meta">

                  <span>
                    {formatSize(
                      file.size
                    )}
                  </span>


                  <span>
                    {file.name
                      .split(".")
                      .pop()
                      .toUpperCase()}
                  </span>


                  <span>
                    Convert → WAV
                  </span>


                  <span>
                    Ready for analysis
                  </span>

                </div>

              </div>


              <div className="file-ready">
                <span>✓</span>
              </div>

            </div>
          )}

        </div>


        {audioUrl && (

          <div className="audio-preview">

            <div className="audio-preview-header">

              <span>
                Original Evidence Preview
              </span>

              <span className="secure-text">
                Backend conversion enabled
              </span>

            </div>


            <audio
              controls
              src={audioUrl}
            />

          </div>
        )}


        <div className="analysis-mode-section">
          <div className="analysis-mode-heading">
            <span className="section-number">02</span>

            <div>
              <h2>Choose Analysis Mode</h2>
            </div>
          </div>

          <div className="analysis-mode-grid">
            <button
              type="button"
              className="analysis-mode-card"
              onClick={analyzeAudio}
              disabled={!file || loading || chunkLoading || mimicryLoading}
            >
              <div className="analysis-mode-icon">◎</div>

              <div>
                <span className="analysis-mode-code">FULL ANALYSIS</span>
                <h3>Forensic Analysis</h3>
              </div>

              <span className="analysis-mode-action" aria-live="polite">
                {loading && <span className="analysis-mode-spinner" aria-hidden="true" />}
                <strong>{loading ? "Analyzing audio" : "Run full analysis"}</strong>
                {!loading && <span className="analysis-mode-arrow" aria-hidden="true">→</span>}
              </span>
            </button>

            <button
              type="button"
              className="analysis-mode-card"
              onClick={analyzeChunks}
              disabled={!file || loading || chunkLoading || mimicryLoading}
            >
              <div className="analysis-mode-icon">◫</div>

              <div>
                <span className="analysis-mode-code">TEMPORAL ANALYSIS</span>
                <h3>Chunk Risk Analysis</h3>
              </div>

              <span className="analysis-mode-action" aria-live="polite">
                {chunkLoading && <span className="analysis-mode-spinner" aria-hidden="true" />}
                <strong>{chunkLoading ? "Analyzing timeline" : "Run chunk analysis"}</strong>
                {!chunkLoading && <span className="analysis-mode-arrow" aria-hidden="true">→</span>}
              </span>
            </button>

            <button
              type="button"
              className="analysis-mode-card"
              onClick={analyzeMimicry}
              disabled={!file || loading || chunkLoading || mimicryLoading}
            >
              <div className="analysis-mode-icon">◉</div>

              <div>
                <span className="analysis-mode-code">IMPERSONATION</span>
                <h3>Human Mimicry Analysis</h3>
              </div>

              <span className="analysis-mode-action" aria-live="polite">
                {mimicryLoading && <span className="analysis-mode-spinner" aria-hidden="true" />}
                <strong>{mimicryLoading ? "Checking mimicry" : "Run mimicry analysis"}</strong>
                {!mimicryLoading && <span className="analysis-mode-arrow" aria-hidden="true">→</span>}
              </span>
            </button>
          </div>

          {error && (
            <div className="error-message">
              <span>!</span>
              {error}
            </div>
          )}

          {chunkError && (
            <div className="error-message">
              <span>!</span>
              {chunkError}
            </div>
          )}

          {mimicryError && (
            <div className="error-message">
              <span>!</span>
              {mimicryError}
            </div>
          )}
        </div>

      </section>


      {/* INPUT PROCESSING */}

      {file && (

        <section className="panel">

          <div className="panel-heading">

            <div>

              <span className="section-number">
                IN
              </span>


              <div>

                <h2>
                  Input Processing
                </h2>


              </div>

            </div>

          </div>


          <div className="tamper-stat-grid">

            <div className="stat-card">

              <span>
                Original Format
              </span>

              <strong>
                {file.name
                  .split(".")
                  .pop()
                  .toUpperCase()}
              </strong>


            </div>


            <div className="stat-card">

              <span>
                Analysis Format
              </span>

              <strong>
                WAV
              </strong>


            </div>


            <div className="stat-card">

              <span>
                Sample Rate
              </span>

              <strong>
                16000 Hz
              </strong>


            </div>

          </div>

        </section>

      )}


      {loading && (

        <section className="analysis-progress panel">

          <div className="scanner">
            <div className="scanner-line"></div>
          </div>


          <div>
            <h3>
              Processing audio evidence
            </h3>

            <p>
              Extracting audio,
              converting to 16 kHz mono
              WAV and running F1–F5
              forensic models.
            </p>
          </div>

        </section>
      )}


      {chunkLoading && (

        <section className="analysis-progress panel">

          <div className="scanner">
            <div className="scanner-line"></div>
          </div>


          <div>

            <h3>
              Processing audio chunks
            </h3>

            <p>
              Dividing the evidence into
              5-second segments and calculating
              voice integrity risk for each chunk.
            </p>

          </div>

        </section>

      )}


      {result &&
        renderCurrentResults()}


      {mimicryLoading && (

        <section className="analysis-progress panel">

          <div className="scanner">
            <div className="scanner-line"></div>
          </div>

          <div>
            <h3>
              Analyzing human mimicry
            </h3>

            <p>
              Dividing the full recording into
              3-second segments and applying the
              Human Mimicry V2 Random Forest model.
            </p>
          </div>

        </section>

      )}


      {chunkResult &&
        renderChunkResults()}


      {mimicryResult &&
        renderMimicryResults()}

    </>
  );


  /* ======================================================
     NORMAL RESULTS
  ====================================================== */

  const renderCurrentResults = () => {

    const incident = result?.incident || null;

    const operationalStatus = String(
      result?.operational_assessment?.status ||
        (result?.security_alert?.alert ? "ACTION_REQUIRED" : "NORMAL")
    ).toUpperCase();

    const verificationRequired = Boolean(
      incident?.verification_required ||
        result?.operational_assessment?.status === "VERIFICATION_REQUIRED" ||
        result?.security_alert?.alert
    );

    const openCurrentIncident = () => {
      if (incident) {
        setSelectedIncident(incident);
      }
      changePage("alerts");
    };

    return (
    <div className="results">

      <section className="panel security-decision-summary">
        <div className="panel-heading">
          <div>
            <span className="section-number">SEC</span>
            <div>
              <h2>Security Decision Summary</h2>
            </div>
          </div>

          <span className={`risk-badge ${riskInfo?.className}`}>
            {String(
              result?.voice_integrity_risk?.risk_level || "UNKNOWN"
            ).toUpperCase()} RISK
          </span>
        </div>

        <div className="tamper-stat-grid">
          <div className="stat-card">
            <span>Final Classification</span>
            <strong>
              {String(result?.classification || "UNKNOWN").toUpperCase()}
            </strong>
          </div>

          <div className="stat-card">
            <span>Confidence</span>
            <strong>{percent(result?.confidence)}</strong>
          </div>

          <div className="stat-card">
            <span>Voice Integrity Risk</span>
            <strong>
              {Number(
                result?.voice_integrity_risk?.risk_score || 0
              ).toFixed(2)}%
            </strong>
            <em className={`prominent-risk-level ${riskInfo?.className}`}>
              {String(result?.voice_integrity_risk?.risk_level || "UNKNOWN").toUpperCase()} RISK
            </em>
          </div>

          <div className="stat-card">
            <span>Operational Status</span>
            <strong>{operationalStatus.replaceAll("_", " ")}</strong>
          </div>

          <div className="stat-card">
            <span>Identity Verification</span>
            <strong>{verificationRequired ? "REQUIRED" : "NORMAL"}</strong>
          </div>

          <div className="stat-card">
            <span>Incident</span>
            <strong>{incident?.incident_uuid ? "CREATED" : "NOT CREATED"}</strong>
          </div>
        </div>

        {incident?.incident_uuid && (
          <div className="analysis-incident-bridge">
            <div className="analysis-incident-status-grid">
              <div className="analysis-incident-status-card">
                <span>SECURITY INCIDENT</span>
                <strong>
                  {String(
                    incident.incident_status || "OPEN"
                  ).replaceAll("_", " ")}
                </strong>
              </div>

              <div className="analysis-incident-status-card">
                <span>TRUSTED VERIFICATION</span>
                <strong>
                  {String(
                    incident.verification_status || "NOT_STARTED"
                  ).replaceAll("_", " ")}
                </strong>
              </div>
            </div>

            <button
              type="button"
              className="analyze-button"
              onClick={openCurrentIncident}
            >
              <span>△</span>
              View Incident & Trusted Verification
            </button>
          </div>
        )}
      </section>

      <section className="result-layout">

        <div
          className={`verdict-card classification-${result.classification?.toLowerCase()}`}
        >

          <div className="verdict-top">

            <span className="section-number">
              02
            </span>


            <span
              className={`risk-badge ${riskInfo?.className}`}
            >
              {riskInfo?.label}
            </span>

          </div>


          <p className="verdict-label">
            FINAL FORENSIC CLASSIFICATION
          </p>


          <h2>
            {result.classification
              ?.toUpperCase()}
          </h2>


          <div className="confidence-row">

            <div>
              <span>
                Confidence
              </span>

              <strong>
                {percent(
                  result.confidence
                )}
              </strong>
            </div>


            <div className="confidence-ring">

              <span>
                {(
                  Number(
                    result.confidence ||
                      0
                  ) * 100
                ).toFixed(0)}
              </span>

              <small>%</small>

            </div>

          </div>


          <p className="risk-description">
            {riskInfo?.description}
          </p>

        </div>


        <div className="panel probability-panel">

          <p className="mini-label">
            MODEL DISTRIBUTION
          </p>

          <h2>
            Class Probabilities
          </h2>


          <div className="probability-list">

            {Object.entries(
              result.class_probabilities ||
                {}
            )
              .sort(
                (a, b) =>
                  b[1] - a[1]
              )

              .map(
                ([
                  label,
                  probability,
                ]) => (

                  <div
                    className="probability-item"
                    key={label}
                  >

                    <div className="probability-header">

                      <span>
                        {label}
                      </span>

                      <strong>
                        {percent(
                          probability
                        )}
                      </strong>

                    </div>


                    <div className="progress-track">

                      <div
                        className={`progress-fill probability-${label.toLowerCase()}`}

                        style={{
                          width: `${Math.min(
                            Number(
                              probability ||
                                0
                            ) * 100,
                            100
                          )}%`,
                        }}
                      />

                    </div>

                  </div>
                )
              )}

          </div>

        </div>

      </section>


      {result.voice_integrity_risk && (

        <section className="panel">

          <div className="panel-heading">

            <div>

              <div>

                <h2>
                  Voice Integrity Risk
                </h2>

                <p>
                  Dynamic security risk
                  calculated from the final
                  SonicT fusion probabilities.
                </p>

              </div>

            </div>


            <span
              className={`risk-badge ${
                riskInfo?.className
              }`}
            >

              {
                result
                  .voice_integrity_risk
                  .risk_level
              }{" "}
              RISK

            </span>

          </div>


          <div className="tamper-stat-grid">

            <div className="stat-card">

              <span>
                Risk Score
              </span>

              <strong>

                {Number(
                  result
                    .voice_integrity_risk
                    .risk_score || 0
                ).toFixed(2)}

                %

              </strong>

              <small>
                100 − Genuine probability
              </small>

            </div>


            <div className="stat-card">

              <span>
                Risk Level
              </span>

              <strong>
                {
                  result
                    .voice_integrity_risk
                    .risk_level
                }
              </strong>

              <small>
                LOW / MEDIUM / HIGH / CRITICAL
              </small>

            </div>


            <div className="stat-card">

              <span>
                Secondary Verification
              </span>

              <strong>

                {result.security_alert
                  ?.alert
                  ? "REQUIRED"
                  : "NORMAL"}

              </strong>

              <small>

                {
                  result
                    .voice_integrity_risk
                    .recommendation
                }

              </small>

            </div>

          </div>


          {result.security_alert && (

            <div
              className={
                result.security_alert.alert
                  ? "error-message"
                  : "safe-window-message"
              }
            >

              <span>
                {result.security_alert.alert
                  ? "!"
                  : "✓"}
              </span>


              <div>

                <h3>
                  {
                    result
                      .security_alert
                      .message
                  }
                </h3>


                <p>

                  <strong>
                    Recommended Action:
                  </strong>{" "}

                  {
                    result
                      .security_alert
                      .action
                  }

                </p>

              </div>

            </div>

          )}

        </section>

      )}


      <section className="panel">

        <div className="panel-heading">

          <div>

            <span className="section-number">
              03
            </span>


            <div>

              <h2>
                Forensic Evidence
              </h2>

              <p>
                Outputs from SonicT's five
                forensic models.
              </p>

            </div>

          </div>

        </div>


        <div className="feature-grid">

          {featureCards.map(
            (feature) => (

              <div
                className="feature-card"
                key={feature.code}
              >

                <div className="feature-header">

                  <div className="feature-icon">
                    {feature.icon}
                  </div>

                  <span>
                    {feature.code}
                  </span>

                </div>


                <h3>
                  {feature.title}
                </h3>


                <p className="model-name">
                  {feature.model}
                </p>


                <div className="feature-score">
                  {percent(
                    feature.value
                  )}
                </div>


                <div className="feature-bar">

                  <div
                    style={{
                      width: `${Math.min(
                        Number(
                          feature.value ||
                            0
                        ) * 100,
                        100
                      )}%`,
                    }}
                  />

                </div>


                <span className="evidence-label">
                  Evidence probability
                </span>

              </div>

            )
          )}

        </div>

      </section>


      <section className="panel">

        <div className="panel-heading">

          <div>

            <span className="section-number">
              04
            </span>


            <div>

              <h2>
                Tampering Intelligence
              </h2>

              <p>
                Temporal evidence from
                the boundary model.
              </p>

            </div>

          </div>

        </div>


        <div className="tamper-stat-grid">

          <div className="stat-card">

            <span>
              Maximum Evidence
            </span>

            <strong>
              {percent(
                result.tampering
                  ?.f4_max
              )}
            </strong>

          </div>


          <div className="stat-card">

            <span>
              Suspicious Ratio
            </span>

            <strong>
              {percent(
                result.tampering
                  ?.f4_suspicious_ratio
              )}
            </strong>

          </div>


          <div className="stat-card">

            <span>
              High Risk Ratio
            </span>

            <strong>
              {percent(
                result.tampering
                  ?.f4_high_ratio
              )}
            </strong>

          </div>

        </div>

      </section>


      <section className="panel">

        <div className="panel-heading">

          <div>

            <span className="section-number">
              05
            </span>


            <div>

              <h2>
                Suspicious Audio Regions
              </h2>

              <p>
                Localized windows requiring
                forensic review.
              </p>

            </div>

          </div>


          <div className="window-count">

            {result.tampering
              ?.suspicious_windows
              ?.length || 0}{" "}
            regions

          </div>

        </div>


        {!result.tampering
          ?.suspicious_windows ||
        result.tampering
          .suspicious_windows
          .length === 0 ? (

          <div className="safe-window-message">

            <div className="safe-check">
              ✓
            </div>


            <div>

              <h3>
                No suspicious regions detected
              </h3>

              <p>
                No temporal windows crossed
                the configured suspicious
                threshold.
              </p>

            </div>

          </div>

        ) : (

          <div className="window-list">

            {result.tampering
              .suspicious_windows
              .map(
                (
                  window,
                  index
                ) => {
                  const regionRisk = getRiskLevel(window.probability);
                  return (

                  <div
                    className="window-card"
                    key={index}
                  >

                    <div className="window-index">

                      {String(
                        index + 1
                      ).padStart(
                        2,
                        "0"
                      )}

                    </div>


                    <div className="window-time">

                      <span>
                        TIME RANGE
                      </span>

                      <strong>

                        {Number(
                          window.start ||
                            0
                        ).toFixed(2)}

                        s →{" "}

                        {Number(
                          window.end ||
                            0
                        ).toFixed(2)}

                        s

                      </strong>

                    </div>


                    <div className="window-score">

                      <span>
                        PROBABILITY
                      </span>

                      <strong>
                        {percent(
                          window.probability
                        )}
                      </strong>

                    </div>

                    <div className="window-score">
                      <span>RISK LEVEL</span>
                      <strong className={`risk-badge ${regionRisk.className}`}>
                        {regionRisk.label}
                      </strong>
                    </div>

                  </div>

                  );
                }
              )}

          </div>
        )}

      </section>

    </div>
    );
  };


  /* ======================================================
     HUMAN MIMICRY RESULTS
  ====================================================== */

  const renderMimicryResults = () => {
    const segmentAnalysis =
      mimicryResult?.segment_analysis || {};

    const finalAssessment = String(
      mimicryResult?.classification || "UNKNOWN"
    ).toUpperCase();

    const mimicProbability = Number(
      mimicryResult?.probabilities?.human_mimic || 0
    );

    const realProbability = Number(
      mimicryResult?.probabilities?.real_human || 0
    );

    const assessmentClass =
      finalAssessment === "REAL_HUMAN"
        ? "risk-low"
        : finalAssessment === "UNCERTAIN"
        ? "risk-medium"
        : "risk-high";

    const assessmentLabel =
      finalAssessment === "REAL_HUMAN"
        ? "REAL HUMAN"
        : finalAssessment === "UNCERTAIN"
        ? "UNCERTAIN"
        : "POSSIBLE HUMAN MIMIC";

    return (
      <div className="results">

        <section className="panel">

          <div className="panel-heading">
            <div>
              <span className="section-number">
                HM
              </span>

              <div>
                <h2>
                  Human Mimicry Analysis
                </h2>

                <p>
                  Separate human impersonation analysis
                  using 3-second segments and median
                  probability aggregation.
                </p>
              </div>
            </div>

            <span
              className={`risk-badge ${assessmentClass}`}
            >
              {assessmentLabel}
            </span>
          </div>


          <div className="tamper-stat-grid">

            <div className="stat-card">
              <span>Final Assessment</span>
              <strong>{assessmentLabel}</strong>
              <small>Human mimicry module decision</small>
            </div>

            <div className="stat-card">
              <span>Real Human</span>
              <strong>
                {(realProbability * 100).toFixed(2)}%
              </strong>
              <small>Final median-based probability</small>
            </div>

            <div className="stat-card">
              <span>Human Mimic</span>
              <strong>
                {(mimicProbability * 100).toFixed(2)}%
              </strong>
              <small>Final median-based probability</small>
            </div>

            <div className="stat-card">
              <span>Segments Analyzed</span>
              <strong>
                {segmentAnalysis.segments_analyzed || 0}
              </strong>
              <small>3-second usable segments</small>
            </div>

            <div className="stat-card">
              <span>Real Segments</span>
              <strong>
                {segmentAnalysis.real_segments || 0}
              </strong>
              <small>Segment mimic score below 50%</small>
            </div>

            <div className="stat-card">
              <span>Mimic Segments</span>
              <strong>
                {segmentAnalysis.mimic_segments || 0}
              </strong>
              <small>Segment mimic score at least 50%</small>
            </div>

          </div>


          <div style={{ marginTop: "28px" }}>

            <div className="panel-heading">
              <div>
                <span className="section-number">
                  ST
                </span>

                <div>
                  <h2>
                    Multi-Segment Statistics
                  </h2>

                  <p>
                    Diagnostic values calculated across
                    the complete recording.
                  </p>
                </div>
              </div>
            </div>


            <div className="tamper-stat-grid">

              <div className="stat-card">
                <span>Mean Mimic</span>
                <strong>
                  {percent(
                    segmentAnalysis.mean_mimic_probability
                  )}
                </strong>
                <small>Average of segment scores</small>
              </div>

              <div className="stat-card">
                <span>Median Mimic</span>
                <strong>
                  {percent(
                    segmentAnalysis.median_mimic_probability
                  )}
                </strong>
                <small>Used for the final decision</small>
              </div>

              <div className="stat-card">
                <span>Mimic Std Deviation</span>
                <strong>
                  {percent(
                    segmentAnalysis.mimic_std_deviation
                  )}
                </strong>
                <small>Variation between segments</small>
              </div>

              <div className="stat-card">
                <span>Segment Consistency</span>
                <strong>
                  {String(
                    segmentAnalysis.segment_consistency ||
                      "UNKNOWN"
                  ).toUpperCase()}
                </strong>
                <small>HIGH / MEDIUM / LOW</small>
              </div>

              <div className="stat-card">
                <span>Audio Duration</span>
                <strong>
                  {Number(
                    segmentAnalysis.total_duration_seconds || 0
                  ).toFixed(2)}s
                </strong>
                <small>Full recording duration</small>
              </div>

              <div className="stat-card">
                <span>Aggregation</span>
                <strong>
                  {String(
                    mimicryResult.aggregation ||
                      "MEDIAN"
                  ).toUpperCase()}
                </strong>
                <small>Reduces isolated noisy outliers</small>
              </div>

            </div>
          </div>


          <div style={{ marginTop: "28px" }}>

            <p className="mini-label">
              HUMAN MIMICRY DISTRIBUTION
            </p>

            <h2>
              Final Probabilities
            </h2>

            <div className="probability-list">

              <div className="probability-item">
                <div className="probability-header">
                  <span>Real Human</span>
                  <strong>
                    {(realProbability * 100).toFixed(2)}%
                  </strong>
                </div>

                <div className="progress-track">
                  <div
                    className="progress-fill probability-genuine"
                    style={{
                      width: `${Math.min(
                        realProbability * 100,
                        100
                      )}%`,
                    }}
                  />
                </div>
              </div>


              <div className="probability-item">
                <div className="probability-header">
                  <span>Human Mimic</span>
                  <strong>
                    {(mimicProbability * 100).toFixed(2)}%
                  </strong>
                </div>

                <div className="progress-track">
                  <div
                    className="progress-fill probability-deepfake"
                    style={{
                      width: `${Math.min(
                        mimicProbability * 100,
                        100
                      )}%`,
                    }}
                  />
                </div>
              </div>

            </div>
          </div>


          {mimicryResult.security_alert && (
            <div
              className={
                mimicryResult.security_alert.alert
                  ? "error-message"
                  : "safe-window-message"
              }
              style={{ marginTop: "22px" }}
            >
              <span>
                {mimicryResult.security_alert.alert
                  ? "!"
                  : "✓"}
              </span>

              <div>
                <h3>
                  {mimicryResult.security_alert.message}
                </h3>

                <p>
                  <strong>
                    Recommended Action:
                  </strong>{" "}
                  {mimicryResult.security_alert.action}
                </p>
              </div>
            </div>
          )}


          <div
            className="safe-window-message"
            style={{ marginTop: "18px" }}
          >
            <div className="safe-check">i</div>

            <div>
              <h3>Decision Bands</h3>

              <p>
                Below 45% mimic = REAL HUMAN •
                45% to below 65% = UNCERTAIN •
                65% or above = POSSIBLE HUMAN MIMIC.
              </p>
            </div>
          </div>


          {segmentAnalysis.segments?.length > 0 && (
            <div style={{ marginTop: "30px" }}>

              <div className="panel-heading">
                <div>
                  <span className="section-number">
                    SG
                  </span>

                  <div>
                    <h2>
                      Segment Evidence
                    </h2>

                    <p>
                      Per-segment human mimicry probabilities
                      for forensic review.
                    </p>
                  </div>
                </div>

                <div className="window-count">
                  {segmentAnalysis.segments.length} segments
                </div>
              </div>


              <div className="window-list">

                {segmentAnalysis.segments.map(
                  (segment, index) => {

                    const segmentMimic = Number(
                      segment.human_mimic || 0
                    );

                    const segmentClass =
                      segmentMimic >= 0.5
                        ? "risk-high"
                        : "risk-low";

                    return (
                      <div
                        className="window-card"
                        key={`${segment.start}-${segment.end}-${index}`}
                      >
                        <div className="window-index">
                          {String(
                            segment.segment || index + 1
                          ).padStart(2, "0")}
                        </div>

                        <div className="window-time">
                          <span>TIME RANGE</span>
                          <strong>
                            {Number(
                              segment.start || 0
                            ).toFixed(2)}s
                            {" → "}
                            {Number(
                              segment.end || 0
                            ).toFixed(2)}s
                          </strong>
                        </div>

                        <div className="window-score">
                          <span>REAL</span>
                          <strong>
                            {percent(segment.real_human)}
                          </strong>
                        </div>

                        <div className="window-score">
                          <span>MIMIC</span>
                          <strong>
                            {percent(segment.human_mimic)}
                          </strong>
                        </div>

                        <div className="window-score">
                          <span>SEGMENT RESULT</span>
                          <strong
                            className={`risk-badge ${segmentClass}`}
                          >
                            {segmentMimic >= 0.5
                              ? "MIMIC"
                              : "REAL"}
                          </strong>
                        </div>
                      </div>
                    );
                  }
                )}

              </div>
            </div>
          )}


          {mimicryResult.evidence_integrity && (
            <div style={{ marginTop: "28px" }}>

              <div className="panel-heading">
                <div>
                  <span className="section-number">
                    SHA
                  </span>

                  <div>
                    <h2>
                      Mimicry Evidence Integrity
                    </h2>

                    <p>
                      SHA-256 fingerprint generated from
                      the original uploaded evidence.
                    </p>
                  </div>
                </div>
              </div>

              <div
                className="safe-window-message"
                style={{
                  alignItems: "flex-start",
                }}
              >
                <div className="safe-check">#</div>

                <div
                  style={{
                    width: "100%",
                    minWidth: 0,
                  }}
                >
                  <h3>SHA-256</h3>

                  <p
                    style={{
                      fontFamily:
                        "Consolas, 'Courier New', monospace",
                      overflowWrap: "anywhere",
                      wordBreak: "break-all",
                      lineHeight: 1.7,
                    }}
                  >
                    {mimicryResult.evidence_integrity.sha256 ||
                      "-"}
                  </p>
                </div>
              </div>

            </div>
          )}

        </section>
      </div>
    );
  };


  /* ======================================================
     CHUNK RISK RESULTS
  ====================================================== */

  const renderChunkResults = () => (

    <div className="results">

      <section className="panel">

        <div className="panel-heading">

          <div>

            <span className="section-number">
              RT
            </span>

            <div>

              <h2>
                Chunk Risk Timeline
              </h2>

              <p>
                Near-real-time voice integrity
                analysis using 5-second audio
                segments.
              </p>

            </div>

          </div>


          <div className="window-count">
            {chunkResult.total_chunks || 0} chunks
          </div>

        </div>


        {/* CHUNK SUMMARY */}

        <div className="tamper-stat-grid">

          <div className="stat-card">
            <span>Audio Duration</span>

            <strong>
              {Number(
                chunkResult.total_duration || 0
              ).toFixed(2)}s
            </strong>

            <small>
              Total analyzed duration
            </small>
          </div>


          <div className="stat-card">
            <span>Chunk Duration</span>

            <strong>
              {Number(
                chunkResult.chunk_duration || 5
              ).toFixed(0)}s
            </strong>

            <small>
              Analysis interval
            </small>
          </div>


          <div className="stat-card">
            <span>Total Chunks</span>

            <strong>
              {chunkResult.total_chunks || 0}
            </strong>

            <small>
              Analyzed segments
            </small>
          </div>


          <div className="stat-card">
            <span>Average Risk</span>

            <strong>
              {Number(
                chunkResult.average_risk || 0
              ).toFixed(2)}%
            </strong>

            <small>
              Average chunk risk
            </small>
          </div>


          <div className="stat-card">
            <span>High Risk Chunks</span>

            <strong>
              {chunkResult.high_risk_chunks || 0}
            </strong>

            <small>
              HIGH / CRITICAL segments
            </small>
          </div>


          <div className="stat-card">

            <span>
              Continuous Threat
            </span>

            <strong>
              {chunkResult.continuous_alert
                ? "DETECTED"
                : "NOT DETECTED"}
            </strong>

            <small>
              {chunkResult.continuous_alert
                ? "2+ consecutive high-risk chunks"
                : "No persistent high-risk sequence"}
            </small>

          </div>

        </div>


        {/* HIGHEST RISK SEGMENT */}

        {chunkResult.highest_risk && (

          <div
            className={
              ["HIGH", "CRITICAL"].includes(
                String(
                  chunkResult
                    .highest_risk
                    .risk_level || ""
                ).toUpperCase()
              )
                ? "error-message"
                : "safe-window-message"
            }
          >

            <span>
              {["HIGH", "CRITICAL"].includes(
                String(
                  chunkResult
                    .highest_risk
                    .risk_level || ""
                ).toUpperCase()
              )
                ? "!"
                : "✓"}
            </span>


            <div>

              <h3>
                Highest Risk Segment
              </h3>

              <p>

                {Number(
                  chunkResult
                    .highest_risk
                    .start || 0
                ).toFixed(2)}s

                {" → "}

                {Number(
                  chunkResult
                    .highest_risk
                    .end || 0
                ).toFixed(2)}s

                {" • "}

                {String(
                  chunkResult
                    .highest_risk
                    .classification || "-"
                ).toUpperCase()}

                {" • "}

                {Number(
                  chunkResult
                    .highest_risk
                    .risk_score || 0
                ).toFixed(2)}% Risk

              </p>


              <p>

                <strong>
                  Risk Level:
                </strong>{" "}

                {String(
                  chunkResult
                    .highest_risk
                    .risk_level || "UNKNOWN"
                ).toUpperCase()}

              </p>

            </div>

          </div>

        )}


        {/* CONTINUOUS THREAT ALERT */}

        {chunkResult.continuous_alert_info && (

          <div
            className={
              chunkResult.continuous_alert
                ? "error-message"
                : "safe-window-message"
            }
          >

            <span>
              {chunkResult.continuous_alert
                ? "!"
                : "✓"}
            </span>


            <div>

              <h3>
                {chunkResult.continuous_alert
                  ? "Continuous Threat Detected"
                  : "No Continuous Threat Detected"}
              </h3>


              <p>
                {
                  chunkResult
                    .continuous_alert_info
                    .message
                }
              </p>


              <p>

                <strong>
                  Recommended Action:
                </strong>{" "}

                {
                  chunkResult
                    .continuous_alert_info
                    .action
                }

              </p>

            </div>

          </div>

        )}


        {/* CONTINUOUS SUSPICIOUS SEGMENTS */}

        {chunkResult
          .continuous_suspicious_segments
          ?.length > 0 && (

          <div
            style={{
              marginTop: "28px",
            }}
          >

            <div className="panel-heading">

              <div>

                <span className="section-number">
                  CT
                </span>


                <div>

                  <h2>
                    Continuous Suspicious Segments
                  </h2>

                  <p>
                    HIGH or CRITICAL risk detected
                    continuously across two or more
                    consecutive audio chunks.
                  </p>

                </div>

              </div>


              <div className="window-count">

                {
                  chunkResult
                    .continuous_suspicious_segments
                    .length
                }{" "}
                detected

              </div>

            </div>


            <div className="window-list">

              {chunkResult
                .continuous_suspicious_segments
                .map(
                  (
                    segment,
                    index
                  ) => (

                    <div
                      className="window-card"
                      key={index}
                    >

                      <div className="window-index">

                        {String(
                          index + 1
                        ).padStart(
                          2,
                          "0"
                        )}

                      </div>


                      <div className="window-time">

                        <span>
                          CONTINUOUS RANGE
                        </span>

                        <strong>

                          {Number(
                            segment.start || 0
                          ).toFixed(2)}s

                          {" → "}

                          {Number(
                            segment.end || 0
                          ).toFixed(2)}s

                        </strong>

                      </div>


                      <div className="window-score">

                        <span>
                          CONSECUTIVE CHUNKS
                        </span>

                        <strong>
                          {segment.chunks || 0}
                        </strong>

                      </div>


                      <div className="window-score">

                        <span>
                          MAXIMUM RISK
                        </span>

                        <strong>
                          {Number(
                            segment.max_risk || 0
                          ).toFixed(2)}%
                        </strong>

                      </div>


                      <div className="window-score">

                        <span>
                          STATUS
                        </span>

                        <strong
                          className="risk-badge risk-high"
                        >
                          THREAT
                        </strong>

                      </div>

                    </div>

                  )
                )}

            </div>

          </div>

        )}


        {/* NO PERSISTENT THREAT */}

        {!chunkResult.continuous_alert &&
          (
            chunkResult
              .continuous_suspicious_segments
              ?.length || 0
          ) === 0 && (

          <div
            className="safe-window-message"
            style={{
              marginTop: "22px",
            }}
          >

            <div className="safe-check">
              ✓
            </div>


            <div>

              <h3>
                No Persistent Threat Pattern
              </h3>

              <p>
                No two consecutive HIGH or
                CRITICAL chunks were detected.
              </p>

            </div>

          </div>

        )}


        {/* INDIVIDUAL CHUNK ANALYSIS */}

        <div
          style={{
            marginTop: "30px",
          }}
        >

          <div className="panel-heading">

            <div>

              <span className="section-number">
                CH
              </span>


              <div>

                <h2>
                  Individual Chunk Analysis
                </h2>

                <p>
                  Classification, confidence
                  and voice integrity risk
                  generated for every 5-second
                  segment.
                </p>

              </div>

            </div>


            <div className="window-count">

              {chunkResult.chunks?.length || 0}{" "}
              results

            </div>

          </div>


          {!chunkResult.chunks ||
          chunkResult.chunks.length === 0 ? (

            <div className="safe-window-message">

              <div className="safe-check">
                i
              </div>


              <div>

                <h3>
                  No chunks available
                </h3>

                <p>
                  No valid audio segments were
                  generated for chunk analysis.
                </p>

              </div>

            </div>

          ) : (

            <div className="window-list">

              {chunkResult.chunks.map(
                (
                  chunk,
                  index
                ) => {

                  const level =
                    String(
                      chunk.risk_level || ""
                    ).toUpperCase();


                  const chunkRiskClass =
                    level === "LOW"
                      ? "risk-low"
                      : level === "MEDIUM"
                      ? "risk-medium"
                      : "risk-high";


                  return (

                    <div
                      className="window-card"
                      key={index}
                    >

                      <div className="window-index">

                        {String(
                          index + 1
                        ).padStart(
                          2,
                          "0"
                        )}

                      </div>


                      <div className="window-time">

                        <span>
                          TIME RANGE
                        </span>

                        <strong>

                          {Number(
                            chunk.start || 0
                          ).toFixed(2)}s

                          {" → "}

                          {Number(
                            chunk.end || 0
                          ).toFixed(2)}s

                        </strong>

                      </div>


                      <div className="window-score">

                        <span>
                          CLASS
                        </span>

                        <strong>

                          {String(
                            chunk.classification ||
                              "-"
                          ).toUpperCase()}

                        </strong>

                      </div>


                      <div className="window-score">

                        <span>
                          CONFIDENCE
                        </span>

                        <strong>
                          {percent(
                            chunk.confidence
                          )}
                        </strong>

                      </div>


                      <div className="window-score">

                        <span>
                          RISK SCORE
                        </span>

                        <strong>
                          {Number(
                            chunk.risk_score || 0
                          ).toFixed(2)}%
                        </strong>

                      </div>


                      <div className="window-score">

                        <span>
                          RISK LEVEL
                        </span>

                        <strong
                          className={`risk-badge ${chunkRiskClass}`}
                        >
                          {level ||
                            "UNKNOWN"}
                        </strong>

                      </div>

                    </div>

                  );

                }
              )}

            </div>

          )}

        </div>


        {/* SECURITY DECISION */}

        {chunkResult.chunks?.length > 0 && (

          <div
            style={{
              marginTop: "28px",
            }}
          >

            <div className="panel-heading">

              <div>

                <span className="section-number">
                  DS
                </span>


                <div>

                  <h2>
                    Security Decision
                  </h2>

                  <p>
                    Decision generated from
                    continuous chunk-level
                    forensic evidence.
                  </p>

                </div>

              </div>

            </div>


            <div className="tamper-stat-grid">

              <div className="stat-card">

                <span>
                  Persistent Attack
                </span>

                <strong>
                  {chunkResult.continuous_alert
                    ? "YES"
                    : "NO"}
                </strong>

                <small>
                  Consecutive risk detection
                </small>

              </div>


              <div className="stat-card">

                <span>
                  High Risk Evidence
                </span>

                <strong>
                  {chunkResult.high_risk_chunks || 0}
                </strong>

                <small>
                  HIGH / CRITICAL chunks
                </small>

              </div>


              <div className="stat-card">

                <span>
                  Verification
                </span>

                <strong>
                  {chunkResult.continuous_alert
                    ? "REQUIRED"
                    : "NORMAL"}
                </strong>

                <small>
                  {chunkResult.continuous_alert
                    ? "Use callback / MFA"
                    : "Continue normal verification"}
                </small>

              </div>

            </div>

          </div>

        )}

      </section>

    </div>
  );


  /* ======================================================
     LIVE MONITORING PAGE
  ====================================================== */

  const renderLiveMonitoringPage = () => {

    const currentLevel =
      String(
        liveResult?.risk_level || ""
      ).toUpperCase();

    const currentRiskClass =
      currentLevel === "LOW"
        ? "risk-low"
        : currentLevel === "MEDIUM"
        ? "risk-medium"
        : "risk-high";

    const liveDynamicRisk = Math.max(
      0,
      Math.min(100, Number(liveResult?.risk_score || 0))
    );

    const liveRiskAction =
      liveDynamicRisk >= 80
        ? "Block this contact and request independent verification."
        : liveDynamicRisk >= 60
        ? "Avoid sensitive actions and verify the caller independently."
        : liveDynamicRisk >= 30
        ? "Continue monitoring. Additional verification is recommended."
        : "No blocking action is currently required.";

    return (

      <>
        <header className="topbar">

          <div>

            <p className="eyebrow">
              REAL-TIME VOICE SECURITY
            </p>

            <h1>
              Live Microphone Monitoring
            </h1>

            <p className="top-description">
              Capture the microphone in
              approximately 5-second chunks
              and analyze each chunk through
              SonicT F1–F5 and the Extra Trees
              fusion model.
            </p>

          </div>


          <div className="top-badge">

            <span
              className="live-dot"
              style={{
                opacity:
                  liveMonitoring
                    ? 1
                    : 0.35,
              }}
            ></span>

            {liveMonitoring
              ? "LIVE MONITORING"
              : "MONITORING STOPPED"}

          </div>

        </header>

        <section className="panel">

          <div className="panel-heading">

            <div>

              <span className="section-number">
                LIVE
              </span>


              <div>

                <h2>
                  Microphone Capture
                </h2>

                <p>
                  SonicT records independent
                  browser audio chunks and
                  sends them to
                  /analyze-live-chunk.
                </p>

              </div>

            </div>

          </div>

          <div className="caller-monitor-card">
            <div className="caller-avatar">●</div>
            <div className="caller-monitor-title">
              <span>LIVE CALLER DETAILS</span>
              <strong>{callerDetails.phone || "Unknown caller"}</strong>
              <small>Manual entry · Automatic caller ID requires phone/telephony integration</small>
            </div>
            <div className="caller-detail-fields">
              <label>Caller number<input value={callerDetails.phone} onChange={(event) => setCallerDetails((current) => ({ ...current, phone: event.target.value }))} placeholder="Phone number" /></label>
              <label>Location<input value={callerDetails.location} onChange={(event) => setCallerDetails((current) => ({ ...current, location: event.target.value }))} placeholder="City, country" /></label>
              <label>Type<select value={callerDetails.type} onChange={(event) => setCallerDetails((current) => ({ ...current, type: event.target.value }))}><option>Mobile</option><option>Landline</option><option>VoIP</option><option>Unknown</option></select></label>
              <label>Known contact<select value={callerDetails.knownContact} onChange={(event) => setCallerDetails((current) => ({ ...current, knownContact: event.target.value }))}><option>Yes</option><option>No</option><option>Unknown</option></select></label>
            </div>
          </div>


          <div className={`android-sync-card ${androidSyncEnabled ? "connected" : ""}`}>
            <div className="android-sync-icon">AP</div>
            <div className="android-sync-copy">
              <span>ANDROID APK CONNECTION</span>
              <strong>{androidSyncStatus}</strong>
              <small>
                {androidLastSeen
                  ? `Latest result: ${new Date(androidLastSeen).toLocaleString()}`
                  : "Displays the APK's latest F1–F5 analysis in this existing dashboard."}
              </small>
            </div>
            <button
              type="button"
              onClick={() => {
                androidLatestIdRef.current = null;
                setAndroidSyncEnabled((enabled) => !enabled);
                if (androidSyncEnabled) {
                  setAndroidSyncStatus("Not connected");
                  setAndroidLastSeen("");
                }
              }}
            >
              {androidSyncEnabled ? "Disconnect APK" : "Connect Android APK"}
            </button>
          </div>


          <div className="tamper-stat-grid">

            <div className="stat-card">

              <span>
                Monitoring Status
              </span>

              <strong>
                {liveMonitoring
                  ? "ACTIVE"
                  : "STOPPED"}
              </strong>

              <small>
                Browser microphone
              </small>

            </div>


            <div className="stat-card">

              <span>
                Chunk Window
              </span>

              <strong>
                5 SEC
              </strong>

              <small>
                Independent live recording
              </small>

            </div>


            <div className="stat-card">

              <span>
                Analyzed Chunks
              </span>

              <strong>
                {liveChunkCount}
              </strong>

              <small>
                Successful backend results
              </small>

            </div>

          </div>


          {!liveMonitoring ? (

            <button
              className="analyze-button"
              onClick={
                startLiveMonitoring
              }
            >

              <span>●</span>

              Start Live Monitoring

            </button>

          ) : (

            <button
              className="analyze-button"
              onClick={
                stopLiveMonitoring
              }
            >

              <span>■</span>

              Stop Live Monitoring

            </button>

          )}


          {liveProcessing && (

            <div
              className="analysis-progress"
              style={{
                marginTop: "20px",
                marginBottom: 0,
              }}
            >

              <div className="scanner">

                <div className="scanner-line"></div>

              </div>


              <div>

                <h3>
                  Analyzing live voice chunk
                </h3>

                <p>
                  Running F1–F5 forensic
                  models and calculating the
                  current voice integrity risk.
                </p>

              </div>

            </div>

          )}


          {liveError && (

            <div className="error-message">

              <span>!</span>

              <div>

                <h3>
                  Live Monitoring Error
                </h3>

                <p>
                  {liveError}
                </p>

              </div>

            </div>

          )}

        </section>


        {liveResult && (

          <section className="panel">

            <div className="panel-heading">

              <div>

                <span className="section-number">
                  NOW
                </span>


                <div>

                  <h2>
                    Current Live Result
                  </h2>

                  <p>
                    Most recently analyzed
                    microphone chunk.
                  </p>

                </div>

              </div>


              <span
                className={`risk-badge ${currentRiskClass}`}
              >

                {currentLevel ||
                  "UNKNOWN"}

              </span>

            </div>


            <div className="tamper-stat-grid">

              <div className="stat-card">

                <span>
                  Classification
                </span>

                <strong>
                  {String(
                    liveResult.classification ||
                      "-"
                  ).toUpperCase()}
                </strong>

                <small>
                  SonicT fusion decision
                  {liveResult.source_client === "android" ? " · Android APK" : ""}
                </small>

              </div>


              <div className="stat-card">

                <span>
                  Confidence
                </span>

                <strong>
                  {percent(
                    liveResult.confidence
                  )}
                </strong>

                <small>
                  Final classification
                  confidence
                </small>

              </div>


              <div className="stat-card">

                <span>
                  Dynamic Risk Score
                </span>

                <strong>
                  {Number(
                    liveResult.risk_score ||
                      0
                  ).toFixed(2)}%
                </strong>

                <small>
                  Current live chunk
                </small>

              </div>

            </div>

            <div className="live-feature-evidence">
              <p className="mini-label">LIVE FIVE-LAYER FORENSICS</p>
              <h2>F1–F5 Evidence Scores</h2>
              <p className="live-feature-intro">
                Independent evidence probabilities returned by the SonicT backend
                for the latest five-second microphone chunk.
              </p>

              <div className="feature-grid">
                {liveFeatureCards.map((feature) => (
                  <div className="feature-card" key={`live-${feature.code}`}>
                    <div className="feature-header">
                      <div className="feature-icon">{feature.icon}</div>
                      <span>{feature.code}</span>
                    </div>

                    <h3>{feature.title}</h3>
                    <p className="model-name">{feature.model}</p>
                    <div className="feature-score">{percent(feature.value)}</div>
                    <div className="feature-bar">
                      <div
                        style={{
                          width: `${Math.min(
                            Math.max(Number(feature.value || 0) * 100, 0),
                            100
                          )}%`,
                        }}
                      />
                    </div>
                    <span className="evidence-label">Evidence probability</span>
                  </div>
                ))}
              </div>
            </div>

            <div className={`live-risk-notification ${liveDynamicRisk >= 80 ? "critical" : liveDynamicRisk >= 60 ? "high" : liveDynamicRisk >= 30 ? "medium" : "safe"}`}>
              <div className="risk-notification-icon">!</div>
              <div>
                <span>DYNAMIC SECURITY RECOMMENDATION</span>
                <h3>{liveDynamicRisk >= 80 ? "Suspicious call detected" : liveDynamicRisk >= 60 ? "High-risk caller activity" : liveDynamicRisk >= 30 ? "Caller requires attention" : "Call remains within the safe range"}</h3>
                <p>{liveRiskAction}</p>
              </div>
              {liveDynamicRisk >= 80 && (
                <button type="button" onClick={() => setContactBlocked(true)} disabled={contactBlocked}>
                  {contactBlocked ? "Contact marked blocked" : "Block Contact"}
                </button>
              )}
            </div>


            {liveResult
              .class_probabilities && (

              <div
                style={{
                  marginTop: "24px",
                }}
              >

                <p className="mini-label">
                  LIVE MODEL DISTRIBUTION
                </p>

                <h2>
                  Class Probabilities
                </h2>


                <div className="probability-list">

                  {Object.entries(
                    liveResult
                      .class_probabilities
                  )
                    .sort(
                      (a, b) =>
                        b[1] -
                        a[1]
                    )

                    .map(
                      ([
                        label,
                        probability,
                      ]) => (

                        <div
                          className="probability-item"
                          key={label}
                        >

                          <div className="probability-header">

                            <span>
                              {label}
                            </span>

                            <strong>
                              {percent(
                                probability
                              )}
                            </strong>

                          </div>


                          <div className="progress-track">

                            <div
                              className={`progress-fill probability-${label.toLowerCase()}`}

                              style={{
                                width: `${Math.min(
                                  Number(
                                    probability ||
                                      0
                                  ) *
                                    100,
                                  100
                                )}%`,
                              }}
                            />

                          </div>

                        </div>

                      )
                    )}

                </div>

              </div>

            )}


            {liveResult
              .security_alert && (

              <div
                className={
                  liveResult
                    .security_alert
                    .alert
                    ? "error-message"
                    : "safe-window-message"
                }

                style={{
                  marginTop: "22px",
                }}
              >

                <span>
                  {liveResult
                    .security_alert
                    .alert
                    ? "!"
                    : "✓"}
                </span>


                <div>

                  <h3>
                    {
                      liveResult
                        .security_alert
                        .message
                    }
                  </h3>


                  <p>

                    <strong>
                      Recommended Action:
                    </strong>{" "}

                    {
                      liveResult
                        .security_alert
                        .action
                    }

                  </p>

                </div>

              </div>

            )}

          </section>

        )}


        <section className="panel">

          <div className="panel-heading">

            <div>

              <span className="section-number">
                CT
              </span>

              <div>

                <h2>
                  Continuous Live Threat
                </h2>

              <p className="important-explanation">
                  SonicT raises a persistent
                  threat alert when two or more
                  consecutive live chunks are
                  HIGH or CRITICAL risk.
                </p>

              </div>

            </div>

            <span
              className={`risk-badge ${
                liveContinuousThreat
                  ? "risk-high"
                  : "risk-low"
              }`}
            >
              {liveContinuousThreat
                ? "THREAT DETECTED"
                : "NO PERSISTENT THREAT"}
            </span>

          </div>


          <div className="tamper-stat-grid">

            <div className="stat-card">
              <span>Persistent Attack</span>

              <strong>
                {liveContinuousThreat
                  ? "YES"
                  : "NO"}
              </strong>

            </div>

            <div className="stat-card">
              <span>Threat Sequences</span>

              <strong>
                {liveSuspiciousSegments.length}
              </strong>

            </div>

            <div className="stat-card">
              <span>Verification</span>

              <strong>
                {liveContinuousThreat
                  ? "REQUIRED"
                  : "NORMAL"}
              </strong>

            </div>

          </div>


          {liveContinuousThreat ? (

            <div
              className="error-message"
              style={{
                marginTop: "22px",
              }}
            >

              <span>!</span>

              <div>

                <h3>
                  Continuous Suspicious Voice Activity Detected
                </h3>

                <p>
                  Two or more consecutive live
                  microphone chunks reached HIGH
                  or CRITICAL voice integrity risk.
                </p>

                <p>
                  <strong>
                    Recommended Action:
                  </strong>{" "}
                  Perform secondary verification
                  immediately using callback, MFA
                  or another trusted channel.
                </p>

              </div>

            </div>

          ) : (

            <div
              className="safe-window-message"
              style={{
                marginTop: "22px",
              }}
            >

              <div className="safe-check">
                ✓
              </div>

              <div>

                <h3>
                  No Continuous Threat Detected
                </h3>


              </div>

            </div>

          )}


          {liveSuspiciousSegments.length > 0 && (

            <div
              style={{
                marginTop: "26px",
              }}
            >

              <div className="panel-heading">

                <div>

                  <div>
                    <h2>
                      Continuous Suspicious Segments
                    </h2>

                    <p>
                      Persistent HIGH / CRITICAL
                      ranges detected during this
                      microphone session.
                    </p>
                  </div>

                </div>

                <div className="window-count">
                  {liveSuspiciousSegments.length} detected
                </div>

              </div>


              <div className="window-list">

                {liveSuspiciousSegments.map(
                  (segment, index) => (

                    <div
                      className="window-card"
                      key={`${segment.start}-${segment.end}-${index}`}
                    >

                      <div className="window-index">
                        {String(index + 1).padStart(
                          2,
                          "0"
                        )}
                      </div>

                      <div className="window-time">
                        <span>LIVE RANGE</span>

                        <strong>
                          {Number(
                            segment.start || 0
                          ).toFixed(2)}s
                          {" → "}
                          {Number(
                            segment.end || 0
                          ).toFixed(2)}s
                        </strong>
                      </div>

                      <div className="window-score">
                        <span>CONSECUTIVE CHUNKS</span>

                        <strong>
                          {segment.chunks || 0}
                        </strong>
                      </div>

                      <div className="window-score">
                        <span>MAXIMUM RISK</span>

                        <strong>
                          {Number(
                            segment.max_risk || 0
                          ).toFixed(2)}%
                        </strong>
                      </div>

                      <div className="window-score">
                        <span>STATUS</span>

                        <strong className="risk-badge risk-high">
                          THREAT
                        </strong>
                      </div>

                    </div>

                  )
                )}

              </div>

            </div>

          )}

        </section>


        <section className="panel">

          <div className="panel-heading">

            <div>

              <span className="section-number">
                HX
              </span>


              <div>

                <h2>
                  Live Risk History
                </h2>


              </div>

            </div>


            <div className="window-count">
              {liveHistory.length} results
            </div>

          </div>


          {liveHistory.length === 0 ? (

            <div className="safe-window-message">

              <div className="safe-check">
                i
              </div>


              <div>

                <h3>
                  No live chunks analyzed yet
                </h3>

                <p>
                  Start live monitoring and
                  allow microphone access.
                  The first result will appear
                  after a complete audio chunk
                  is processed.
                </p>

              </div>

            </div>

          ) : (

            <div className="window-list">

              {liveHistory.map(
                (
                  item,
                  index
                ) => {

                  const level =
                    String(
                      item.risk_level || ""
                    ).toUpperCase();

                  const riskClass =
                    level === "LOW"
                      ? "risk-low"
                      : level === "MEDIUM"
                      ? "risk-medium"
                      : level === "CRITICAL"
                      ? "risk-critical"
                      : "risk-high";

                  return (

                    <div
                      className="window-card"
                      key={`${item.timestamp}-${index}`}
                    >

                      <div className="window-index">

                        {String(
                          liveHistory.length -
                            index
                        ).padStart(
                          2,
                          "0"
                        )}

                      </div>


                      <div className="window-time">

                        <span>
                          ANALYSIS TIME
                        </span>

                        <strong>
                          {Number(
                            item.start || 0
                          ).toFixed(0)}s
                          {" → "}
                          {Number(
                            item.end || 0
                          ).toFixed(0)}s
                          {" • "}
                          {item.timestamp}
                        </strong>

                      </div>


                      <div className="window-score">

                        <span>
                          CLASS
                        </span>

                        <strong>
                          {String(
                            item.classification ||
                              "-"
                          ).toUpperCase()}
                        </strong>

                      </div>


                      <div className="window-score">

                        <span>
                          CONFIDENCE
                        </span>

                        <strong>
                          {percent(
                            item.confidence
                          )}
                        </strong>

                      </div>


                      <div className="window-score">

                        <span>
                          RISK
                        </span>

                        <strong>
                          {Number(
                            item.risk_score ||
                              0
                          ).toFixed(2)}%
                        </strong>

                      </div>


                      <div className="window-score">

                        <span>
                          RISK LEVEL
                        </span>

                        <strong
                          className={`risk-badge ${riskClass}`}
                        >
                          {level ||
                            "UNKNOWN"}
                        </strong>

                      </div>

                    </div>

                  );

                }
              )}

            </div>

          )}

        </section>

      </>

    );
  };


  /* ======================================================
     FORENSICS PAGE
  ====================================================== */

  const renderForensicsPage = () => {
    const report = result?.forensic_report || null;
    const assessment = report?.final_assessment || null;
    const reportInfo = report?.report_information || null;
    const evidenceIntegrity =
      report?.evidence_integrity ||
      result?.evidence_integrity ||
      null;
    const privacyRetention =
      report?.privacy_and_retention ||
      result?.privacy_and_retention ||
      null;
    const speaker = report?.speaker_consistency || null;
    const reportAlert = report?.security_alert || null;

    const reportRiskLevel = String(
      assessment?.risk_level ||
        result?.voice_integrity_risk?.risk_level ||
        ""
    ).toUpperCase();

    const reportRiskClass =
      reportRiskLevel === "LOW"
        ? "risk-low"
        : reportRiskLevel === "MEDIUM"
        ? "risk-medium"
        : "risk-high";

    return (
      <>
        <header className="topbar">
          <div>
            <p className="eyebrow">
              FORENSIC EVIDENCE
            </p>

            <h1>
              Detailed Forensics
            </h1>

            <p className="top-description">
              Review the structured SonicT forensic report,
              F1–F5 evidence, temporal tampering information
              and the final security assessment.
            </p>
          </div>

          {report && (
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "12px",
                flexWrap: "wrap",
              }}
            >
              <div className="top-badge">
                <span className="live-dot"></span>
                FORENSIC REPORT READY
              </div>

              <button
                className="clear-button"
                onClick={downloadForensicReport}
                disabled={reportDownloading}
              >
                {reportDownloading
                  ? "Generating PDF..."
                  : "Download PDF"}
              </button>
            </div>
          )}
        </header>

        {reportDownloadError && (
          <div
            className="error-message"
            style={{
              marginBottom: "20px",
            }}
          >
            <span>!</span>
            {reportDownloadError}
          </div>
        )}

        {!result ? (
          <section className="panel">
            <div className="safe-window-message">
              <div className="safe-check">i</div>

              <div>
                <h3>No current analysis</h3>
                <p>Analyze an audio file first.</p>
              </div>
            </div>

            <button
              className="analyze-button"
              onClick={() => changePage("analysis")}
            >
              Go to Analysis
            </button>
          </section>
        ) : (
          <>
            {report ? (
              <section className="panel">
                <div className="panel-heading">
                  <div>
                    <span className="section-number">
                      FR
                    </span>

                    <div>
                      <h2>
                        SonicT Forensic Report
                      </h2>

                      <p>
                        Structured investigator-facing summary
                        generated from the current SonicT analysis.
                      </p>
                    </div>
                  </div>

                  <span
                    className={`risk-badge ${reportRiskClass}`}
                  >
                    {reportRiskLevel || "UNKNOWN"} RISK
                  </span>
                </div>

                <div className="tamper-stat-grid">
                  <div className="stat-card">
                    <span>Final Classification</span>

                    <strong>
                      {String(
                        assessment?.classification ||
                          result.classification ||
                          "-"
                      ).toUpperCase()}
                    </strong>

                    <small>Fusion model decision</small>
                  </div>

                  <div className="stat-card">
                    <span>Confidence</span>

                    <strong>
                      {assessment?.confidence_percentage != null
                        ? `${Number(
                            assessment.confidence_percentage
                          ).toFixed(2)}%`
                        : percent(result.confidence)}
                    </strong>

                    <small>
                      Final classification confidence
                    </small>
                  </div>

                  <div className="stat-card">
                    <span>Voice Integrity Risk</span>

                    <strong>
                      {Number(
                        assessment?.voice_integrity_risk ??
                          result.voice_integrity_risk?.risk_score ??
                          0
                      ).toFixed(2)}%
                    </strong>

                    <small>
                      Security-oriented risk score
                    </small>
                  </div>

                  <div className="stat-card">
                    <span>Report File</span>

                    <strong>
                      {reportInfo?.filename ||
                        result.input_file?.original_filename ||
                        file?.name ||
                        "-"}
                    </strong>

                    <small>
                      Evidence under examination
                    </small>
                  </div>

                  <div className="stat-card">
                    <span>Generated At</span>

                    <strong>
                      {formatDate(reportInfo?.generated_at)}
                    </strong>

                    <small>
                      Report generation time
                    </small>
                  </div>

                  <div className="stat-card">
                    <span>Speaker Consistency</span>

                    <strong>
                      {speaker?.evaluated
                        ? String(
                            speaker.status || "UNKNOWN"
                          ).toUpperCase()
                        : "NOT EVALUATED"}
                    </strong>

                    <small>
                      Cross-session speaker check
                    </small>
                  </div>
                </div>

                <div style={{ marginTop: "28px" }}>
                  <div className="panel-heading">
                    <div>
                      <span className="section-number">
                        SHA
                      </span>

                      <div>
                        <h2>Evidence Integrity</h2>

                        <p>
                          SHA-256 fingerprint generated from the
                          original uploaded evidence before audio
                          conversion or normalization.
                        </p>
                      </div>
                    </div>

                    <span className="top-badge">
                      {evidenceIntegrity?.status ||
                        "NOT AVAILABLE"}
                    </span>
                  </div>

                  <div className="tamper-stat-grid">
                    <div className="stat-card">
                      <span>Hash Algorithm</span>

                      <strong>
                        {evidenceIntegrity?.algorithm ||
                          "SHA-256"}
                      </strong>

                      <small>
                        Evidence fingerprint method
                      </small>
                    </div>

                    <div className="stat-card">
                      <span>Integrity Status</span>

                      <strong>
                        {String(
                          evidenceIntegrity?.status ||
                            "NOT AVAILABLE"
                        ).toUpperCase()}
                      </strong>

                      <small>
                        Original evidence hash state
                      </small>
                    </div>
                  </div>

                  <div
                    className="safe-window-message"
                    style={{
                      marginTop: "18px",
                      alignItems: "flex-start",
                    }}
                  >
                    <div className="safe-check">#</div>

                    <div
                      style={{
                        width: "100%",
                        minWidth: 0,
                      }}
                    >
                      <h3>SHA-256 Evidence Fingerprint</h3>

                      <p
                        style={{
                          fontFamily:
                            "Consolas, 'Courier New', monospace",
                          overflowWrap: "anywhere",
                          wordBreak: "break-all",
                          lineHeight: 1.7,
                        }}
                      >
                        {evidenceIntegrity?.sha256 ||
                          result?.input_file
                            ?.evidence_sha256 ||
                          "Hash not available. Analyze the evidence again using the SHA-256-enabled backend."}
                      </p>

                      <p>
                        {evidenceIntegrity?.description ||
                          "This fingerprint can be compared later to verify that the evidence file remains byte-for-byte unchanged."}
                      </p>
                    </div>
                  </div>
                </div>

                <div
                  className={
                    ["HIGH", "CRITICAL"].includes(reportRiskLevel)
                      ? "error-message"
                      : "safe-window-message"
                  }
                  style={{ marginTop: "22px" }}
                >
                  <span>
                    {["HIGH", "CRITICAL"].includes(reportRiskLevel)
                      ? "!"
                      : "✓"}
                  </span>

                  <div>
                    <div style={{ marginTop: "28px" }}>
                    <div className="panel-heading">
                      <div>
                        <span className="section-number">
                          P
                        </span>

                        <div>
                          <h2>Privacy & Evidence Retention</h2>

                          <p>
                            Shows how SonicT handles uploaded audio
                            during forensic analysis.
                          </p>
                        </div>
                      </div>

                      <span className="top-badge">
                        {privacyRetention?.privacy_status ||
                          "AUTO CLEANUP"}
                      </span>
                    </div>

                    <div className="tamper-stat-grid">
                      <div className="stat-card">
                        <span>Raw Audio Retained</span>

                        <strong>
                          {privacyRetention?.raw_audio_retained
                            ? "YES"
                            : "NO"}
                        </strong>

                        <small>
                          Original evidence is not permanently retained
                        </small>
                      </div>

                      <div className="stat-card">
                        <span>Temporary Storage</span>

                        <strong>
                          {privacyRetention?.temporary_storage
                            ? "YES"
                            : "NO"}
                        </strong>

                        <small>
                          Used only while the request is processed
                        </small>
                      </div>

                      <div className="stat-card">
                        <span>Automatic Cleanup</span>

                        <strong>
                          {privacyRetention?.automatic_cleanup
                            ? "ENABLED"
                            : "NOT REPORTED"}
                        </strong>

                        <small>
                          Temporary audio files are removed automatically
                        </small>
                      </div>

                      <div className="stat-card">
                        <span>Stored Record</span>

                        <strong>
                          METADATA / RESULT
                        </strong>

                        <small>
                          {privacyRetention?.stored_record_type ||
                            "Analysis metadata and forensic result only"}
                        </small>
                      </div>
                    </div>

                    <div
                      className="safe-window-message"
                      style={{
                        marginTop: "18px",
                        alignItems: "flex-start",
                      }}
                    >
                      <div className="safe-check">✓</div>

                      <div>
                        <h3>Retention Policy</h3>

                        <p>
                          {privacyRetention?.retention_policy ||
                            "Temporary uploaded and converted audio files are deleted after request processing."}
                        </p>

                        <p>
                          Processing scope:{" "}
                          <strong>
                            {privacyRetention?.processing_scope ||
                              "Current SonicT backend"}
                          </strong>
                        </p>
                      </div>
                    </div>
                  </div>

                  <h3>Forensic Conclusion</h3>

                    <p>
                      {report.forensic_conclusion ||
                        "No forensic conclusion was returned."}
                    </p>

                    {report.recommended_action && (
                      <p>
                        <strong>
                          Recommended Action:
                        </strong>{" "}
                        {report.recommended_action}
                      </p>
                    )}
                  </div>
                </div>

                {reportAlert && (
                  <div
                    className={
                      reportAlert.alert
                        ? "error-message"
                        : "safe-window-message"
                    }
                    style={{ marginTop: "18px" }}
                  >
                    <span>
                      {reportAlert.alert ? "!" : "✓"}
                    </span>

                    <div>
                      <h3>Security Decision</h3>

                      <p>{reportAlert.message}</p>

                      <p>
                        <strong>Action:</strong>{" "}
                        {reportAlert.action}
                      </p>
                    </div>
                  </div>
                )}

                <div style={{ marginTop: "28px" }}>
                  <div className="panel-heading">
                    <div>
                      <span className="section-number">
                        SP
                      </span>

                      <div>
                        <h2>Speaker Consistency</h2>

                        <p>
                          Cross-session speaker identity
                          verification information.
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="tamper-stat-grid">
                    <div className="stat-card">
                      <span>Evaluation Status</span>

                      <strong>
                        {speaker?.evaluated
                          ? "EVALUATED"
                          : "NOT EVALUATED"}
                      </strong>

                      <small>Current evidence</small>
                    </div>

                    <div className="stat-card">
                      <span>Speaker Status</span>

                      <strong>
                        {String(
                          speaker?.status || "NOT EVALUATED"
                        ).toUpperCase()}
                      </strong>

                      <small>VERIFIED / MISMATCH</small>
                    </div>

                    <div className="stat-card">
                      <span>Similarity</span>

                      <strong>
                        {speaker?.similarity_percentage != null
                          ? `${Number(
                              speaker.similarity_percentage
                            ).toFixed(2)}%`
                          : "-"}
                      </strong>

                      <small>
                        Available after speaker verification
                      </small>
                    </div>
                  </div>
                </div>

                <div style={{ marginTop: "30px" }}>
                  <p className="mini-label">
                    FORENSIC MODEL DISTRIBUTION
                  </p>

                  <h2>Class Probabilities</h2>

                  <div className="probability-list">
                    {Object.entries(
                      report.class_probabilities ||
                        result.class_probabilities ||
                        {}
                    )
                      .sort((a, b) => b[1] - a[1])
                      .map(([label, probability]) => (
                        <div
                          className="probability-item"
                          key={label}
                        >
                          <div className="probability-header">
                            <span>{label}</span>

                            <strong>
                              {percent(probability)}
                            </strong>
                          </div>

                          <div className="progress-track">
                            <div
                              className={`progress-fill probability-${label.toLowerCase()}`}
                              style={{
                                width: `${Math.min(
                                  Number(probability || 0) * 100,
                                  100
                                )}%`,
                              }}
                            />
                          </div>
                        </div>
                      ))}
                  </div>
                </div>

                {report.disclaimer && (
                  <div
                    className="safe-window-message"
                    style={{ marginTop: "24px" }}
                  >
                    <div className="safe-check">i</div>

                    <div>
                      <h3>Forensic Use Notice</h3>
                      <p>{report.disclaimer}</p>
                    </div>
                  </div>
                )}
              </section>
            ) : (
              <section className="panel">
                <div className="safe-window-message">
                  <div className="safe-check">i</div>

                  <div>
                    <h3>
                      Structured report not available
                    </h3>

                    <p>
                      The analysis result is available, but the
                      backend did not return forensic_report.
                      Restart the updated backend and analyze
                      the evidence again.
                    </p>
                  </div>
                </div>
              </section>
            )}

            {result?.experimental_f1b && (
              <section className="panel experimental-evidence-panel">
                <div className="panel-heading">
                  <div>
                    <span className="section-number">F1B</span>

                    <div>
                      <h2>Experimental Robustness Evidence</h2>
                      <p>
                        Separate WavLM robustness layer for additional deepfake evidence.
                        This output is not part of the validated F1–F5 Extra Trees fusion.
                      </p>
                    </div>
                  </div>

                  <span className="top-badge">EXPERIMENTAL</span>
                </div>

                <div className="tamper-stat-grid">
                  <div className="stat-card">
                    <span>Experimental Prediction</span>
                    <strong>
                      {String(
                        result.experimental_f1b.prediction || "UNKNOWN"
                      ).toUpperCase()}
                    </strong>
                    <small>Separate robustness-layer decision</small>
                  </div>

                  <div className="stat-card">
                    <span>Deepfake Probability</span>
                    <strong>
                      {percent(
                        result.experimental_f1b.deepfake_probability
                      )}
                    </strong>
                    <small>Experimental F1B evidence only</small>
                  </div>

                  <div className="stat-card">
                    <span>Fusion Participation</span>
                    <strong>SEPARATE</strong>
                    <small>Does not modify the Extra Trees classification</small>
                  </div>
                </div>

                <div className="safe-window-message" style={{ marginTop: "18px" }}>
                  <div className="safe-check">i</div>
                  <div>
                    <h3>Experimental Evidence Notice</h3>
                    <p>
                      F1B is displayed as supporting robustness evidence. The final
                      SonicT forensic classification continues to come from the
                      validated F1–F5 feature fusion pipeline.
                    </p>
                  </div>
                </div>
              </section>
            )}

            <section className="panel">
              <div className="panel-heading">
                <div>
                  <span className="section-number">F</span>

                  <div>
                    <h2>Five Forensic Models</h2>
                    <p>
                      Individual evidence from SonicT F1–F5.
                    </p>
                  </div>
                </div>
              </div>

              <div className="feature-grid">
                {featureCards.map((feature) => (
                  <div
                    className="feature-card"
                    key={feature.code}
                  >
                    <div className="feature-header">
                      <div className="feature-icon">
                        {feature.icon}
                      </div>

                      <span>{feature.code}</span>
                    </div>

                    <h3>{feature.title}</h3>

                    <p className="model-name">
                      {feature.model}
                    </p>

                    <div className="feature-score">
                      {percent(feature.value)}
                    </div>

                    <div className="feature-bar">
                      <div
                        style={{
                          width: `${Math.min(
                            Number(feature.value || 0) * 100,
                            100
                          )}%`,
                        }}
                      />
                    </div>

                    <span className="evidence-label">
                      Evidence probability
                    </span>
                  </div>
                ))}
              </div>
            </section>

            <section className="panel">
              <div className="panel-heading">
                <div>
                  <span className="section-number">TM</span>

                  <div>
                    <h2>
                      Temporal Tampering Features
                    </h2>

                    <p>
                      Statistical evidence generated from
                      the F4 boundary detector.
                    </p>
                  </div>
                </div>
              </div>

              <div className="tamper-stat-grid">
                <div className="stat-card">
                  <span>F4 Maximum</span>
                  <strong>
                    {percent(result.tampering?.f4_max)}
                  </strong>
                </div>

                <div className="stat-card">
                  <span>F4 Mean</span>
                  <strong>
                    {percent(result.tampering?.f4_mean)}
                  </strong>
                </div>

                <div className="stat-card">
                  <span>F4 Median</span>
                  <strong>
                    {percent(result.tampering?.f4_median)}
                  </strong>
                </div>

                <div className="stat-card">
                  <span>F4 Standard Deviation</span>
                  <strong>
                    {percent(result.tampering?.f4_std)}
                  </strong>
                </div>

                <div className="stat-card">
                  <span>Suspicious Ratio</span>
                  <strong>
                    {percent(
                      result.tampering?.f4_suspicious_ratio
                    )}
                  </strong>
                </div>

                <div className="stat-card">
                  <span>High Risk Ratio</span>
                  <strong>
                    {percent(
                      result.tampering?.f4_high_ratio
                    )}
                  </strong>
                </div>
              </div>
            </section>

            <section className="panel">
              <div className="panel-heading">
                <div>
                  <span className="section-number">RG</span>

                  <div>
                    <h2>Suspicious Audio Regions</h2>

                    <p>
                      Localized temporal regions requiring
                      investigator review.
                    </p>
                  </div>
                </div>

                <div className="window-count">
                  {result.tampering?.suspicious_windows?.length || 0}{" "}
                  regions
                </div>
              </div>

              {!result.tampering?.suspicious_windows ||
              result.tampering.suspicious_windows.length === 0 ? (
                <div className="safe-window-message">
                  <div className="safe-check">✓</div>

                  <div>
                    <h3>
                      No suspicious regions detected
                    </h3>

                    <p>
                      No temporal windows crossed the
                      configured suspicious threshold.
                    </p>
                  </div>
                </div>
              ) : (
                <div className="window-list">
                  {result.tampering.suspicious_windows.map(
                    (window, index) => (
                      <div
                        className="window-card"
                        key={index}
                      >
                        <div className="window-index">
                          {String(index + 1).padStart(2, "0")}
                        </div>

                        <div className="window-time">
                          <span>TIME RANGE</span>

                          <strong>
                            {Number(window.start || 0).toFixed(2)}s
                            {" → "}
                            {Number(window.end || 0).toFixed(2)}s
                          </strong>
                        </div>

                        <div className="window-score">
                          <span>PROBABILITY</span>

                          <strong>
                            {percent(window.probability)}
                          </strong>
                        </div>
                      </div>
                    )
                  )}
                </div>
              )}
            </section>
          </>
        )}
      </>
    );
  };


/* ======================================================
     REPORTS
  ====================================================== */

  const renderReportsPage = () => (

    <>
      <header className="topbar">

        <div>

          <h1>
            Forensic Reports
          </h1>

          <p className="top-description">
            Previous SonicT analyses stored
            in SQLite.
          </p>

        </div>


        <button
          className="clear-button"
          onClick={loadReports}
        >
          Refresh
        </button>

      </header>


      <section className="panel">

        <div className="panel-heading">

          <div>

            <span className="section-number">
              R
            </span>


            <div>

              <h2>
                Saved Analyses
              </h2>

              <p>
                SQLite forensic history.
              </p>

            </div>

          </div>


          <div className="window-count">
            {reports.length} reports
          </div>

        </div>


        {reportsLoading ? (

          <div className="analysis-progress">

            <span className="spinner"></span>

            Loading reports...

          </div>

        ) : reportsError ? (

          <div className="error-message">

            <span>!</span>

            {reportsError}

          </div>

        ) : reports.length === 0 ? (

          <div className="safe-window-message">

            <div className="safe-check">
              i
            </div>


            <div>

              <h3>
                No reports found
              </h3>

              <p>
                Analyze your first audio file.
              </p>

            </div>

          </div>

        ) : (

          <div className="window-list">

            {reports.map(
              (
                report,
                index
              ) => (

                <div
                  className="window-card"
                  key={
                    report.id ||
                    index
                  }
                >

                  <div className="window-index">
                    #
                    {report.id ||
                      index + 1}
                  </div>


                  <div className="window-time">

                    <span>
                      AUDIO FILE
                    </span>

                    <strong>
                      {report.filename ||
                        "Unknown"}
                    </strong>

                    <small>
                      {formatDate(
                        report.created_at
                      )}
                    </small>

                  </div>


                  <div className="window-score">

                    <span>
                      CLASSIFICATION
                    </span>

                    <strong>
                      {(
                        report.classification ||
                        "-"
                      ).toUpperCase()}
                    </strong>

                  </div>


                  <div className="window-score">

                    <span>
                      CONFIDENCE
                    </span>

                    <strong>
                      {percent(
                        report.confidence
                      )}
                    </strong>

                  </div>

                </div>

              )
            )}

          </div>
        )}

      </section>

    </>
  );


  /* ======================================================
     MODEL STATUS
  ====================================================== */

  const renderModelStatusPage = () => {

    const backendOnline =
      systemStatus.api;


    const metricValue = (value) =>
      value === undefined || value === null
        ? "No analysis yet"
        : percent(Number(value));

    const modelList = [

      {
        key: "F1",
        name:
          "F1 Voice Clone Detection",
        model:
          "WavLM + Attention Classifier",
        purpose: "Detects AI-generated or voice-cloned speech patterns.",
        statistics: [
          ["Voice Clone Probability", metricValue(result?.features?.voice_clone_probability)],
          ["Latest Classification", String(result?.classification || "No analysis yet").toUpperCase()],
          ["Input Evidence", result ? "Latest uploaded audio" : "Waiting for analysis"],
          ["Model Status", backendOnline ? "ONLINE" : "UNKNOWN"],
        ],
      },

      {
        key: "F2",
        name:
          "F2 Spectrogram Analysis",
        model:
          "Log-Mel + CNN",
        purpose: "Examines time-frequency artifacts commonly found in manipulated audio.",
        statistics: [
          ["Artifact Probability", metricValue(result?.features?.spectrogram_probability)],
          ["Representation", "Log-Mel Spectrogram"],
          ["Analysis Type", "CNN artifact detection"],
          ["Model Status", backendOnline ? "ONLINE" : "UNKNOWN"],
        ],
      },

      {
        key: "F3",
        name:
          "F3 Voice Feature Analysis",
        model:
          "MFCC + Random Forest",
        purpose: "Measures acoustic voice features for genuine-versus-synthetic evidence.",
        statistics: [
          ["Voice Feature Probability", metricValue(result?.features?.voice_feature_probability)],
          ["Feature Family", "MFCC + Acoustic"],
          ["Classifier", "Random Forest"],
          ["Model Status", backendOnline ? "ONLINE" : "UNKNOWN"],
        ],
      },

      {
        key: "F4",
        name:
          "F4 Tampering Detection",
        model:
          "Boundary CNN + Temporal",
        purpose: "Detects splicing or manipulation and localizes suspicious regions.",
        statistics: [
          ["Peak Tampering Score", metricValue(result?.tampering?.f4_max)],
          ["Mean Tampering Score", metricValue(result?.tampering?.f4_mean)],
          ["Suspicious Ratio", metricValue(result?.tampering?.f4_suspicious_ratio)],
          ["Suspicious Regions", result ? String(result?.tampering?.suspicious_windows?.length || 0) : "No analysis yet"],
        ],
      },

      {
        key: "F5",
        name:
          "F5 Replay Detection",
        model:
          "Replay CNN",
        purpose: "Detects audio replayed through speakers or captured from another device.",
        statistics: [
          ["Replay Probability", metricValue(result?.features?.replay_probability)],
          ["Attack Type", "Physical replay"],
          ["Classifier", "Replay CNN"],
          ["Model Status", backendOnline ? "ONLINE" : "UNKNOWN"],
        ],
      },

      {
        key: "FX",
        name:
          "Final Fusion",
        model:
          "17-Feature Extra Trees",
        purpose: "Combines F1–F5 evidence into the final SonicT decision.",
        statistics: [
          ["Final Classification", String(result?.classification || "No analysis yet").toUpperCase()],
          ["Confidence", metricValue(result?.confidence)],
          ["Dynamic Risk Score", result ? `${Number(result?.voice_integrity_risk?.risk_score || 0).toFixed(2)} / 100` : "No analysis yet"],
          ["Fusion Model", "Extra Trees"],
        ],
      },

      {
        key: "HM",
        name:
          "Human Mimicry Detection",
        model:
          "Random Forest V2 + Multi-Segment Median",
        purpose: "Separately evaluates whether a real person may be imitating another speaker.",
        statistics: [
          ["Human Mimic Probability", metricValue(mimicryResult?.probabilities?.human_mimic)],
          ["Real Human Probability", metricValue(mimicryResult?.probabilities?.real_human)],
          ["Segments Analyzed", mimicryResult ? String(mimicryResult?.segment_analysis?.segments_analyzed || 0) : "No analysis yet"],
          ["Final Assessment", String(mimicryResult?.classification || "No analysis yet").replaceAll("_", " ").toUpperCase()],
        ],
      },

    ];

    const selectedModel =
      modelList.find((item) => item.key === selectedModelKey) || modelList[0];


    return (

      <>

        <header className="topbar">

          <div>

            <p className="eyebrow">
              SYSTEM HEALTH
            </p>

            <h1>
              Model Status
            </h1>

          </div>


          <button
            className="clear-button"

            onClick={
              checkSystemStatus
            }
          >
            Check Status
          </button>

        </header>


        <section className="panel">

          <h2>
            SonicT Backend
          </h2>


          <span
            className={`risk-badge ${
              backendOnline
                ? "risk-low"
                : "risk-high"
            }`}
          >

            {systemStatus.loading
              ? "CHECKING"
              : backendOnline
              ? "ONLINE"
              : "OFFLINE"}

          </span>


          <p>
            {systemStatus.message}
          </p>

        </section>


        <section className="panel">

          <h2>
            Forensic Model Pipeline
          </h2>


          <div className="feature-grid">

            {modelList.map(
              (
                item,
                index
              ) => (

                <button
                  type="button"
                  className={`feature-card model-selector-card ${selectedModel.key === item.key ? "selected" : ""}`}
                  key={item.name}
                  onClick={() => setSelectedModelKey(item.key)}
                >

                  <div className="feature-header">

                    <div className="feature-icon">

                      {item.key}

                    </div>


                    <span>

                      {backendOnline
                        ? "ONLINE"
                        : "UNKNOWN"}

                    </span>

                  </div>


                  <h3>
                    {item.name}
                  </h3>


                  <p className="model-name">
                    {item.model}
                  </p>

                  <span className="model-view-statistics">
                    {selectedModel.key === item.key ? "VIEWING STATISTICS" : "VIEW STATISTICS"}
                  </span>

                </button>

              )
            )}

          </div>

          <div className="selected-model-statistics">
            <div className="selected-model-heading">
              <div className="feature-icon">{selectedModel.key}</div>
              <div>
                <span>SELECTED MODEL STATISTICS</span>
                <h2>{selectedModel.name}</h2>
                <p>{selectedModel.purpose}</p>
              </div>
              <span className={`risk-badge ${backendOnline ? "risk-low" : "risk-high"}`}>
                {backendOnline ? "ONLINE" : "UNKNOWN"}
              </span>
            </div>

            <div className="selected-model-stat-grid">
              {selectedModel.statistics.map(([label, value]) => (
                <div key={label}>
                  <span>{label}</span>
                  <strong>{value}</strong>
                </div>
              ))}
            </div>

            {!result && selectedModel.key !== "HM" && (
              <p className="model-stat-note">Run an audio analysis to populate live statistics for this model.</p>
            )}
            {!mimicryResult && selectedModel.key === "HM" && (
              <p className="model-stat-note">Run Human Mimicry Analysis to populate these statistics.</p>
            )}
          </div>

        </section>

      </>

    );
  };


  /* ======================================================
     DASHBOARD + STRUCTURED SECURITY PAGES
  ====================================================== */

  const renderDashboardPage = () => {
    const dynamicRiskScore = Math.max(
      0,
      Math.min(
        100,
        Number(
          liveResult?.risk_score ??
            result?.voice_integrity_risk?.risk_score ??
            0
        )
      )
    );

    const dynamicRiskLevel =
      dynamicRiskScore >= 80
        ? "CRITICAL"
        : dynamicRiskScore >= 60
        ? "HIGH"
        : dynamicRiskScore >= 30
        ? "MODERATE"
        : "LOW";

    const riskColor =
      dynamicRiskScore >= 80
        ? "#d84a5f"
        : dynamicRiskScore >= 60
        ? "#e07f2f"
        : dynamicRiskScore >= 30
        ? "#d69a25"
        : "#07966f";

    const quickActions = [
      {
        title: "Analyze Audio",
        icon: "◎",
        page: "analysis",
      },
      {
        title: "Live Analysis",
        icon: "◉",
        page: "live",
      },
      {
        title: "Evidence Viewer",
        icon: "◇",
        page: "evidence",
      },
      {
        title: "Forensic Reports",
        icon: "▤",
        page: "reports",
      },
    ];

    return (
      <>
        <header className="topbar dashboard-topbar">
          <div>
            <p className="eyebrow">VOICE INTEGRITY SECURITY CENTER</p>
            <h1>SonicT Security Dashboard</h1>
          </div>

        </header>

        <section className="dashboard-hero">
          <div>
            <h2>AI-Powered Voice Integrity Verification</h2>
          </div>

          <div className="hero-actions">
            <button className="cyber-primary" onClick={() => changePage("analysis")}>
              Run New Analysis
            </button>
            <button className="cyber-secondary" onClick={() => changePage("live")}>
              Start Live Monitor
            </button>
          </div>
        </section>

        <section className="panel dashboard-section">
          <div className="panel-heading">
            <div>
              <span className="section-number">QA</span>
              <div>
                <h2>Quick Actions</h2>
              </div>
            </div>
          </div>

          <div className="quick-action-grid">
            {quickActions.map((item) => (
              <button
                type="button"
                className="quick-action-card"
                key={item.page}
                onClick={() => changePage(item.page)}
              >
                <span className="quick-action-icon">{item.icon}</span>
                <strong>{item.title}</strong>
                <span className="quick-action-link">Open</span>
              </button>
            ))}
          </div>
        </section>

        <section className="dashboard-security-grid dashboard-risk-section">
          <article className="dynamic-risk-card">
            <div className="security-card-heading">
              <span className="section-number">DR</span>
              <div><h2>Dynamic Risk Score</h2></div>
            </div>
            <div className="dynamic-risk-body">
              <div className="dynamic-risk-ring" style={{ "--risk-value": `${dynamicRiskScore * 3.6}deg`, "--risk-color": riskColor }}>
                <div><strong>{dynamicRiskScore.toFixed(0)}</strong><span>/100</span></div>
              </div>
              <div className="dynamic-risk-summary">
                <span style={{ color: riskColor }}>{dynamicRiskLevel} RISK</span>
                <strong>{dynamicRiskScore >= 80 ? "Block contact recommended" : dynamicRiskScore >= 60 ? "Independent verification required" : dynamicRiskScore >= 30 ? "Additional verification recommended" : "Normal monitoring"}</strong>
              </div>
            </div>
          </article>

          <article className="dashboard-caller-card">
            <div className="security-card-heading">
              <span className="section-number">CL</span>
              <div><h2>Live Caller</h2></div>
            </div>
            <div className="caller-summary-row"><span>☎</span><div><strong>{callerDetails.phone || "Unknown caller"}</strong>{callerDetails.location && <small>{callerDetails.location}</small>}</div></div>
            <dl className="caller-summary-list">
              <div><dt>Type</dt><dd>{callerDetails.type}</dd></div>
              <div><dt>Known Contact</dt><dd className={callerDetails.knownContact === "Yes" ? "known" : "unknown"}>{callerDetails.knownContact}</dd></div>
              <div><dt>Status</dt><dd>{contactBlocked ? "BLOCKED" : liveMonitoring ? "MONITORING" : "STANDBY"}</dd></div>
            </dl>
            <button type="button" className="dashboard-card-link" onClick={() => changePage("live")}>Open Live Monitoring →</button>
          </article>

          <article className="audit-integrity-card">
            <div className="security-card-heading">
              <span className="section-number">AU</span>
              <div><h2>Audit & Integrity</h2></div>
            </div>
            <div className="integrity-status-list">
              <div><span>✓</span><p><strong>Tamper-evident audit trail</strong></p><em>ACTIVE</em></div>
              <div><span>✓</span><p><strong>SHA-256 evidence fingerprint</strong></p><em>ACTIVE</em></div>
              <div className="not-connected"><span>○</span><p><strong>Blockchain ledger</strong></p><em>NOT CONNECTED</em></div>
            </div>
          </article>
        </section>

        {dynamicRiskScore >= 60 && !riskNoticeDismissed && (
          <section className={`dashboard-risk-alert ${dynamicRiskScore >= 80 ? "critical" : "high"}`} role="alert">
            <div className="risk-alert-symbol">!</div>
            <div>
              <span>VOICEGUARD SECURITY NOTIFICATION</span>
              <h2>{dynamicRiskScore >= 80 ? "Suspicious call detected" : "Elevated impersonation risk detected"}</h2>
              <p>Dynamic Risk Score: <strong>{dynamicRiskScore.toFixed(0)} / 100</strong>. {dynamicRiskScore >= 80 ? "Block this contact and verify the caller through an independent channel." : "Do not share sensitive information until the caller is verified."}</p>
            </div>
            <div className="risk-alert-actions">
              {dynamicRiskScore >= 80 && <button type="button" className="block-contact-button" onClick={() => setContactBlocked(true)} disabled={contactBlocked}>{contactBlocked ? "Contact Blocked" : "Block Contact"}</button>}
              <button type="button" className="dismiss-alert-button" onClick={() => setRiskNoticeDismissed(true)}>Dismiss</button>
            </div>
          </section>
        )}

        <div className="dashboard-two-column">
          <section className="panel">
            <div className="panel-heading">
              <div>
                <span className="section-number">AI</span>
                <div>
                  <h2>Forensic Intelligence Layers</h2>
                </div>
              </div>
            </div>

            <div className="model-status-list">
              {[
                ["F1", "Voice Clone Detection"],
                ["F2", "Spectrogram Analysis"],
                ["F3", "Voice Feature Analysis"],
                ["F4", "Tampering & Localization"],
                ["F5", "Replay Attack Detection"],
              ].map(([code, name]) => (
                <div className="model-status-row" key={code}>
                  <span className="model-code">{code}</span>
                  <div>
                    <strong>{name}</strong>
                  </div>
                  <span className="model-ready">READY</span>
                </div>
              ))}

              <div className="model-status-row experimental-row">
                <span className="model-code">F1B</span>
                <div>
                  <strong>Robust Deepfake Layer</strong>
                </div>
                <span className="model-experimental">EXPERIMENTAL</span>
              </div>
            </div>
          </section>

          <section className="panel">
            <div className="panel-heading">
              <div>
                <span className="section-number">SEC</span>
                <div>
                  <h2>Security Controls</h2>
                </div>
              </div>
            </div>

            <div className="security-control-list">
              <div><span>✓</span><p><strong>API Authentication</strong></p></div>
              <div><span>✓</span><p><strong>Evidence Integrity</strong></p></div>
              <div><span>✓</span><p><strong>Privacy Cleanup</strong></p></div>
              <div><span>✓</span><p><strong>Secondary Verification</strong></p></div>
            </div>
          </section>
        </div>
      </>
    );
  };

  const renderAdminSettingsPage = () => {
    const adminSections = [
      {
        code: "US",
        title: "Users & Investigation Access",
        description: "Control authorized forensic users, roles, teams and sessions.",
        items: [
          ["User Provisioning", "Invite, activate, suspend or remove authorized users.", "12 ACTIVE"],
          ["Forensic Roles", "Admin, Investigator, Analyst, Reviewer and Viewer permissions.", "5 ROLES"],
          ["Investigation Teams", "Organize cybercrime, forensic analysis and review teams.", "3 TEAMS"],
          ["Active Sessions", "Review signed-in devices and force logout when required.", "2 SESSIONS"],
        ],
      },
      {
        code: "SC",
        title: "Security & Compliance",
        description: "Protect accounts, restrict access and preserve traceability.",
        items: [
          ["Single Sign-On", "Optional enterprise identity-provider connection.", "SSO", "sso"],
          ["Two-Factor Authentication", "Require a second factor for privileged accounts.", "2FA", "twoFactor"],
          ["IP Whitelisting", "Restrict access to approved networks or VPN ranges.", "IP", "ipWhitelist"],
          ["Login Protection", "Lock accounts after repeated failed sign-in attempts.", "5 ATTEMPTS"],
        ],
      },
      {
        code: "AI",
        title: "Forensic Models & Live Monitoring",
        description: "Enable validated SonicT branches used by uploaded and live audio analysis.",
        items: [
          ["F1 Voice Clone Detection", "WavLM and attention-based synthetic voice detection.", "F1", "f1Enabled"],
          ["F2 Spectrogram Analysis", "Log-Mel CNN artifact analysis.", "F2", "f2Enabled"],
          ["F3 Voice Feature Analysis", "MFCC and acoustic Random Forest evidence.", "F3", "f3Enabled"],
          ["F4 Tampering Detection", "Temporal boundary and manipulation localization.", "F4", "f4Enabled"],
          ["F5 Replay Detection", "Replay-attack CNN evidence.", "F5", "f5Enabled"],
        ],
      },
      {
        code: "EV",
        title: "Evidence Retention & Privacy",
        description: "Protect uploaded audio, reports and forensic chain of custody.",
        items: [
          ["SHA-256 Fingerprinting", "Generate an integrity fingerprint for original evidence.", "HASH", "shaFingerprinting"],
          ["Automatic Audio Cleanup", "Delete temporary audio after the configured retention period.", "CLEANUP", "automaticCleanup"],
          ["Forensic Report Retention", "Retain reports and metadata for investigation review.", "365 DAYS"],
          ["Secure Evidence Deletion", "Require an authorized, audit-recorded deletion action.", "AUDITED"],
        ],
      },
      {
        code: "IR",
        title: "Incidents & Trusted Verification",
        description: "Configure case creation, escalation and independent identity checks.",
        items: [
          ["Automatic Incident Creation", "Create a case when Dynamic Risk reaches the threshold.", "≥ 60"],
          ["Trusted-Channel OTP", "Verify suspicious callers through an independent channel.", "5 MINUTES"],
          ["Verification Attempts", "Lock verification after repeated invalid OTP submissions.", "5 ATTEMPTS"],
          ["Investigation Workflow", "Open → Analysis → Verification → Review → Resolved.", "ACTIVE"],
        ],
      },
      {
        code: "NT",
        title: "Notifications, Audit & System Health",
        description: "Control operational warnings and monitor security readiness.",
        items: [
          ["High-Risk Notifications", "Warn investigators when calls reach HIGH or CRITICAL risk.", "RISK", "highRiskNotifications"],
          ["System Health Notifications", "Warn when the backend or a forensic model is unavailable.", "HEALTH", "systemHealthNotifications"],
          ["Security Email Alerts", "Send important access, incident and verification alerts.", "EMAIL", "securityEmails"],
          ["Tamper-Evident Audit Logs", "Record configuration, evidence and incident actions.", "365 DAYS"],
        ],
      },
      {
        code: "ADV",
        title: "Advanced System Access",
        description: "Optional administrator-only connectivity controls.",
        items: [
          ["Protected Backend", "Existing FastAPI analysis endpoints remain connected.", "ONLINE"],
          ["Scoped API Tokens", "Issue and revoke restricted service credentials.", "1 TOKEN"],
          ["Incident Webhooks", "Send approved incident events to security systems.", "0 ACTIVE"],
          ["Default Localization", "Apply the organization timezone to reports and audit events.", "ASIA/KOLKATA"],
        ],
      },
    ];

    const toggleAdminPreference = (key) => {
      setAdminPreferences((current) => ({ ...current, [key]: !current[key] }));
      setAdminSaveMessage("");
    };

    const enabledModelCount = ["f1Enabled", "f2Enabled", "f3Enabled", "f4Enabled", "f5Enabled"]
      .filter((key) => adminPreferences[key]).length;

    const updateAdminConfig = (event) => {
      const { name, value } = event.target;
      setAdminConfig((current) => ({ ...current, [name]: value }));
      setAdminSaveMessage("");
    };

    const saveAdminConfig = (event) => {
      event.preventDefault();

      if (Number(adminConfig.highRiskThreshold) >= Number(adminConfig.criticalRiskThreshold)) {
        setAdminSaveMessage("High-risk threshold must be lower than the critical threshold.");
        return;
      }

      if (Number(adminConfig.f4SuspiciousThreshold) >= Number(adminConfig.f4HighThreshold)) {
        setAdminSaveMessage("F4 suspicious threshold must be lower than the F4 high-risk threshold.");
        return;
      }

      setAdminSaveMessage("Admin settings validated and saved for this frontend session.");
    };

    return (
      <>
        <header className="topbar admin-topbar">
          <div>
            <p className="eyebrow">PLATFORM ADMINISTRATION</p>
            <h1>Admin & Settings</h1>
          </div>
          <span className="admin-access-badge">ADMINISTRATOR ACCESS</span>
        </header>

        <section className="admin-summary-grid" aria-label="Administration summary">
          <div><span>Active Users</span><strong>12</strong></div>
          <div><span>Security</span><strong>{adminPreferences.twoFactor ? "ENFORCED" : "OPTIONAL"}</strong></div>
          <div><span>Forensic Models</span><strong>{enabledModelCount} / 5</strong></div>
          <div><span>Audit Retention</span><strong>365 DAYS</strong></div>
        </section>

        <section className="admin-settings-grid">
          {adminSections.map((section) => (
            <article className={`admin-settings-card ${section.code === "ADV" ? "advanced-admin-card" : ""}`} key={section.code}>
              <div className="admin-card-heading">
                <span>{section.code}</span>
                <div><h2>{section.title}</h2></div>
              </div>
              <div className="admin-setting-list">
                {section.items.map(([title, , status, toggleKey]) => (
                  <div className="admin-setting-item" key={title}>
                    <div><strong>{title}</strong></div>
                    {toggleKey ? (
                      <button
                        type="button"
                        className={`admin-toggle ${adminPreferences[toggleKey] ? "enabled" : ""}`}
                        role="switch"
                        aria-checked={adminPreferences[toggleKey]}
                        aria-label={`${title}: ${adminPreferences[toggleKey] ? "enabled" : "disabled"}`}
                        onClick={() => toggleAdminPreference(toggleKey)}
                      >
                        <span></span>{adminPreferences[toggleKey] ? "ON" : "OFF"}
                      </button>
                    ) : <em>{status}</em>}
                  </div>
                ))}
              </div>
            </article>
          ))}
        </section>

        <form className="panel admin-configuration-form" onSubmit={saveAdminConfig}>
          <div className="panel-heading">
            <div><span className="section-number">CFG</span><div><h2>Required Configuration Fields</h2></div></div>
          </div>

          <div className="admin-form-grid">
            <fieldset className="admin-field-group">
              <legend><span>01</span><div>Organization & Localization</div></legend>
              <div className="admin-fields two-column">
                <label>Organization name<input name="organizationName" value={adminConfig.organizationName} onChange={updateAdminConfig} required /></label>
                <label>Security support email<input name="supportEmail" type="email" value={adminConfig.supportEmail} onChange={updateAdminConfig} required /></label>
                <label>Default time zone<select name="timezone" value={adminConfig.timezone} onChange={updateAdminConfig}><option>Asia/Kolkata</option><option>UTC</option><option>Asia/Singapore</option><option>Europe/London</option></select></label>
                <label>Date format<select name="dateFormat" value={adminConfig.dateFormat} onChange={updateAdminConfig}><option>DD/MM/YYYY</option><option>MM/DD/YYYY</option><option>YYYY-MM-DD</option></select></label>
              </div>
            </fieldset>

            <fieldset className="admin-field-group">
              <legend><span>02</span><div>Access & Session Policy</div></legend>
              <div className="admin-fields two-column">
                <label>Default user role<select name="defaultRole" value={adminConfig.defaultRole} onChange={updateAdminConfig}><option>Investigator</option><option>Analyst</option><option>Reviewer</option><option>Viewer</option><option>Guest</option></select></label>
                <label>Session timeout (minutes)<input name="sessionTimeoutMinutes" type="number" min="5" max="480" value={adminConfig.sessionTimeoutMinutes} onChange={updateAdminConfig} required /></label>
                <label>Maximum failed logins<input name="maxFailedLogins" type="number" min="3" max="10" value={adminConfig.maxFailedLogins} onChange={updateAdminConfig} required /></label>
                <label>Password expiry (days)<input name="passwordExpiryDays" type="number" min="30" max="365" value={adminConfig.passwordExpiryDays} onChange={updateAdminConfig} required /></label>
              </div>
            </fieldset>

            <fieldset className="admin-field-group">
              <legend><span>03</span><div>Model & Live Risk Controls</div></legend>
              <div className="admin-fields three-column">
                <label>High risk begins at (0–100)<input name="highRiskThreshold" type="number" min="1" max="99" value={adminConfig.highRiskThreshold} onChange={updateAdminConfig} required /></label>
                <label>Critical risk begins at (0–100)<input name="criticalRiskThreshold" type="number" min="2" max="100" value={adminConfig.criticalRiskThreshold} onChange={updateAdminConfig} required /></label>
                <label>Live chunk duration (seconds)<input name="liveChunkSeconds" type="number" min="3" max="30" value={adminConfig.liveChunkSeconds} onChange={updateAdminConfig} required /></label>
                <label>F4 suspicious threshold<input name="f4SuspiciousThreshold" type="number" min="0" max="1" step="0.01" value={adminConfig.f4SuspiciousThreshold} onChange={updateAdminConfig} required /></label>
                <label>F4 high-risk threshold<input name="f4HighThreshold" type="number" min="0" max="1" step="0.01" value={adminConfig.f4HighThreshold} onChange={updateAdminConfig} required /></label>
                <label>Experimental F1B<input value="Separate from validated fusion" readOnly /></label>
              </div>
            </fieldset>

            <fieldset className="admin-field-group">
              <legend><span>04</span><div>Evidence & Privacy</div></legend>
              <div className="admin-fields three-column">
                <label>Maximum upload size (MB)<input name="maxUploadMb" type="number" min="10" max="2048" value={adminConfig.maxUploadMb} onChange={updateAdminConfig} required /></label>
                <label>Temporary audio retention (hours)<input name="temporaryRetentionHours" type="number" min="1" max="168" value={adminConfig.temporaryRetentionHours} onChange={updateAdminConfig} required /></label>
                <label>Report retention (days)<input name="reportRetentionDays" type="number" min="30" max="3650" value={adminConfig.reportRetentionDays} onChange={updateAdminConfig} required /></label>
              </div>
              <div className="admin-required-status"><span>✓ SHA-256 fingerprinting enabled</span><span>✓ Temporary audio auto-cleanup enabled</span><span>✓ Deletion actions written to audit log</span></div>
            </fieldset>

            <fieldset className="admin-field-group">
              <legend><span>05</span><div>Incident & Trusted Verification</div></legend>
              <div className="admin-fields four-column">
                <label>Auto-create incident at risk score<input name="autoIncidentThreshold" type="number" min="1" max="100" value={adminConfig.autoIncidentThreshold} onChange={updateAdminConfig} required /></label>
                <label>OTP expiry (minutes)<input name="otpExpiryMinutes" type="number" min="1" max="15" value={adminConfig.otpExpiryMinutes} onChange={updateAdminConfig} required /></label>
                <label>Maximum OTP attempts<input name="otpMaxAttempts" type="number" min="3" max="10" value={adminConfig.otpMaxAttempts} onChange={updateAdminConfig} required /></label>
                <label>OTP resend cooldown (seconds)<input name="otpResendCooldown" type="number" min="30" max="300" value={adminConfig.otpResendCooldown} onChange={updateAdminConfig} required /></label>
              </div>
            </fieldset>
          </div>

          <div className="admin-form-footer">
            {adminSaveMessage && <p className={adminSaveMessage.includes("validated") ? "success" : ""} role="status">{adminSaveMessage}</p>}
            <button type="submit" className="cyber-primary">Save Admin Settings</button>
          </div>
        </form>
      </>
    );
  };


  const renderInformationPage = (page) => {
    const pages = {
      alerts: {
        eyebrow: "SECURITY OPERATIONS",
        title: "Alerts & Incidents",
        description: "Review operational warnings generated by SonicT forensic evidence.",
        code: "AL",
        heading: "Current Security Decision",
      },
      speaker: {
        eyebrow: "IDENTITY VERIFICATION",
        title: "Speaker Verification",
        description: "Workspace for trusted-speaker enrollment and cross-session consistency checks.",
        code: "SP",
        heading: "Speaker Identity Verification",
      },
      analytics: {
        eyebrow: "FORENSIC INTELLIGENCE",
        title: "Analytics",
        description: "Security-oriented overview of SonicT analysis activity and model evidence.",
        code: "AN",
        heading: "Analysis Intelligence",
      },
      integrations: {
        eyebrow: "PLATFORM CONNECTIVITY",
        title: "Integrations",
        description: "Integration surface for enterprise, telephony and application workflows.",
        code: "API",
        heading: "SonicT Integration Layer",
      },
      settings: {
        eyebrow: "PLATFORM ADMINISTRATION",
        title: "Admin & Settings",
        description: "Security, model, privacy and deployment configuration overview.",
        code: "CFG",
        heading: "System Configuration",
      },
    };

    const config = pages[page] || pages.analytics;

    return (
      <>
        <header className="topbar">
          <div>
            <p className="eyebrow">{config.eyebrow}</p>
            <h1>{config.title}</h1>
            <p className="top-description">{config.description}</p>
          </div>
        </header>

        <section className="panel">
          <div className="panel-heading">
            <div>
              <span className="section-number">{config.code}</span>
              <div>
                <h2>{config.heading}</h2>
                <p>This page is connected to the existing SonicT frontend structure without changing backend endpoints.</p>
              </div>
            </div>
          </div>

          {page === "alerts" && (
            result?.security_alert ? (
              <div className={result.security_alert.alert ? "error-message" : "safe-window-message"}>
                <span>{result.security_alert.alert ? "!" : "✓"}</span>
                <div>
                  <h3>{result.security_alert.message || "Security assessment"}</h3>
                  <p><strong>Action:</strong> {result.security_alert.action || "Continue normal verification."}</p>
                </div>
              </div>
            ) : (
              <div className="empty-state"><span>△</span><h3>No active incident</h3><p>Run an audio analysis to generate a security decision.</p><button className="cyber-primary" onClick={() => changePage("analysis")}>Analyze Audio</button></div>
            )
          )}

          {page === "speaker" && (
            <div className="empty-state"><span>◎</span><h3>Speaker verification workspace</h3><p>Your backend speaker-verification capability can be connected here without changing the existing analysis pipeline. Enrollment thresholds should be treated as configurable/experimental until calibrated.</p></div>
          )}

          {page === "analytics" && (
            <div className="dashboard-stat-grid compact-stats">
              <div className="dashboard-stat-card"><span>Saved Reports</span><strong>{reports.length}</strong><small>Loaded in this browser session</small></div>
              <div className="dashboard-stat-card"><span>Live Chunks</span><strong>{liveChunkCount}</strong><small>Current live session</small></div>
              <div className="dashboard-stat-card"><span>Threat Sequences</span><strong>{liveSuspiciousSegments.length}</strong><small>Current live session</small></div>
              <div className="dashboard-stat-card"><span>Current Classification</span><strong>{String(result?.classification || "-").toUpperCase()}</strong><small>Latest full analysis</small></div>
            </div>
          )}

          {page === "integrations" && (
            <div className="integration-grid">
              <div className="integration-card"><span>REST</span><strong>FastAPI Backend</strong><small>{API_BASE}</small></div>
              <div className="integration-card"><span>AUTH</span><strong>API Key</strong><small>X-API-Key request authentication</small></div>
              <div className="integration-card"><span>LIVE</span><strong>Microphone Chunks</strong><small>/analyze-live-chunk</small></div>
              <div className="integration-card"><span>PDF</span><strong>Forensic Reporting</strong><small>/download-forensic-report</small></div>
            </div>
          )}

          {page === "settings" && (
            <div className="settings-grid">
              <div className="setting-row"><div><strong>Voice Integrity Risk</strong><small>LOW &lt; 30 • MEDIUM 30–&lt;60 • HIGH 60–&lt;80 • CRITICAL ≥ 80</small></div><span>ACTIVE</span></div>
              <div className="setting-row"><div><strong>F4 Suspicious Threshold</strong><small>Temporal suspicious-window evidence threshold</small></div><span>0.50</span></div>
              <div className="setting-row"><div><strong>F4 High-Risk Threshold</strong><small>Temporal high-risk evidence threshold</small></div><span>0.70</span></div>
              <div className="setting-row"><div><strong>Experimental F1B</strong><small>Displayed separately; not part of validated Extra Trees fusion</small></div><span>SEPARATE</span></div>
            </div>
          )}
        </section>
      </>
    );
  };

const renderAlertsPage = () => {
  const incident = selectedIncident;
  const riskLevel = String(incident?.risk_level || "").toUpperCase();
  const verificationStatus = String(
    incident?.verification_status || "NOT_STARTED"
  ).toUpperCase();
  const caseStatus = String(incident?.case_status || "OPEN").toUpperCase();

  const riskClass =
    riskLevel === "CRITICAL" || riskLevel === "HIGH"
      ? "critical"
      : riskLevel === "MEDIUM"
      ? "warning"
      : "normal";

  return (
    <>
      <header className="topbar">
        <div>
          <p className="eyebrow">SECURITY OPERATIONS</p>
          <h1>Alerts & Incidents</h1>
        </div>

        <button
          type="button"
          className="secondary-action-button"
          onClick={loadIncidents}
          disabled={incidentsLoading}
        >
          {incidentsLoading ? "Refreshing..." : "Refresh Incidents"}
        </button>
      </header>

      {incidentsError && (
        <div className="error-message">
          <span>!</span>
          <div>
            <h3>Incident service error</h3>
            <p>{incidentsError}</p>
          </div>
        </div>
      )}

      {!incident && !incidentsLoading ? (
        <section className="panel">
          <div className="empty-state">
            <div className="empty-state-icon">△</div>
            <h3>No security incidents</h3>
            <p>
              HIGH/CRITICAL risk or conflicting forensic evidence will create
              an incident automatically after audio analysis.
            </p>
            <button
              type="button"
              className="action-button"
              onClick={() => changePage("analysis")}
            >
              Analyze Audio
            </button>
          </div>
        </section>
      ) : incident ? (
        <>
          <div className={`incident-banner ${riskClass}`}>
            <span className="incident-alert-symbol">!</span>
            <div>
              <h3>{incident.alert_message || "Security incident detected"}</h3>
            </div>
            <button
              type="button"
              className="incident-report-button"
              onClick={downloadIncidentReport}
              disabled={incidentReportLoading}
            >
              {incidentReportLoading ? "Preparing..." : "Download Evidence Report"}
            </button>
          </div>

          <section className="incident-summary-grid">
            <div className="incident-metric-card">
              <span>FORENSIC RESULT</span>
              <strong>{String(incident.classification || "—").toUpperCase()}</strong>
              <strong className="incident-metric-secondary">
                {Number(incident.confidence || 0) > 0
                  ? `${(Number(incident.confidence) * 100).toFixed(2)}% CONFIDENCE`
                  : "CONFIDENCE UNAVAILABLE"}
              </strong>
            </div>

            <div className="incident-metric-card">
              <span>VOICE INTEGRITY RISK</span>
              <strong className={`incident-risk-text risk-text-${riskLevel.toLowerCase()}`}>
                {riskLevel || "—"}
              </strong>
              <strong className="incident-metric-secondary">{Number(incident.risk_score || 0).toFixed(2)} / 100</strong>
            </div>

            <div className="incident-metric-card">
              <span>CASE STATUS</span>
              <strong>{caseStatus.replaceAll("_", " ")}</strong>
            </div>

            <div className="incident-metric-card">
              <span>IDENTITY VERIFICATION</span>
              <strong className={`verification-text verification-${verificationStatus.toLowerCase()}`}>
                {verificationStatus.replaceAll("_", " ")}
              </strong>
            </div>
          </section>

          <div className="incident-workspace-grid">
            <section className="panel">
              <div className="panel-heading">
                <div>
                  <span className="section-number">AL</span>
                  <div>
                    <h2>Incident Evidence</h2>
                  </div>
                </div>
              </div>

              <div className="incident-detail-list">
                <div>
                  <span>Recommended action</span>
                  <strong>
                    {incident.recommended_action || "Independent verification recommended."}
                  </strong>
                </div>

                <div>
                  <span>Verification required</span>
                  <strong>{incident.verification_required ? "YES" : "NO"}</strong>
                </div>

                <div>
                  <span>Evidence warning</span>
                  <strong>{incident.evidence_warning ? "YES" : "NO"}</strong>
                </div>

                <div>
                  <span>Created</span>
                  <strong>
                    {incident.created_at
                      ? new Date(incident.created_at).toLocaleString()
                      : "—"}
                  </strong>
                </div>
              </div>

              <div className="evidence-flag-area">
                <span className="incident-section-label">EVIDENCE FLAGS</span>
                <div className="evidence-flag-list">
                  {Array.isArray(incident.evidence_flags) &&
                  incident.evidence_flags.length ? (
                    incident.evidence_flags.map((flag) => (
                      <span className="evidence-flag-chip" key={flag}>
                        {String(flag).replaceAll("_", " ")}
                      </span>
                    ))
                  ) : (
                    <span className="evidence-flag-chip neutral">
                      No additional evidence flags
                    </span>
                  )}
                </div>
              </div>

              <div className="case-management-area">
                <span className="incident-section-label">CASE MANAGEMENT</span>

                <div className="case-action-row">
                  {caseStatus === "OPEN" && (
                    <button type="button" className="secondary-action-button" onClick={() => updateIncidentLifecycle("ACKNOWLEDGE")} disabled={incidentActionLoading}>Acknowledge</button>
                  )}
                  {caseStatus === "ACKNOWLEDGED" && (
                    <button type="button" className="secondary-action-button" onClick={() => updateIncidentLifecycle("START_INVESTIGATION")} disabled={incidentActionLoading}>Start Investigation</button>
                  )}
                  {caseStatus === "RESOLVED" && (
                    <button type="button" className="secondary-action-button" onClick={() => updateIncidentLifecycle("REOPEN")} disabled={incidentActionLoading}>Reopen Incident</button>
                  )}
                </div>

                <label htmlFor="investigator-notes">Investigator notes</label>
                <textarea
                  id="investigator-notes"
                  value={investigatorNotes}
                  onChange={(event) => setInvestigatorNotes(event.target.value)}
                  placeholder="Record observations, checks and decisions..."
                  rows={4}
                />

                <button type="button" className="secondary-action-button" onClick={() => updateIncidentLifecycle("ADD_NOTE")} disabled={incidentActionLoading || !investigatorNotes.trim()}>
                  Save Notes
                </button>

                {caseStatus !== "RESOLVED" && (
                  <div className="case-resolution-row">
                    <select value={resolutionChoice} onChange={(event) => setResolutionChoice(event.target.value)}>
                      <option value="INCONCLUSIVE">Inconclusive</option>
                      <option value="CONFIRMED_THREAT">Confirmed Threat</option>
                      <option value="FALSE_POSITIVE">False Positive</option>
                      <option value="VERIFIED_LEGITIMATE">Verified Legitimate</option>
                    </select>
                    <button type="button" className="action-button" onClick={() => updateIncidentLifecycle("RESOLVE")} disabled={incidentActionLoading}>
                      {incidentActionLoading ? "Updating..." : "Resolve Incident"}
                    </button>
                  </div>
                )}

                {incident.resolution && (
                  <div className="case-resolution-summary">
                    Resolution: <strong>{String(incident.resolution).replaceAll("_", " ")}</strong>
                  </div>
                )}
              </div>
            </section>

            <section className="panel verification-panel">
              <div className="panel-heading">
                <div>
                  <span className="section-number">ID</span>
                  <div>
                    <h2>Trusted-Channel Verification</h2>
                  </div>
                </div>
              </div>

              <div className={`verification-status-box verification-box-${verificationStatus.toLowerCase()}`}>
                <span>VERIFICATION STATUS</span>
                <strong>{verificationStatus.replaceAll("_", " ")}</strong>
              </div>

              {verificationStatus === "VERIFIED" ? (
                <div className="verification-success-card">
                  <span className="verification-check">✓</span>
                  <div>
                    <h3>Trusted channel verified</h3>
                    <p>
                      Identity verification succeeded. The forensic classification
                      remains <strong>{String(incident.classification || "").toUpperCase()}</strong>.
                    </p>
                    {incident.verified_at && (
                      <small>
                        Verified: {new Date(incident.verified_at).toLocaleString()}
                      </small>
                    )}
                  </div>
                </div>
              ) : (
                <>
                  <button
                    type="button"
                    className="action-button verification-start-button"
                    onClick={requestTrustedVerification}
                    disabled={
                      verificationLoading ||
                      !incident.verification_required
                    }
                  >
                    {verificationLoading
                      ? "Processing..."
                      : verificationStatus === "PENDING"
                      ? "Generate New OTP"
                      : "Start Trusted Verification"}
                  </button>

                  {developmentOtp && (
                    <div className="development-otp-card">
                      <span>DEVELOPMENT OTP</span>
                      <strong>{developmentOtp}</strong>
                      <p>
                        Local SIH prototype only. Production must deliver this
                        through a pre-registered SMS/email channel.
                      </p>
                    </div>
                  )}

                  {verificationStatus === "PENDING" && (
                    <div className="otp-form">
                      <label htmlFor="incident-otp">
                        Enter 6-digit trusted-channel OTP
                      </label>

                      <input
                        id="incident-otp"
                        type="text"
                        inputMode="numeric"
                        maxLength={6}
                        value={otpInput}
                        onChange={(event) =>
                          setOtpInput(
                            event.target.value.replace(/\D/g, "").slice(0, 6)
                          )
                        }
                        placeholder="000000"
                      />

                      <button
                        type="button"
                        className="action-button"
                        onClick={verifyTrustedOtp}
                        disabled={verificationLoading || otpInput.length !== 6}
                      >
                        {verificationLoading ? "Verifying..." : "Verify Identity"}
                      </button>
                    </div>
                  )}
                </>
              )}

              {verificationMessage && (
                <div className="verification-message success">
                  {verificationMessage}
                </div>
              )}

              {verificationError && (
                <div className="verification-message error">
                  {verificationError}
                </div>
              )}

            </section>
          </div>

          <section className="panel incident-audit-panel">
            <div className="panel-heading">
              <div>
                <span className="section-number">AU</span>
                <div>
                  <h2>Incident Audit Trail</h2>
                </div>
              </div>
              <button
                type="button"
                className="secondary-action-button"
                onClick={() => loadIncidentAudit(incident.incident_uuid)}
                disabled={incidentAuditLoading}
              >
                {incidentAuditLoading ? "Loading..." : "Refresh Audit"}
              </button>
            </div>

            <div className="incident-audit-list">
              {incidentAudit.length ? (
                incidentAudit.map((event) => (
                  <div className="incident-audit-item" key={event.id}>
                    <span className="incident-audit-dot" />
                    <div>
                      <strong>{String(event.event_type || "EVENT").replaceAll("_", " ")}</strong>
                      <p>{event.message || "Operational event recorded."}</p>
                      <small>
                        {event.created_at
                          ? new Date(event.created_at).toLocaleString()
                          : "Time unavailable"}
                        {event.event_status ? ` • ${event.event_status}` : ""}
                      </small>
                    </div>
                  </div>
                ))
              ) : (
                <div className="empty-state compact-empty-state">
                  <p>{incidentAuditLoading ? "Loading audit history..." : "No audit events recorded yet."}</p>
                </div>
              )}
            </div>
          </section>

          <section className="panel recent-incidents-panel">
            <div className="panel-heading">
              <div>
                <span className="section-number">HI</span>
                <div>
                  <h2>Recent Incidents</h2>
                  <p>Select an incident to review or verify.</p>
                </div>
              </div>
              <span className="window-count">{incidents.length} INCIDENTS</span>
            </div>

            <div className="incident-table-wrap">
              <table className="incident-table">
                <thead>
                  <tr>
                    <th>File</th>
                    <th>Classification</th>
                    <th>Risk</th>
                    <th>Verification</th>
                    <th>Status</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {incidents.map((item) => (
                    <tr
                      key={item.incident_uuid}
                      className={
                        item.incident_uuid === incident.incident_uuid
                          ? "selected-incident-row"
                          : ""
                      }
                    >
                      <td>{item.filename || "Unknown audio"}</td>
                      <td>{String(item.classification || "—").toUpperCase()}</td>
                      <td>
                        <span className={`incident-risk-pill risk-pill-${String(item.risk_level || "").toLowerCase()}`}>
                          {String(item.risk_level || "—").toUpperCase()}
                          {Number.isFinite(Number(item.risk_score))
                            ? ` · ${Number(item.risk_score).toFixed(1)}`
                            : ""}
                        </span>
                      </td>
                      <td>
                        {String(item.verification_status || "NOT_STARTED").replaceAll("_", " ")}
                      </td>
                      <td>
                        {String(item.case_status || "OPEN").replaceAll("_", " ")}
                      </td>
                      <td>
                        <button
                          type="button"
                          className="row-action"
                          onClick={() => {
                            setSelectedIncident(item);
                            setOtpInput("");
                            setDevelopmentOtp("");
                            setVerificationError("");
                            setVerificationMessage("");
                            loadIncidentDetails(item.incident_uuid);
                          }}
                        >
                          Open
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="verification-guidance">
            <div><span>LOW</span><p>No security alert required.</p></div>
            <div><span>MEDIUM</span><p>Verify identity before sensitive action.</p></div>
            <div><span>HIGH</span><p>Use independent callback or MFA.</p></div>
            <div><span>CRITICAL</span><p>Do not authorize until independently verified.</p></div>
          </section>
        </>
      ) : null}
    </>
  );
};

  /* ======================================================
     MAIN
  ====================================================== */

  const updateAuthField = (event) => {
    const { name, value, checked, type } = event.target;
    setAuthForm((current) => ({
      ...current,
      [name]: type === "checkbox" ? checked : value,
    }));
    setAuthError("");
  };

  const submitAuth = async (event) => {
    event.preventDefault();
    const email = authForm.email.trim().toLowerCase();

    if (authMode === "signup" && authForm.name.trim().length < 2) {
      setAuthError("Please enter your full name.");
      return;
    }
    if (!/^\S+@\S+\.\S+$/.test(email)) {
      setAuthError("Please enter a valid email address.");
      return;
    }
    if (authForm.password.length < 8) {
      setAuthError("Password must contain at least 8 characters.");
      return;
    }
    if (authMode === "signup" && authForm.password !== authForm.confirmPassword) {
      setAuthError("Passwords do not match.");
      return;
    }

    setAuthLoading(true);
    setAuthError("");
    try {
      const response = await fetch(`${API_BASE}/auth/${authMode === "signup" ? "signup" : "login"}`, {
        method: "POST",
        headers: {
  "Content-Type": "application/json",
  "ngrok-skip-browser-warning": "true",
},
        body: JSON.stringify({
          name: authForm.name.trim(),
          email,
          password: authForm.password,
        }),
      });
      const data = await response.json().catch(() => null);
      if (!response.ok) {
        throw new Error(data?.detail || "Authentication failed. Please try again.");
      }

      const storage = authForm.remember ? localStorage : sessionStorage;
      localStorage.removeItem("sonict_access_token");
      sessionStorage.removeItem("sonict_access_token");
      storage.setItem("sonict_access_token", data.access_token);
      storage.setItem("sonict_user", JSON.stringify(data.user));
      setAuthToken(data.access_token);
    } catch (authRequestError) {
      setAuthError(authRequestError.message || "Unable to connect to the SonicT backend.");
    } finally {
      setAuthLoading(false);
    }
  };

  const enterDemoMode = () => {
    sessionStorage.setItem("sonict_demo_mode", "true");
    sessionStorage.setItem("sonict_demo_sample", "cloned");
    sessionStorage.removeItem("sonict_access_token");
    sessionStorage.removeItem("sonict_user");
    setDemoMode(true);
    setDemoSampleKey("cloned");
    setAuthToken(DEMO_TOKEN);
    setResult(DEMO_RESULT);
    setReports(DEMO_REPORTS);
    setIncidents(DEMO_INCIDENTS);
    setSelectedIncident(DEMO_INCIDENTS[0]);
    setActivePage("dashboard");
    setAuthError("");
  };

  const selectDemoSample = (sampleKey) => {
    const sample = DEMO_SAMPLES[sampleKey] || DEMO_SAMPLES.cloned;
    const sampleReports = createDemoReport(sample.result);
    const sampleIncidents = createDemoIncidents(sample.result);
    sessionStorage.setItem("sonict_demo_sample", sampleKey);
    setDemoSampleKey(sampleKey);
    setResult(sample.result);
    setReports(sampleReports);
    setIncidents(sampleIncidents);
    setSelectedIncident(sampleIncidents[0] || null);
    setLiveResult(null);
    setError("");
    setChunkError("");
    setMimicryError("");
  };

  const logout = () => {
    localStorage.removeItem("sonict_access_token");
    localStorage.removeItem("sonict_user");
    sessionStorage.removeItem("sonict_access_token");
    sessionStorage.removeItem("sonict_user");
    sessionStorage.removeItem("sonict_demo_mode");
    sessionStorage.removeItem("sonict_demo_sample");
    setDemoMode(false);
    setAuthToken("");
    setResult(null);
    setReports([]);
    setIncidents([]);
    setSelectedIncident(null);
    setAuthForm((current) => ({ ...current, password: "", confirmPassword: "" }));
  };

  if (!authToken) {
    return (
      <main className="auth-page">
        <section className="auth-visual" aria-label="SonicT security overview">
          <div className="auth-brand">
            <div className="auth-brand-icon">≋</div>
            <div><strong>SonicT</strong><span>Audio Forensics</span></div>
          </div>
          <div className="auth-visual-copy">
            <span className="auth-eyebrow">SECURE FORENSIC WORKSPACE</span>
            <h1>Protect Every Voice Investigation.</h1>
            <p>Access explainable audio analysis, incident evidence and trusted-channel verification from one protected workspace.</p>
            <div className="auth-trust-list">
              <div><span>✓</span><p><strong>Controlled access</strong>Authentication-ready investigation workspace</p></div>
              <div><span>✓</span><p><strong>Evidence integrity</strong>SHA-256 fingerprint and audit support</p></div>
              <div><span>✓</span><p><strong>Privacy first</strong>Temporary audio processing and cleanup</p></div>
            </div>
          </div>
          <p className="auth-visual-foot">SonicT Security Center · Authorized users only</p>
        </section>

        <section className="auth-form-side">
          <div className="auth-card">
            <div className="auth-mobile-brand"><span>≋</span><strong>SonicT</strong></div>
            <span className="auth-eyebrow">IDENTITY ACCESS</span>
            <h2>{authMode === "signin" ? "Welcome back" : "Create your account"}</h2>
            <p className="auth-intro">
              {authMode === "signin"
                ? "Sign in to continue to your forensic dashboard."
                : "Register to access the SonicT security workspace."}
            </p>

            <div className="auth-tabs" role="tablist" aria-label="Authentication mode">
              <button type="button" className={authMode === "signin" ? "active" : ""} onClick={() => { setAuthMode("signin"); setAuthError(""); }}>Sign in</button>
              <button type="button" className={authMode === "signup" ? "active" : ""} onClick={() => { setAuthMode("signup"); setAuthError(""); }}>Sign up</button>
            </div>

            <form className="auth-form" onSubmit={submitAuth}>
              {authMode === "signup" && (
                <label>Full name<input name="name" value={authForm.name} onChange={updateAuthField} placeholder="Enter your full name" autoComplete="name" /></label>
              )}
              <label>Email address<input name="email" type="email" value={authForm.email} onChange={updateAuthField} placeholder="name@organization.com" autoComplete="email" /></label>
              <label>Password
                <span className="password-field">
                  <input name="password" type={showPassword ? "text" : "password"} value={authForm.password} onChange={updateAuthField} placeholder="Minimum 8 characters" autoComplete={authMode === "signin" ? "current-password" : "new-password"} />
                  <button type="button" onClick={() => setShowPassword((visible) => !visible)} aria-label={showPassword ? "Hide password" : "Show password"}>{showPassword ? "Hide" : "Show"}</button>
                </span>
              </label>
              {authMode === "signup" && (
                <label>Confirm password<input name="confirmPassword" type="password" value={authForm.confirmPassword} onChange={updateAuthField} placeholder="Enter password again" autoComplete="new-password" /></label>
              )}

              {authMode === "signin" && (
                <div className="auth-options">
                  <label className="remember-check"><input name="remember" type="checkbox" checked={authForm.remember} onChange={updateAuthField} /> Remember me</label>
                  <button type="button" className="forgot-link">Forgot password?</button>
                </div>
              )}

              {authError && <div className="auth-error" role="alert">! {authError}</div>}
              <button className="auth-submit" type="submit" disabled={authLoading}>{authLoading ? "Securing access..." : authMode === "signin" ? "Sign in securely" : "Create secure account"}<span>{authLoading ? "" : "→"}</span></button>
            </form>

            <div className="demo-entry">
              <div className="demo-divider"><span>or</span></div>
              <button type="button" className="demo-entry-button" onClick={enterDemoMode}>
                <span className="demo-entry-icon">▶</span>
                <span><strong>View Interactive Demo</strong><small>No login or backend required</small></span>
                <span aria-hidden="true">→</span>
              </button>
              <p>Uses a pre-analysed prototype sample. Results are for demonstration only.</p>
            </div>

            <p className="auth-security-note"><span>⌾</span> Protected access · Hashed credentials · Signed session token</p>
          </div>
        </section>
      </main>
    );
  }

  return (

    <div className={`app ${demoMode ? "demo-mode-active" : ""}`}>

      {demoMode && (
        <div className="demo-mode-banner" role="status">
          <span><strong>DEMO MODE</strong> · Pre-analysed prototype sample · No backend connection required</span>
          <button type="button" onClick={logout}>Exit demo</button>
        </div>
      )}

      <aside className="sidebar">

        <div className="brand">

          <div className="brand-icon">
            <span className="wave-mini">
              ≋
            </span>
          </div>


          <div>

            <h2>
              SonicT
            </h2>

            <span>
              Audio Forensics
            </span>

          </div>

        </div>


        <nav className="nav-menu">
          {[
            ["dashboard", "⌂", "Dashboard"],
            ["analysis", "◎", "Analyze Audio"],
            ["live", "◉", "Live Analysis"],
            ["alerts", "△", "Alerts & Incidents"],
            ["speaker", "◌", "Speaker Verification"],
            ["evidence", "◇", "Evidence Viewer"],
            ["reports", "▤", "Forensic Reports"],
            ["analytics", "▥", "Analytics"],
            ["status", "◈", "System Status"],
            ["settings", "⚙", "Admin & Settings"],
          ].map(([page, icon, label]) => (
            <button
              key={page}
              type="button"
              className={`nav-item ${activePage === page ? "active" : ""}`}
              onClick={() => changePage(page)}
            >
              <span>{icon}</span>
              {label}
            </button>
          ))}
        </nav>


        <div className="sidebar-status">

          <div className="status-dot"></div>


          <div>

            <strong>
              SonicT Engine
            </strong>

            <span>
              {demoMode ? "Offline sample · backend not connected" : "5 forensic models + fusion + mimicry"}
            </span>

          </div>

        </div>

        <button type="button" className="sidebar-logout" onClick={logout}>
          <span>↪</span> {demoMode ? "Exit demo" : "Sign out"}
        </button>

      </aside>


      <main className="main-content">

        {demoMode && (
          <section className="demo-sample-panel" aria-label="Pre-analysed demonstration samples">
            <div className="demo-sample-heading">
              <div>
                <span>OFFLINE TEST CASES</span>
                <strong>Select a pre-analysed audio sample</strong>
              </div>
              <small>Selection updates the dashboard, evidence, alerts and report.</small>
            </div>
            <div className="demo-sample-options">
              {Object.entries(DEMO_SAMPLES).map(([sampleKey, sample]) => (
                <button
                  key={sampleKey}
                  type="button"
                  className={demoSampleKey === sampleKey ? "active" : ""}
                  onClick={() => selectDemoSample(sampleKey)}
                >
                  <span className={`demo-sample-dot ${sampleKey}`} />
                  <span><strong>{sample.label}</strong><small>{sample.description}</small></span>
                  <b>{sample.result.voice_integrity_risk.risk_score}/100</b>
                </button>
              ))}
            </div>
            <p>These are stored prototype results. Uploading and analysing a new audio file requires the live SonicT API.</p>
          </section>
        )}

        {activePage === "dashboard" && renderDashboardPage()}

        {activePage === "analysis" && renderAnalysisPage()}

        {activePage === "live" && renderLiveMonitoringPage()}

        {activePage === "alerts" && renderAlertsPage()}

        {activePage === "speaker" && renderInformationPage("speaker")}

        {activePage === "evidence" && renderForensicsPage()}

        {activePage === "reports" && renderReportsPage()}

        {activePage === "analytics" && renderInformationPage("analytics")}

        {activePage === "status" && renderModelStatusPage()}

        {activePage === "settings" && renderAdminSettingsPage()}

      </main>

    </div>
  );
}


export default App;
