import sqlite3
import json
from datetime import datetime

DB_NAME = "sonict.db"


def get_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    # Existing SonicT analysis table - preserved.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            created_at TEXT,
            classification TEXT,
            confidence REAL,
            genuine_probability REAL,
            deepfake_probability REAL,
            tampered_probability REAL,
            replay_probability REAL,
            voice_clone_probability REAL,
            spectrogram_probability REAL,
            voice_feature_probability REAL,
            replay_feature_probability REAL,
            f4_max REAL,
            f4_mean REAL,
            f4_median REAL,
            f4_std REAL,
            f4_suspicious_ratio REAL,
            f4_high_ratio REAL,
            suspicious_windows TEXT
        )
    """)

    # New operational incident table.
    # This does not modify or delete existing analysis records.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            incident_uuid TEXT UNIQUE NOT NULL,
            analysis_id INTEGER,
            filename TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            classification TEXT,
            confidence REAL,
            risk_score REAL,
            risk_level TEXT,
            alert_message TEXT,
            recommended_action TEXT,
            evidence_warning INTEGER DEFAULT 0,
            evidence_flags TEXT,
            incident_status TEXT DEFAULT 'OPEN',
            verification_required INTEGER DEFAULT 1,
            verification_status TEXT DEFAULT 'NOT_STARTED',
            verification_method TEXT,
            verified_at TEXT,
            otp_hash TEXT,
            otp_expires_at TEXT,
            otp_attempts INTEGER DEFAULT 0,
            FOREIGN KEY (analysis_id) REFERENCES analyses(id)
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_incidents_created_at
        ON incidents(created_at)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_incidents_status
        ON incidents(incident_status)
    """)

    # Safe migration for databases created before case management.
    cursor.execute("PRAGMA table_info(incidents)")
    incident_columns = {row[1] for row in cursor.fetchall()}
    lifecycle_columns = {
        "case_status": "TEXT DEFAULT 'OPEN'",
        "investigator_notes": "TEXT",
        "resolution": "TEXT",
        "acknowledged_at": "TEXT",
        "resolved_at": "TEXT",
    }
    for column_name, column_definition in lifecycle_columns.items():
        if column_name not in incident_columns:
            cursor.execute(
                f"ALTER TABLE incidents ADD COLUMN {column_name} {column_definition}"
            )

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS incident_audit_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            incident_uuid TEXT NOT NULL,
            event_type TEXT NOT NULL,
            event_status TEXT,
            message TEXT,
            event_metadata TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (incident_uuid)
                REFERENCES incidents(incident_uuid)
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_incident_audit_uuid
        ON incident_audit_events(incident_uuid, id)
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'INVESTIGATOR',
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            last_login_at TEXT
        )
    """)

    cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email)")

    # Recent live-client results used to synchronize the Android APK with the
    # existing SonicT web dashboard. Only compact JSON results are stored;
    # recorded audio is never retained here.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mobile_analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            received_at TEXT NOT NULL,
            client_type TEXT NOT NULL DEFAULT 'android',
            device_id TEXT,
            result_json TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_mobile_results_client_id
        ON mobile_analysis_results(client_type, id DESC)
    """)

    conn.commit()
    conn.close()


def save_mobile_analysis(result, client_type="android", device_id=None):
    """Store a compact live-analysis result for cross-device UI sync."""
    safe_client = str(client_type or "android").strip().lower()[:32]
    safe_device = str(device_id or "").strip()[:128] or None

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO mobile_analysis_results (
            received_at, client_type, device_id, result_json
        ) VALUES (?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        safe_client,
        safe_device,
        json.dumps(result, ensure_ascii=False),
    ))
    result_id = cursor.lastrowid

    # Prevent continuous monitoring from growing the local database forever.
    cursor.execute("""
        DELETE FROM mobile_analysis_results
        WHERE id NOT IN (
            SELECT id FROM mobile_analysis_results
            ORDER BY id DESC LIMIT 100
        )
    """)
    conn.commit()
    conn.close()
    return result_id


def get_latest_mobile_analysis(client_type="android"):
    """Return the most recent synchronized result for the requested client."""
    safe_client = str(client_type or "android").strip().lower()[:32]
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, received_at, client_type, device_id, result_json
        FROM mobile_analysis_results
        WHERE client_type = ?
        ORDER BY id DESC
        LIMIT 1
    """, (safe_client,))
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    return {
        "id": row["id"],
        "received_at": row["received_at"],
        "client_type": row["client_type"],
        "device_id": row["device_id"],
        "result": json.loads(row["result_json"]),
    }


def save_analysis(filename, result):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO analyses (
            filename,
            created_at,
            classification,
            confidence,
            genuine_probability,
            deepfake_probability,
            tampered_probability,
            replay_probability,
            voice_clone_probability,
            spectrogram_probability,
            voice_feature_probability,
            replay_feature_probability,
            f4_max,
            f4_mean,
            f4_median,
            f4_std,
            f4_suspicious_ratio,
            f4_high_ratio,
            suspicious_windows
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        filename,
        datetime.now().isoformat(),
        result["classification"],
        result["confidence"],

        result["class_probabilities"].get("genuine", 0),
        result["class_probabilities"].get("deepfake", 0),
        result["class_probabilities"].get("tampered", 0),
        result["class_probabilities"].get("replay", 0),

        result["features"].get("voice_clone_probability", 0),
        result["features"].get("spectrogram_probability", 0),
        result["features"].get("voice_feature_probability", 0),
        result["features"].get("replay_probability", 0),

        result["tampering"].get("f4_max", 0),
        result["tampering"].get("f4_mean", 0),
        result["tampering"].get("f4_median", 0),
        result["tampering"].get("f4_std", 0),
        result["tampering"].get("f4_suspicious_ratio", 0),
        result["tampering"].get("f4_high_ratio", 0),

        json.dumps(
            result["tampering"].get(
                "suspicious_windows",
                []
            )
        )
    ))

    analysis_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return analysis_id


def get_reports():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM analyses
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()
    reports = []

    for row in rows:
        report = dict(row)

        try:
            report["suspicious_windows"] = json.loads(
                report["suspicious_windows"] or "[]"
            )
        except (json.JSONDecodeError, TypeError):
            report["suspicious_windows"] = []

        reports.append(report)

    conn.close()
    return reports


# =========================================================
# INCIDENT MANAGEMENT
# =========================================================

def create_incident(
    incident_uuid,
    filename,
    classification,
    confidence,
    risk_score,
    risk_level,
    alert_message,
    recommended_action,
    evidence_warning=False,
    evidence_flags=None,
    analysis_id=None,
    verification_required=True
):
    now = datetime.now().isoformat()
    flags = evidence_flags or []

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO incidents (
            incident_uuid,
            analysis_id,
            filename,
            created_at,
            updated_at,
            classification,
            confidence,
            risk_score,
            risk_level,
            alert_message,
            recommended_action,
            evidence_warning,
            evidence_flags,
            incident_status,
            verification_required,
            verification_status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        incident_uuid,
        analysis_id,
        filename,
        now,
        now,
        classification,
        confidence,
        risk_score,
        risk_level,
        alert_message,
        recommended_action,
        1 if evidence_warning else 0,
        json.dumps(flags),
        "OPEN",
        1 if verification_required else 0,
        "NOT_STARTED" if verification_required else "NOT_REQUIRED"
    ))

    conn.commit()
    incident_id = cursor.lastrowid
    conn.close()

    add_incident_audit_event(
        incident_uuid=incident_uuid,
        event_type="INCIDENT_CREATED",
        event_status="OPEN",
        message="Security incident created from SonicT forensic analysis.",
        metadata={
            "risk_level": risk_level,
            "classification": classification,
            "verification_required": bool(verification_required),
        },
    )

    return get_incident_by_id(incident_id)


def create_user(name, email, password_hash):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (str(name).strip(), str(email).strip().lower(), password_hash, datetime.now().isoformat()),
        )
        user_id = cursor.lastrowid
        conn.commit()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        return dict(cursor.fetchone())
    finally:
        conn.close()


def get_user_by_email(email):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (str(email).strip().lower(),))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def mark_user_login(user_id):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "UPDATE users SET last_login_at = ? WHERE id = ?",
            (datetime.now().isoformat(), int(user_id)),
        )
        conn.commit()
    finally:
        conn.close()


def _deserialize_incident(row):
    if row is None:
        return None

    incident = dict(row)

    try:
        incident["evidence_flags"] = json.loads(
            incident.get("evidence_flags") or "[]"
        )
    except (json.JSONDecodeError, TypeError):
        incident["evidence_flags"] = []

    incident["evidence_warning"] = bool(
        incident.get("evidence_warning")
    )
    incident["verification_required"] = bool(
        incident.get("verification_required")
    )

    # Never return the stored OTP hash through API helpers.
    incident.pop("otp_hash", None)

    return incident


def get_incident_by_id(incident_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM incidents
        WHERE id = ?
    """, (incident_id,))

    row = cursor.fetchone()
    conn.close()

    return _deserialize_incident(row)


def get_incident_by_uuid(incident_uuid):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM incidents
        WHERE incident_uuid = ?
    """, (incident_uuid,))

    row = cursor.fetchone()
    conn.close()

    return _deserialize_incident(row)


def get_incident_verification_secret(incident_uuid):
    """
    Internal backend helper. Includes otp_hash and must never be
    returned directly to the frontend.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM incidents
        WHERE incident_uuid = ?
    """, (incident_uuid,))

    row = cursor.fetchone()
    conn.close()

    return dict(row) if row else None


def get_incidents(limit=100):
    safe_limit = max(1, min(int(limit), 500))

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM incidents
        ORDER BY id DESC
        LIMIT ?
    """, (safe_limit,))

    rows = cursor.fetchall()
    conn.close()

    return [
        _deserialize_incident(row)
        for row in rows
    ]


def save_otp_challenge(
    incident_uuid,
    otp_hash,
    otp_expires_at,
    otp_attempts=0
):
    now = datetime.now().isoformat()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE incidents
        SET
            otp_hash = ?,
            otp_expires_at = ?,
            otp_attempts = ?,
            verification_status = 'PENDING',
            verification_method = 'TRUSTED_CHANNEL_OTP',
            updated_at = ?
        WHERE incident_uuid = ?
    """, (
        otp_hash,
        otp_expires_at,
        otp_attempts,
        now,
        incident_uuid
    ))

    changed = cursor.rowcount
    conn.commit()
    conn.close()

    if changed > 0:
        add_incident_audit_event(
            incident_uuid=incident_uuid,
            event_type="OTP_CHALLENGE_CREATED",
            event_status="PENDING",
            message="Trusted-channel OTP challenge created.",
            metadata={"expires_at": otp_expires_at},
        )

    return changed > 0


def update_verification_result(
    incident_uuid,
    verification_status,
    otp_attempts,
    verified=False
):
    now = datetime.now().isoformat()
    verified_at = now if verified else None

    if verified:
        incident_status = "VERIFIED_VIA_TRUSTED_CHANNEL"
    elif verification_status in ("LOCKED", "EXPIRED", "FAILED"):
        incident_status = "VERIFICATION_FAILED"
    else:
        incident_status = "OPEN"

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE incidents
        SET
            verification_status = ?,
            otp_attempts = ?,
            verified_at = ?,
            incident_status = ?,
            updated_at = ?
        WHERE incident_uuid = ?
    """, (
        verification_status,
        otp_attempts,
        verified_at,
        incident_status,
        now,
        incident_uuid
    ))

    changed = cursor.rowcount
    conn.commit()
    conn.close()

    if changed > 0:
        add_incident_audit_event(
            incident_uuid=incident_uuid,
            event_type=(
                "IDENTITY_VERIFIED"
                if verified
                else "OTP_VERIFICATION_ATTEMPT"
            ),
            event_status=verification_status,
            message=(
                "Trusted-channel identity verification succeeded."
                if verified
                else "Trusted-channel OTP verification did not succeed."
            ),
            metadata={
                "attempts": int(otp_attempts or 0),
                "forensic_classification_changed": False,
            },
        )

    return changed > 0


def add_incident_audit_event(
    incident_uuid,
    event_type,
    event_status=None,
    message=None,
    metadata=None,
):
    """Append an operational event without changing forensic results."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO incident_audit_events (
            incident_uuid,
            event_type,
            event_status,
            message,
            event_metadata,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        incident_uuid,
        event_type,
        event_status,
        message,
        json.dumps(metadata or {}),
        datetime.now().isoformat(),
    ))

    event_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return event_id


def get_incident_audit_events(incident_uuid):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, incident_uuid, event_type, event_status,
               message, event_metadata, created_at
        FROM incident_audit_events
        WHERE incident_uuid = ?
        ORDER BY id ASC
    """, (incident_uuid,))
    rows = cursor.fetchall()
    conn.close()

    events = []
    for row in rows:
        event = dict(row)
        try:
            event["metadata"] = json.loads(
                event.pop("event_metadata") or "{}"
            )
        except (json.JSONDecodeError, TypeError):
            event.pop("event_metadata", None)
            event["metadata"] = {}
        events.append(event)
    return events


def update_incident_lifecycle(
    incident_uuid,
    action,
    investigator_notes=None,
    resolution=None,
):
    """Update case-management state without changing forensic output."""
    action = str(action or "").strip().upper()
    now = datetime.now().isoformat()

    transitions = {
        "ACKNOWLEDGE": "ACKNOWLEDGED",
        "START_INVESTIGATION": "INVESTIGATING",
        "ADD_NOTE": None,
        "RESOLVE": "RESOLVED",
        "REOPEN": "OPEN",
    }
    if action not in transitions:
        raise ValueError("Unsupported incident lifecycle action.")

    current = get_incident_by_uuid(incident_uuid)
    if current is None:
        return None

    new_status = transitions[action] or current.get("case_status") or "OPEN"
    notes = (
        str(investigator_notes).strip()
        if investigator_notes is not None
        else current.get("investigator_notes")
    )
    final_resolution = (
        str(resolution).strip().upper()
        if action == "RESOLVE" and resolution
        else (None if action == "REOPEN" else current.get("resolution"))
    )
    acknowledged_at = (
        now
        if action == "ACKNOWLEDGE" and not current.get("acknowledged_at")
        else current.get("acknowledged_at")
    )
    resolved_at = now if action == "RESOLVE" else (
        None if action == "REOPEN" else current.get("resolved_at")
    )

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE incidents
        SET case_status = ?, investigator_notes = ?, resolution = ?,
            acknowledged_at = ?, resolved_at = ?, updated_at = ?
        WHERE incident_uuid = ?
    """, (
        new_status,
        notes,
        final_resolution,
        acknowledged_at,
        resolved_at,
        now,
        incident_uuid,
    ))
    conn.commit()
    conn.close()

    event_messages = {
        "ACKNOWLEDGE": "Incident acknowledged by investigator.",
        "START_INVESTIGATION": "Formal incident investigation started.",
        "ADD_NOTE": "Investigator notes updated.",
        "RESOLVE": "Incident investigation resolved.",
        "REOPEN": "Incident reopened for further investigation.",
    }
    add_incident_audit_event(
        incident_uuid=incident_uuid,
        event_type=f"CASE_{action}",
        event_status=new_status,
        message=event_messages[action],
        metadata={
            "resolution": final_resolution,
            "notes_present": bool(notes),
            "forensic_classification_changed": False,
        },
    )
    return get_incident_by_uuid(incident_uuid)
