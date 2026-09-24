import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone


OTP_EXPIRY_MINUTES = 5
OTP_MAX_ATTEMPTS = 5
OTP_SECRET = os.getenv(
    "SONICT_OTP_SECRET",
    "sonict-local-development-otp-secret",
)


def _utc_now():
    return datetime.now(timezone.utc)


def _hash_otp(incident_id, otp):
    payload = f"{incident_id}:{otp}".encode("utf-8")
    return hmac.new(
        OTP_SECRET.encode("utf-8"),
        payload,
        hashlib.sha256,
    ).hexdigest()


def generate_otp():
    return f"{secrets.randbelow(1_000_000):06d}"


def create_otp_record(incident_id):
    otp = generate_otp()
    expires_at = _utc_now() + timedelta(minutes=OTP_EXPIRY_MINUTES)

    return {
        "otp_hash": _hash_otp(incident_id, otp),
        "otp_expires_at": expires_at.isoformat(),
        "otp_attempts": 0,
        "verification_status": "PENDING",
        # Local prototype only. Never return an OTP in production.
        "development_otp": otp,
    }


def verify_otp_value(
    incident_id,
    supplied_otp,
    stored_hash,
    expires_at,
    attempts,
):
    attempts = int(attempts or 0)

    if attempts >= OTP_MAX_ATTEMPTS:
        return False, "LOCKED", attempts, "Maximum OTP attempts exceeded."

    try:
        expiry = datetime.fromisoformat(expires_at)
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return False, "FAILED", attempts, "Invalid OTP expiry information."

    if _utc_now() > expiry:
        return False, "EXPIRED", attempts, "OTP has expired."

    new_attempts = attempts + 1
    supplied_hash = _hash_otp(incident_id, str(supplied_otp).strip())

    if hmac.compare_digest(str(stored_hash), supplied_hash):
        return (
            True,
            "VERIFIED",
            new_attempts,
            "Trusted-channel verification successful.",
        )

    if new_attempts >= OTP_MAX_ATTEMPTS:
        return (
            False,
            "LOCKED",
            new_attempts,
            "Invalid OTP. Maximum attempts exceeded.",
        )

    return False, "PENDING", new_attempts, "Invalid OTP."
