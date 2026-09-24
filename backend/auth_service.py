import base64
import hashlib
import hmac
import json
import os
import secrets
import time

PASSWORD_ITERATIONS = 310_000
TOKEN_TTL_SECONDS = 8 * 60 * 60
AUTH_SECRET = os.getenv("SONICT_AUTH_SECRET", "sonict-local-development-auth-secret-change-me")

def _encode(value):
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")

def _decode(value):
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))

def hash_password(password):
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", str(password).encode(), salt, PASSWORD_ITERATIONS)
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${_encode(salt)}${_encode(digest)}"

def verify_password(password, stored_hash):
    try:
        algorithm, iterations, salt, expected = str(stored_hash).split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        actual = hashlib.pbkdf2_hmac("sha256", str(password).encode(), _decode(salt), int(iterations))
        return hmac.compare_digest(_encode(actual), expected)
    except (TypeError, ValueError):
        return False

def create_access_token(user):
    now = int(time.time())
    payload = {"sub": int(user["id"]), "email": user["email"], "name": user["name"], "iat": now, "exp": now + TOKEN_TTL_SECONDS, "nonce": secrets.token_hex(8)}
    encoded = _encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())
    signature = hmac.new(AUTH_SECRET.encode(), encoded.encode(), hashlib.sha256).digest()
    return f"{encoded}.{_encode(signature)}"

def verify_access_token(token):
    try:
        encoded, supplied = str(token).split(".", 1)
        expected = hmac.new(AUTH_SECRET.encode(), encoded.encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(_encode(expected), supplied):
            return None
        payload = json.loads(_decode(encoded).decode())
        return payload if int(payload.get("exp", 0)) > int(time.time()) else None
    except (ValueError, TypeError, json.JSONDecodeError):
        return None
