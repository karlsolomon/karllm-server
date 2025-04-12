import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml
from authlib.jose import JsonWebKey, jwt
from authlib.jose.errors import JoseError
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# Constants
ALGORITHM = "EdDSA"
SESSION_TIMEOUT = timedelta(minutes=2)
auth_scheme = HTTPBearer()
ACTIVE_SESSIONS = {}  # session_id -> {"username": str, "last_seen": datetime}


# Load JWT signing/verification key algorithm
def get_config_dir():
    return Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser() / "karllm"


# Load trusted public keys from YAML config + separate PEM files
def load_public_keys():
    config_dir = get_config_dir()
    conf_path = config_dir / "server.conf"
    keys_dir = config_dir / "keys"

    if not conf_path.exists():
        raise RuntimeError(f"Missing config file: {conf_path}")
    if not keys_dir.exists():
        raise RuntimeError(f"Missing key directory: {keys_dir}")

    with open(conf_path, "r") as f:
        config = yaml.safe_load(f)

    keys = {}
    for username, key_filename in config.get("clients", []):
        key_path = keys_dir / key_filename
        if not key_path.exists():
            raise RuntimeError(f"Missing key file for user '{username}': {key_path}")
        pem = key_path.read_text().strip()
        if not pem.startswith("-----BEGIN PUBLIC KEY-----"):
            raise RuntimeError(
                f"Key for user '{username}' is not PEM format: {key_path}"
            )
        keys[username] = pem
    return keys


# Load trusted PEM-formatted public keys from disk
PUBLIC_KEYS = load_public_keys()


# Decode + verify JWT, create a temporary session
def verify_jwt_and_create_session(
    credentials: HTTPAuthorizationCredentials = Depends(auth_scheme),
):
    token = credentials.credentials

    try:
        # Try each trusted public key until one successfully verifies
        for username, pem in PUBLIC_KEYS.items():
            try:
                jwk = JsonWebKey.import_key(pem, {"kty": "OKP"})
                claims = jwt.decode(token, key=jwk)
                claims.validate()
                # If validation passes, we found the correct user
                session_id = str(uuid.uuid4())
                ACTIVE_SESSIONS[session_id] = {
                    "username": username,
                    "last_seen": datetime.now(timezone.utc),
                }
                return {"session_id": session_id, "username": username}
            except JoseError:
                continue

        raise HTTPException(status_code=401, detail="No matching key for token")

    except Exception as e:
        raise HTTPException(status_code=401, detail=f"JWT validation failed: {str(e)}")


def require_session(request: Request):
    session_id = request.headers.get("X-Session-Token")
    if not session_id:
        raise HTTPException(status_code=401, detail="Missing session token")

    session = ACTIVE_SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid session token")

    now = datetime.now(timezone.utc)
    last_seen = session["last_seen"]

    # Fix: ensure last_seen is timezone-aware
    if last_seen.tzinfo is None:
        last_seen = last_seen.replace(tzinfo=timezone.utc)

    if now - last_seen > SESSION_TIMEOUT:
        del ACTIVE_SESSIONS[session_id]
        raise HTTPException(status_code=401, detail="Session expired")

    return session
