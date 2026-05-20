import os
import time
from pathlib import Path
import jwt
import secrets
from datetime import datetime, timezone, timedelta

import config.settings

# Fallbacks for settings
CACHE_DIR = getattr(config.settings, "CACHE_DIR", Path(os.getcwd()) / "cache")
JWT_SECRET = getattr(config.settings, "JWT_SECRET", "")

def get_jwt_secret() -> str:
    global JWT_SECRET
    if JWT_SECRET:
        return JWT_SECRET
        
    # Fallback to generated secret
    secret_path = CACHE_DIR / ".jwt_secret"
    if secret_path.exists():
        with open(secret_path, "r", encoding="utf-8") as f:
            JWT_SECRET = f.read().strip()
    else:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        JWT_SECRET = secrets.token_hex(32)
        with open(secret_path, "w", encoding="utf-8") as f:
            f.write(JWT_SECRET)
    return JWT_SECRET

def issue_token(user_id: int, email: str, ttl_hours: int = 168) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "email": email,
        "iat": now,
        "exp": now + timedelta(hours=ttl_hours)
    }
    return jwt.encode(payload, get_jwt_secret(), algorithm="HS256")

def decode_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, get_jwt_secret(), algorithms=["HS256"])
    except Exception:
        return None
