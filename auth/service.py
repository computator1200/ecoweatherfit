import uuid
import json
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from email_validator import validate_email, EmailNotValidError

from auth.database import get_session
from auth.models import User, UserPreference, WardrobeItem, ForecastHistory
from auth.password import hash_password, verify_password
from auth.jwt_handler import issue_token, decode_token
from auth.email_service import send_verification_email

import re

USERNAME_RE = re.compile(r"^[A-Za-z0-9_.\-]{3,32}$")


@dataclass
class AuthResult:
    success: bool
    error: str | None = None
    user_id: int | None = None
    verification_url: str | None = None
    jwt_token: str | None = None
    email: str | None = None
    username: str | None = None


def now_utc():
    return datetime.now(timezone.utc)


def _validate_username(username: str) -> str | None:
    """Return an error message if the username is invalid; else None."""
    if not username or not USERNAME_RE.match(username):
        return ("Username must be 3–32 characters: letters, digits, "
                "underscore, dot, or hyphen.")
    return None


def signup(
    email: str,
    username: str,
    password: str,
    app_base_url: str = "",
    signup_code: str = "",
) -> AuthResult:
    """Create a new user.

    Two abuse-prevention paths are supported:
    1. Email verification (default). Account is created unverified; a link
       is sent (or shown in dev fallback). Login blocked until verified.
    2. Pre-shared signup code. When ``config.settings.SIGNUP_CODE`` is set
       to a non-empty value, the caller must pass a matching ``signup_code``;
       on match, the user is created with ``email_verified=True`` and can
       log in immediately. This is the path used in deployments without SMTP.

    Both paths require a unique email and a unique username.
    """
    # Local import so settings overridden at test time are honoured.
    from config.settings import SIGNUP_CODE as required_code

    try:
        valid = validate_email(email, check_deliverability=False)
        clean_email = valid.normalized
    except EmailNotValidError as e:
        return AuthResult(success=False, error=str(e))

    clean_username = (username or "").strip()
    if (err := _validate_username(clean_username)):
        return AuthResult(success=False, error=err)

    if len(password) < 8:
        return AuthResult(success=False, error="Password must be at least 8 characters long.")

    code_gated = bool(required_code)
    if code_gated and signup_code != required_code:
        return AuthResult(
            success=False,
            error="Invalid signup code. This deployment requires a valid code to create an account.",
        )

    with get_session() as session:
        if session.query(User).filter_by(email=clean_email).first():
            return AuthResult(success=False, error="Email already registered.")
        if session.query(User).filter_by(username=clean_username).first():
            return AuthResult(success=False, error="Username already taken.")

        if code_gated:
            # Skip the email-verification dance entirely.
            user = User(
                email=clean_email,
                username=clean_username,
                password_hash=hash_password(password),
                email_verified=True,
            )
            session.add(user)
            session.flush()
            session.add(UserPreference(user_id=user.id))
            session.commit()
            return AuthResult(
                success=True, user_id=user.id,
                username=clean_username, email=clean_email,
                verification_url=None,
            )

        # Email-verification path
        token = str(uuid.uuid4())
        expires = now_utc() + timedelta(hours=24)

        user = User(
            email=clean_email,
            username=clean_username,
            password_hash=hash_password(password),
            verification_token=token,
            verification_token_expires_at=expires,
        )
        session.add(user)
        session.flush()
        session.add(UserPreference(user_id=user.id))
        session.commit()

        url_sep = "&" if "?" in app_base_url else "?"
        verification_url = f"{app_base_url}{url_sep}verify_token={token}"
        send_verification_email(clean_email, verification_url)

        return AuthResult(
            success=True, user_id=user.id,
            username=clean_username, email=clean_email,
            verification_url=verification_url,
        )

def verify_email_token(token: str) -> AuthResult:
    with get_session() as session:
        user = session.query(User).filter_by(verification_token=token).first()
        if not user:
            return AuthResult(success=False, error="Invalid verification token.")

        # SQLite does not preserve timezone info; normalise both sides to
        # tz-aware UTC before comparing so we don't blow up on naive reads.
        expires_at = user.verification_token_expires_at
        if expires_at is not None:
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            if expires_at < now_utc():
                return AuthResult(success=False, error="Verification token has expired.")
            
        user.email_verified = True
        user.verification_token = None
        user.verification_token_expires_at = None
        session.commit()
        
        return AuthResult(success=True)

def login(email: str, password: str) -> AuthResult:
    """Authenticate by email OR username (the first arg accepts either).

    Kept named ``email`` for backward compatibility with existing tests and
    callers; the actual lookup tries both fields against the user-supplied
    identifier so the UI can present a single ‘Email or username’ field.
    """
    identifier = (email or "").strip()
    if not identifier:
        return AuthResult(success=False, error="Invalid credentials.")

    with get_session() as session:
        user = (
            session.query(User)
            .filter(
                (User.email == identifier.lower()) | (User.username == identifier)
            )
            .first()
        )
        if not user:
            return AuthResult(success=False, error="Invalid credentials.")

        if not verify_password(password, user.password_hash):
            return AuthResult(success=False, error="Invalid credentials.")

        if not user.email_verified:
            return AuthResult(success=False, error="Email not verified.")

        user.last_login_at = now_utc()
        session.commit()

        token = issue_token(user.id, user.email)
        return AuthResult(
            success=True, jwt_token=token,
            user_id=user.id, email=user.email, username=user.username,
        )

def get_current_user_from_token(token: str) -> User | None:
    payload = decode_token(token)
    if not payload:
        return None
        
    user_id = payload.get("sub")
    if not user_id:
        return None
        
    try:
        user_id_int = int(user_id)
    except ValueError:
        return None
        
    with get_session() as session:
        user = session.query(User).filter_by(id=user_id_int).first()
        # Keep user detached from session so it can be returned safely
        if user:
            session.expunge(user)
        return user

def update_preferences(user_id: int, **fields) -> AuthResult:
    with get_session() as session:
        pref = session.query(UserPreference).filter_by(user_id=user_id).first()
        if not pref:
            return AuthResult(success=False, error="Preferences not found.")
            
        for k, v in fields.items():
            if hasattr(pref, k):
                setattr(pref, k, v)
                
        session.commit()
        return AuthResult(success=True)

def add_wardrobe_item(user_id: int, label: str, category: str, warmth: str, waterproof: bool, notes: str | None = None) -> AuthResult:
    with get_session() as session:
        item = WardrobeItem(
            user_id=user_id,
            label=label,
            category=category,
            warmth=warmth,
            waterproof=waterproof,
            notes=notes
        )
        session.add(item)
        session.commit()
        return AuthResult(success=True, user_id=item.id)

def list_wardrobe(user_id: int) -> list[WardrobeItem]:
    with get_session() as session:
        items = session.query(WardrobeItem).filter_by(user_id=user_id).all()
        for item in items:
            session.expunge(item)
        return items

def delete_wardrobe_item(user_id: int, item_id: int) -> AuthResult:
    with get_session() as session:
        item = session.query(WardrobeItem).filter_by(id=item_id, user_id=user_id).first()
        if not item:
            return AuthResult(success=False, error="Item not found or unauthorized.")
            
        session.delete(item)
        session.commit()
        return AuthResult(success=True)

def record_forecast(user_id: int, city: str, forecast_summary: str, recommendation_text: str, constraints_json: str | None = None) -> AuthResult:
    with get_session() as session:
        if isinstance(constraints_json, dict):
            constraints_json = json.dumps(constraints_json)
            
        hist = ForecastHistory(
            user_id=user_id,
            city=city,
            forecast_summary=forecast_summary,
            recommendation_text=recommendation_text,
            constraints_json=constraints_json
        )
        session.add(hist)
        session.commit()
        return AuthResult(success=True)

def list_history(user_id: int, limit: int = 20) -> list[ForecastHistory]:
    with get_session() as session:
        history = session.query(ForecastHistory).filter_by(user_id=user_id).order_by(ForecastHistory.requested_at.desc()).limit(limit).all()
        for h in history:
            session.expunge(h)
        return history
