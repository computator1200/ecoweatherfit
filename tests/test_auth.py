import pytest
from pathlib import Path
from datetime import datetime, timezone, timedelta
import uuid

import sys
import config.settings

# Mock config settings so imports don't fail
config.settings.DATABASE_URL = "sqlite:///:memory:"
config.settings.JWT_SECRET = "test-secret-at-least-thirty-two-bytes-long-for-hs256"
config.settings.JWT_TTL_HOURS = 168
config.settings.SMTP_HOST = ""
config.settings.SMTP_PORT = 587
config.settings.SMTP_USER = ""
config.settings.SMTP_PASS = ""
config.settings.SMTP_FROM = "noreply@test.com"
config.settings.APP_BASE_URL = "http://localhost:8501"

from auth.database import configure_engine, init_db, get_session
from auth.models import User, UserPreference, WardrobeItem, ForecastHistory
from auth.password import hash_password, verify_password
from auth.jwt_handler import issue_token, decode_token
from auth.service import (
    signup, login, verify_email_token, get_current_user_from_token,
    update_preferences, add_wardrobe_item, list_wardrobe, delete_wardrobe_item,
    record_forecast, list_history
)

@pytest.fixture(scope="session")
def engine_setup(tmp_path_factory):
    # Use a temp directory for the database
    db_dir = tmp_path_factory.mktemp("db")
    db_path = db_dir / "test_users.db"
    db_url = f"sqlite:///{db_path}"
    
    # Configure the engine to use the temp DB
    configure_engine(db_url)
    init_db()
    return db_url

@pytest.fixture(autouse=True)
def clean_db(engine_setup):
    """Clean the DB before each test by clearing tables without dropping them."""
    with get_session() as session:
        session.query(ForecastHistory).delete()
        session.query(WardrobeItem).delete()
        session.query(UserPreference).delete()
        session.query(User).delete()
        session.commit()

class TestPassword:
    def test_hash_returns_different_from_plain(self):
        plain = "password123"
        hashed = hash_password(plain)
        assert plain != hashed
        assert len(hashed) > 0

    def test_verify_matches_hash(self):
        plain = "password123"
        hashed = hash_password(plain)
        assert verify_password(plain, hashed) is True

    def test_verify_rejects_wrong_password(self):
        plain = "password123"
        hashed = hash_password(plain)
        assert verify_password("wrongpass", hashed) is False

class TestJWT:
    def test_issue_and_decode_round_trip(self):
        token = issue_token(1, "test@test.com", 24)
        payload = decode_token(token)
        assert payload is not None
        assert payload["sub"] == "1"
        assert payload["email"] == "test@test.com"

    def test_decode_rejects_tampered_signature(self):
        token = issue_token(1, "test@test.com", 24)
        tampered_token = token[:-5] + "aaaaa"
        payload = decode_token(tampered_token)
        assert payload is None

    def test_decode_returns_none_on_expired_token(self):
        token = issue_token(1, "test@test.com", -1) # expired 1 hour ago
        payload = decode_token(token)
        assert payload is None

    def test_decode_returns_none_on_malformed_token(self):
        payload = decode_token("not.a.token")
        assert payload is None

class TestSignupLoginFlow:
    def test_signup_creates_unverified_user(self):
        res = signup("test@example.com", "test", "password123")
        assert res.success is True
        assert res.user_id is not None
        assert res.verification_url is not None
        
        with get_session() as session:
            user = session.query(User).filter_by(id=res.user_id).first()
            assert user.email == "test@example.com"
            assert user.email_verified is False

    def test_signup_rejects_duplicate_email(self):
        signup("test@example.com", "test", "password123")
        res = signup("test@example.com", "test", "password123")
        assert res.success is False
        assert "already registered" in res.error

    def test_signup_rejects_invalid_email(self):
        res = signup("notanemail", "notanemail", "password123")
        assert res.success is False
        assert res.error is not None

    def test_signup_rejects_short_password(self):
        res = signup("test2@example.com", "test2", "short")
        assert res.success is False
        assert "least 8 characters" in res.error

    def test_login_rejected_before_email_verified(self):
        signup("test3@example.com", "test3", "password123")
        res = login("test3@example.com", "password123")
        assert res.success is False
        assert "Email not verified" in res.error

    def test_verify_email_token_then_login_succeeds(self):
        res1 = signup("test4@example.com", "test4", "password123")
        token = res1.verification_url.split("verify_token=")[-1]
        
        res2 = verify_email_token(token)
        assert res2.success is True
        
        res3 = login("test4@example.com", "password123")
        assert res3.success is True
        assert res3.jwt_token is not None

    def test_login_rejects_wrong_password(self):
        res1 = signup("test5@example.com", "test5", "password123")
        token = res1.verification_url.split("verify_token=")[-1]
        verify_email_token(token)
        
        res2 = login("test5@example.com", "wrongpass")
        assert res2.success is False
        assert "Invalid credentials" in res2.error

    def test_verify_email_token_rejects_invalid_token(self):
        res = verify_email_token("invalid-token-uuid")
        assert res.success is False
        assert "Invalid verification token" in res.error

    def test_verify_email_token_rejects_expired_token(self):
        res1 = signup("test6@example.com", "test6", "password123")
        token = res1.verification_url.split("verify_token=")[-1]
        
        with get_session() as session:
            user = session.query(User).filter_by(id=res1.user_id).first()
            user.verification_token_expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
            session.commit()
            
        res2 = verify_email_token(token)
        assert res2.success is False
        assert "expired" in res2.error

class TestUsernameAndSignupCode:
    def test_username_required_at_signup(self):
        res = signup("u_norm@example.com", "", "password123")
        assert res.success is False
        assert "Username must be" in res.error

    def test_username_rejects_too_short(self):
        res = signup("u_short@example.com", "ab", "password123")
        assert res.success is False
        assert "Username must be" in res.error

    def test_username_rejects_bad_chars(self):
        res = signup("u_bad@example.com", "hey!there", "password123")
        assert res.success is False
        assert "Username must be" in res.error

    def test_duplicate_username_rejected(self):
        signup("u_first@example.com", "duplicateuser", "password123")
        res = signup("u_second@example.com", "duplicateuser", "password123")
        assert res.success is False
        assert "Username already taken" in res.error

    def test_login_by_username(self):
        r = signup("u_login@example.com", "loginuser", "password123")
        verify_email_token(r.verification_url.split("verify_token=")[-1])
        res = login("loginuser", "password123")
        assert res.success is True
        assert res.username == "loginuser"

    def test_signup_code_required_when_set(self, monkeypatch):
        import config.settings
        monkeypatch.setattr(config.settings, "SIGNUP_CODE", "open-sesame-2026")
        # Wrong code
        bad = signup("c_bad@example.com", "codebad", "password123", signup_code="wrong")
        assert bad.success is False
        assert "Invalid signup code" in bad.error
        # Right code → auto-verified, can log in immediately without a token
        good = signup("c_good@example.com", "codegood", "password123",
                      signup_code="open-sesame-2026")
        assert good.success is True
        assert good.verification_url is None  # no email-verification path
        res = login("codegood", "password123")
        assert res.success is True

    def test_signup_code_ignored_when_unset(self, monkeypatch):
        import config.settings
        monkeypatch.setattr(config.settings, "SIGNUP_CODE", "")
        # signup_code argument is irrelevant when the deployment has no code set
        res = signup("c_off@example.com", "codeoff", "password123",
                     signup_code="anything")
        assert res.success is True
        assert res.verification_url is not None  # default email path


class TestUserScopedQueries:
    def test_wardrobe_isolated_between_users(self):
        r1 = signup("user1@example.com", "user1", "password123")
        r2 = signup("user2@example.com", "user2", "password123")
        
        add_wardrobe_item(r1.user_id, "Jacket", "coat", "heavy", False)
        add_wardrobe_item(r2.user_id, "Boots", "shoes", "warm", True)
        
        items1 = list_wardrobe(r1.user_id)
        assert len(items1) == 1
        assert items1[0].label == "Jacket"
        
        items2 = list_wardrobe(r2.user_id)
        assert len(items2) == 1
        assert items2[0].label == "Boots"

    def test_delete_wardrobe_blocks_cross_user(self):
        r1 = signup("user11@example.com", "user11", "password123")
        r2 = signup("user22@example.com", "user22", "password123")
        
        res_add = add_wardrobe_item(r1.user_id, "Jacket", "coat", "heavy", False)
        item_id = res_add.user_id # the item id was returned here
        
        res_del = delete_wardrobe_item(r2.user_id, item_id)
        assert res_del.success is False
        
        items1 = list_wardrobe(r1.user_id)
        assert len(items1) == 1

    def test_history_isolated_between_users(self):
        r1 = signup("hist1@example.com", "hist1", "password123")
        r2 = signup("hist2@example.com", "hist2", "password123")
        
        record_forecast(r1.user_id, "London", "Rain", "Wear a coat")
        record_forecast(r2.user_id, "Paris", "Sunny", "T-shirt")
        
        hist1 = list_history(r1.user_id)
        assert len(hist1) == 1
        assert hist1[0].city == "London"
        
        hist2 = list_history(r2.user_id)
        assert len(hist2) == 1
        assert hist2[0].city == "Paris"
