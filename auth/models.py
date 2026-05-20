from datetime import datetime, timezone
import uuid
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship

from auth.database import Base

def now_utc():
    return datetime.now(timezone.utc)

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # Both email and username are unique identifiers that the user can log in
    # with. Email is also the address that a verification link would be sent
    # to in the SMTP-enabled deployment path; username is the display name
    # rendered in the sidebar.
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=now_utc, nullable=False)
    email_verified = Column(Boolean, default=False, nullable=False)
    verification_token = Column(String, nullable=True)
    verification_token_expires_at = Column(DateTime, nullable=True)
    last_login_at = Column(DateTime, nullable=True)

class UserPreference(Base):
    __tablename__ = "user_preferences"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)
    default_city = Column(String, default="London", nullable=False)
    gender_presentation = Column(String, default="Unisex", nullable=False)
    style_preference = Column(String, default="Casual", nullable=False)
    show_eco_tips = Column(Boolean, default=True, nullable=False)
    show_circular_loop = Column(Boolean, default=True, nullable=False)
    updated_at = Column(DateTime, default=now_utc, onupdate=now_utc, nullable=False)

class WardrobeItem(Base):
    __tablename__ = "wardrobe_items"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    label = Column(String, nullable=False)
    category = Column(String, nullable=False)
    warmth = Column(String, nullable=False)
    waterproof = Column(Boolean, default=False, nullable=False)
    notes = Column(String, nullable=True)
    created_at = Column(DateTime, default=now_utc, nullable=False)

class ForecastHistory(Base):
    __tablename__ = "forecast_history"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    city = Column(String, nullable=False)
    requested_at = Column(DateTime, default=now_utc, nullable=False)
    forecast_summary = Column(String, nullable=False)
    recommendation_text = Column(String, nullable=False)
    constraints_json = Column(String, nullable=True)
