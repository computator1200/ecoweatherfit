import smtplib
from email.message import EmailMessage
import logging
import streamlit as st

import config.settings

logger = logging.getLogger(__name__)

# Fallbacks
SMTP_HOST = getattr(config.settings, "SMTP_HOST", "")
SMTP_PORT = getattr(config.settings, "SMTP_PORT", 587)
SMTP_USER = getattr(config.settings, "SMTP_USER", "")
SMTP_PASS = getattr(config.settings, "SMTP_PASS", "")
SMTP_FROM = getattr(config.settings, "SMTP_FROM", "noreply@ecoweatherfit.local")

def send_verification_email(email: str, verification_url: str) -> None:
    if SMTP_HOST and SMTP_USER and SMTP_PASS and SMTP_FROM:
        # Send via SMTP
        msg = EmailMessage()
        msg["Subject"] = "Verify your EcoWeatherFit account"
        msg["From"] = SMTP_FROM
        msg["To"] = email
        
        html_content = f"""
        <html>
            <body>
                <h2>Welcome to EcoWeatherFit!</h2>
                <p>Please verify your email address by clicking the link below:</p>
                <p><a href="{verification_url}">Verify Email</a></p>
            </body>
        </html>
        """
        msg.set_content("Please verify your email address: " + verification_url)
        msg.add_alternative(html_content, subtype="html")
        
        try:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
                server.starttls()
                server.login(SMTP_USER, SMTP_PASS)
                server.send_message(msg)
            logger.info(f"Verification email sent to {email}")
        except Exception as e:
            logger.error(f"Failed to send verification email to {email}: {e}")
    else:
        # Dev fallback
        logger.info(f"DEV FALLBACK: Verification URL for {email}: {verification_url}")
        st.session_state["_last_verification_link"] = verification_url
