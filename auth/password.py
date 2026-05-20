"""Password hashing.

Bcrypt via passlib at cost factor 12. The plain-text password is held only
within the local scope of ``hash_password`` and is never logged, persisted, or
returned. Bcrypt encodes its cost into the hash string, so the cost factor
can later be raised in a single config change without re-hashing existing
rows — passlib will transparently upgrade on next login.
"""

from passlib.context import CryptContext

# Cost factor 12: ~250 ms verify on a modern laptop CPU (passlib benchmark
# guidance), prohibitive for offline attack at this cost.
#
# Note on bcrypt version pinning: passlib 1.7 has a self-test (detect_wrap_bug)
# that fails on bcrypt 4.1+ because the new library rejects >72-byte inputs
# rather than truncating. requirements.txt pins bcrypt<4.1 to keep passlib's
# bcrypt backend usable; no runtime monkey-patching is required.
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)


def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)
