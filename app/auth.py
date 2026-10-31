"""Password hashing and JWT access tokens."""

import secrets
from datetime import UTC, datetime, timedelta

import jwt
from pwdlib import PasswordHash

_hasher = PasswordHash.recommended()  # Argon2
_ALGORITHM = "HS256"


class InvalidTokenError(Exception):
    pass


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return _hasher.verify(password, password_hash)


# Checked against when the username does not exist, so that login costs the same.
DUMMY_HASH = hash_password(secrets.token_urlsafe(16))


def resolve_secret(configured: str) -> str:
    """Use the configured secret, or a random one (tokens then reset on restart)."""
    return configured or secrets.token_urlsafe(32)


def create_access_token(user_id: int, secret: str, expires_minutes: int) -> str:
    expires = datetime.now(UTC) + timedelta(minutes=expires_minutes)
    return jwt.encode({"sub": str(user_id), "exp": expires}, secret, algorithm=_ALGORITHM)


def read_access_token(token: str, secret: str) -> int:
    """Return the user ID inside a valid token; raise InvalidTokenError otherwise."""
    try:
        payload = jwt.decode(token, secret, algorithms=[_ALGORITHM])
        return int(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError) as exc:
        raise InvalidTokenError(str(exc)) from exc
