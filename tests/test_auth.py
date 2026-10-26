from datetime import UTC, datetime, timedelta

import jwt
import pytest

from app.auth import (
    InvalidTokenError,
    create_access_token,
    hash_password,
    read_access_token,
    verify_password,
)
from tests.test_api import DOC, QUESTION, sign_up, upload

PASSWORD = "s3cret-pass"
SECRET = "unit-test-secret-that-is-at-least-32-bytes"


def test_passwords_are_hashed_not_stored():
    hashed = hash_password(PASSWORD)

    assert PASSWORD not in hashed
    assert hashed.startswith("$argon2")
    assert verify_password(PASSWORD, hashed)
    assert not verify_password("wrong-pass", hashed)


def test_token_round_trip_and_rejections():
    token = create_access_token(7, SECRET, expires_minutes=5)

    assert read_access_token(token, SECRET) == 7
    with pytest.raises(InvalidTokenError):
        read_access_token(token, "another-secret-that-is-at-least-32-bytes")
    expired = jwt.encode(
        {"sub": "7", "exp": datetime.now(UTC) - timedelta(minutes=1)}, SECRET, algorithm="HS256"
    )
    with pytest.raises(InvalidTokenError):
        read_access_token(expired, SECRET)


def test_register_login_me(client):
    created = client.post("/api/auth/register", json={"username": "bob", "password": PASSWORD})
    assert created.status_code == 201

    auth = sign_up(client, "bob")  # registering again is a no-op 409; login still works

    assert client.get("/api/auth/me", headers=auth).json()["username"] == "bob"


def test_duplicate_username_is_rejected(client):
    body = {"username": "bob", "password": PASSWORD}
    client.post("/api/auth/register", json=body)

    assert client.post("/api/auth/register", json=body).status_code == 409


@pytest.mark.parametrize(
    "body",
    [
        {"username": "ab", "password": PASSWORD},  # too short
        {"username": "bad name!", "password": PASSWORD},  # bad characters
        {"username": "carol", "password": "short"},  # weak password
    ],
)
def test_invalid_registrations(client, body):
    assert client.post("/api/auth/register", json=body).status_code == 422


def test_wrong_password_is_rejected(client):
    client.post("/api/auth/register", json={"username": "bob", "password": PASSWORD})

    response = client.post("/api/auth/login", data={"username": "bob", "password": "nope-nope"})

    assert response.status_code == 401


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/api/documents"),
        ("post", "/api/ask"),
        ("get", "/api/history"),
        ("get", "/api/auth/me"),
        ("delete", "/api/documents/abc"),
    ],
)
def test_protected_routes_need_a_token(client, method, path):
    assert getattr(client, method)(path).status_code == 401
    bad = {"Authorization": "Bearer not-a-token"}
    assert getattr(client, method)(path, headers=bad).status_code == 401


def test_history_is_saved_and_private(client):
    alice = sign_up(client, "alice")
    bob = sign_up(client, "bob")
    upload(client, alice, data=DOC)

    client.post("/api/ask", json=QUESTION, headers=alice)

    history = client.get("/api/history", headers=alice).json()
    assert [h["question"] for h in history] == [QUESTION["question"]]
    assert history[0]["citations"][0]["filename"] == "notes.md"
    assert client.get("/api/history", headers=bob).json() == []

    assert client.delete("/api/history", headers=alice).status_code == 204
    assert client.get("/api/history", headers=alice).json() == []
