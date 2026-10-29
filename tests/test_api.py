import json

import pytest
from fastapi.testclient import TestClient

from app.llm import LLMError
from app.main import create_app

DOC = b"Chunk overlap repeats a few hundred characters between neighbouring chunks."
QUESTION = {"question": "What does chunk overlap repeat?"}


def sign_up(client: TestClient, username: str = "alice") -> dict[str, str]:
    client.post("/api/auth/register", json={"username": username, "password": "s3cret-pass"})
    token = client.post(
        "/api/auth/login", data={"username": username, "password": "s3cret-pass"}
    ).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def auth(client) -> dict[str, str]:
    return sign_up(client)


def upload(client, auth, name="notes.md", data=DOC):
    return client.post("/api/documents", files={"file": (name, data)}, headers=auth)


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_upload_list_ask_delete(client, auth):
    created = upload(client, auth)
    assert created.status_code == 201
    doc_id = created.json()["doc_id"]

    docs = client.get("/api/documents", headers=auth).json()
    assert docs == [{"doc_id": doc_id, "filename": "notes.md", "pages": 1, "chunks": 1}]

    answer = client.post("/api/ask", json=QUESTION, headers=auth).json()
    assert answer["grounded"] is True
    assert answer["citations"][0]["filename"] == "notes.md"
    assert answer["mode"] == "hybrid_rerank"

    assert client.delete(f"/api/documents/{doc_id}", headers=auth).status_code == 204
    assert client.delete(f"/api/documents/{doc_id}", headers=auth).status_code == 404


def test_ask_accepts_a_search_mode(client, auth):
    upload(client, auth)

    answer = client.post("/api/ask", json=QUESTION | {"mode": "semantic"}, headers=auth)

    assert answer.json()["mode"] == "semantic"
    invalid = client.post("/api/ask", json=QUESTION | {"mode": "magic"}, headers=auth)
    assert invalid.status_code == 422


def test_unsupported_file_is_rejected(client, auth):
    response = upload(client, auth, name="photo.png", data=b"\x89PNG")

    assert response.status_code == 400
    assert "Unsupported" in response.json()["detail"]


def test_broken_pdf_is_rejected(client, auth):
    response = upload(client, auth, name="report.pdf", data=b"not really a pdf")

    assert response.status_code == 400
    assert "not a readable PDF" in response.json()["detail"]


def test_too_large_file_is_rejected(client, auth, settings):
    settings.max_upload_mb = 0

    assert upload(client, auth).status_code == 413


def test_empty_question_is_rejected(client, auth):
    assert client.post("/api/ask", json={"question": ""}, headers=auth).status_code == 422


def test_model_down_returns_503(client, auth, llm):
    upload(client, auth)
    llm.reply = LLMError("Cannot reach Ollama")

    response = client.post("/api/ask", json=QUESTION, headers=auth)

    assert response.status_code == 503
    assert "Ollama" in response.json()["detail"]


def stream_events(client, auth, body=QUESTION) -> list[dict]:
    response = client.post("/api/ask/stream", json=body, headers=auth)
    assert response.status_code == 200
    return [json.loads(line) for line in response.text.splitlines()]


def test_ask_stream_sends_text_then_the_answer_and_saves_it(client, auth):
    upload(client, auth)

    events = stream_events(client, auth)

    *deltas, final = events
    assert len(deltas) > 1 and all(e["type"] == "delta" for e in deltas)
    assert "".join(e["text"] for e in deltas) == "Chunk overlap repeats text [1]."
    assert final["type"] == "answer"
    assert final["grounded"] is True
    assert final["citations"][0]["filename"] == "notes.md"
    history = client.get("/api/history", headers=auth).json()
    assert [item["answer"] for item in history] == [final["answer"]]


def test_ask_stream_off_topic_question_sends_only_the_answer(client, auth):
    upload(client, auth)

    events = stream_events(client, auth, {"question": "zebra giraffe safari"})

    assert [e["type"] for e in events] == ["answer"]
    assert events[0]["answer"] == "I don't know."


def test_ask_stream_reports_model_errors(client, auth, llm):
    upload(client, auth)
    llm.reply = LLMError("Cannot reach Ollama")

    events = stream_events(client, auth)

    assert events == [{"type": "error", "detail": "Cannot reach Ollama"}]


def test_evaluation_results_are_served(client, settings):
    assert client.get("/api/evaluation").json() == {"retrieval": None, "answers": None}

    settings.eval_results_dir.mkdir()
    (settings.eval_results_dir / "retrieval.json").write_text(json.dumps({"questions": 70}))

    assert client.get("/api/evaluation").json()["retrieval"] == {"questions": 70}


def test_built_frontend_is_served_at_root(rag, settings):
    settings.frontend_dist.mkdir()
    (settings.frontend_dist / "index.html").write_text("<h1>chat</h1>")

    with TestClient(create_app(service=rag, settings=settings)) as client:
        assert "chat" in client.get("/").text
        assert client.get("/api/health").status_code == 200
