import pytest
from fastapi.testclient import TestClient

from app.llm import LLMError
from app.main import create_app

DOC = b"Chunk overlap repeats a few hundred characters between neighbouring chunks."


@pytest.fixture
def client(rag, settings) -> TestClient:
    with TestClient(create_app(service=rag, settings=settings)) as client:
        yield client


def upload(client, name="notes.md", data=DOC):
    return client.post("/documents", files={"file": (name, data)})


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_upload_list_ask_delete(client):
    created = upload(client)
    assert created.status_code == 201
    doc_id = created.json()["doc_id"]

    docs = client.get("/documents").json()
    assert docs == [{"doc_id": doc_id, "filename": "notes.md", "pages": 1, "chunks": 1}]

    answer = client.post("/ask", json={"question": "What does chunk overlap repeat?"}).json()
    assert answer["grounded"] is True
    assert answer["citations"][0]["filename"] == "notes.md"

    assert client.delete(f"/documents/{doc_id}").status_code == 204
    assert client.delete(f"/documents/{doc_id}").status_code == 404


def test_unsupported_file_is_rejected(client):
    response = upload(client, name="photo.png", data=b"\x89PNG")

    assert response.status_code == 400
    assert "Unsupported" in response.json()["detail"]


def test_too_large_file_is_rejected(client, settings):
    settings.max_upload_mb = 0

    assert upload(client).status_code == 413


def test_empty_question_is_rejected(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_model_down_returns_503(client, llm):
    upload(client)
    llm.reply = LLMError("Cannot reach Ollama")

    response = client.post("/ask", json={"question": "What does chunk overlap repeat?"})

    assert response.status_code == 503
    assert "Ollama" in response.json()["detail"]
