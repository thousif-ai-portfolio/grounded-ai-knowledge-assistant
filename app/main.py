from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile, status

from app.config import Settings, get_settings
from app.embeddings import SentenceTransformerEmbedder
from app.ingest import UnsupportedFileError
from app.llm import LLMError, create_llm
from app.rag import EmptyDocumentError, RagService
from app.schemas import AskIn, AskOut, DocumentOut, UploadOut
from app.vector_store import ChromaStore


def build_service(settings: Settings) -> RagService:
    return RagService(
        embedder=SentenceTransformerEmbedder(settings.embedding_model),
        store=ChromaStore(settings.data_dir, settings.collection_name),
        llm=create_llm(settings),
        settings=settings,
    )


def create_app(service: RagService | None = None, settings: Settings | None = None) -> FastAPI:
    """Pass `service` in tests to use fakes; otherwise the real one is built at startup."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if app.state.service is None:
            app.state.service = build_service(settings)
        yield

    app = FastAPI(title="Grounded AI Knowledge Assistant", version="0.1.0", lifespan=lifespan)
    app.state.service = service

    def get_service(request: Request) -> RagService:
        return request.app.state.service

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/documents", status_code=status.HTTP_201_CREATED)
    def upload_document(file: UploadFile, rag: RagService = Depends(get_service)) -> UploadOut:
        max_bytes = settings.max_upload_mb * 1024 * 1024
        data = file.file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(413, f"File is over {settings.max_upload_mb} MB")
        try:
            added = rag.add_document(file.filename or "upload.txt", data)
        except (UnsupportedFileError, EmptyDocumentError) as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
        return UploadOut(**asdict(added))

    @app.get("/documents")
    def list_documents(rag: RagService = Depends(get_service)) -> list[DocumentOut]:
        return [DocumentOut(**asdict(doc)) for doc in rag.list_documents()]

    @app.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_document(doc_id: str, rag: RagService = Depends(get_service)) -> None:
        if not rag.delete_document(doc_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")

    @app.post("/ask")
    def ask(body: AskIn, rag: RagService = Depends(get_service)) -> AskOut:
        try:
            answer = rag.ask(body.question, body.top_k)
        except LLMError as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
        return AskOut(**asdict(answer))

    return app


app = create_app()
