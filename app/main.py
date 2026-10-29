import json
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, UploadFile, status
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.auth import (
    InvalidTokenError,
    create_access_token,
    hash_password,
    read_access_token,
    resolve_secret,
    verify_password,
)
from app.config import Settings, get_settings
from app.db import ChatMessage, User, create_session_factory
from app.embeddings import SentenceTransformerEmbedder
from app.ingest import UnsupportedFileError
from app.llm import LLMError, OllamaLLM
from app.rag import EmptyDocumentError, RagService
from app.retrieval import CrossEncoderReranker
from app.schemas import (
    AskIn,
    AskOut,
    DocumentOut,
    EvaluationOut,
    HistoryItemOut,
    RegisterIn,
    TokenOut,
    UploadOut,
    UserOut,
)
from app.vector_store import ChromaStore


def build_service(settings: Settings) -> RagService:
    return RagService(
        embedder=SentenceTransformerEmbedder(settings.embedding_model),
        store=ChromaStore(settings.data_dir, settings.collection_name),
        llm=OllamaLLM(
            host=settings.ollama_host,
            model=settings.llm_model,
            keep_alive=settings.llm_keep_alive,
            max_tokens=settings.max_answer_tokens,
        ),
        settings=settings,
        reranker=CrossEncoderReranker(settings.rerank_model),
    )


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def create_app(service: RagService | None = None, settings: Settings | None = None) -> FastAPI:
    """Pass `service` in tests to use fakes; otherwise the real one is built at startup."""
    settings = settings or get_settings()
    jwt_secret = resolve_secret(settings.jwt_secret)
    session_factory = create_session_factory(settings.database_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if app.state.service is None:
            app.state.service = build_service(settings)
        yield

    app = FastAPI(title="Grounded AI Knowledge Assistant", version="1.0.0", lifespan=lifespan)
    app.state.service = service
    api = APIRouter(prefix="/api")
    oauth2 = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

    def get_service(request: Request) -> RagService:
        return request.app.state.service

    def get_db() -> Iterator[Session]:
        with session_factory() as db:
            yield db

    def current_user(
        token: Annotated[str, Depends(oauth2)], db: Annotated[Session, Depends(get_db)]
    ) -> User:
        unauthorized = HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
        try:
            user_id = read_access_token(token, jwt_secret)
        except InvalidTokenError as exc:
            raise unauthorized from exc
        user = db.get(User, user_id)
        if user is None:
            raise unauthorized
        return user

    Rag = Annotated[RagService, Depends(get_service)]
    Db = Annotated[Session, Depends(get_db)]
    CurrentUser = Annotated[User, Depends(current_user)]

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    # --- Accounts -------------------------------------------------------------

    @api.post("/auth/register", status_code=status.HTTP_201_CREATED)
    def register(body: RegisterIn, db: Db) -> UserOut:
        if db.scalar(select(User).where(User.username == body.username)):
            raise HTTPException(status.HTTP_409_CONFLICT, "Username is already taken")
        user = User(username=body.username, password_hash=hash_password(body.password))
        db.add(user)
        db.commit()
        return UserOut.model_validate(user)

    @api.post("/auth/login")
    def login(form: Annotated[OAuth2PasswordRequestForm, Depends()], db: Db) -> TokenOut:
        user = db.scalar(select(User).where(User.username == form.username))
        # Same message for unknown user and wrong password, so usernames can't be probed.
        if user is None or not verify_password(form.password, user.password_hash):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Wrong username or password")
        return TokenOut(
            access_token=create_access_token(user.id, jwt_secret, settings.jwt_expire_minutes)
        )

    @api.get("/auth/me")
    def me(user: CurrentUser) -> UserOut:
        return UserOut.model_validate(user)

    # --- Documents (shared knowledge base; signing in required) -----------------

    @api.post("/documents", status_code=status.HTTP_201_CREATED)
    def upload_document(file: UploadFile, rag: Rag, _: CurrentUser) -> UploadOut:
        max_bytes = settings.max_upload_mb * 1024 * 1024
        data = file.file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(413, f"File is over {settings.max_upload_mb} MB")
        try:
            added = rag.add_document(file.filename or "upload.txt", data)
        except (UnsupportedFileError, EmptyDocumentError) as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
        return UploadOut(**asdict(added))

    @api.get("/documents")
    def list_documents(rag: Rag, _: CurrentUser) -> list[DocumentOut]:
        return [DocumentOut(**asdict(doc)) for doc in rag.list_documents()]

    @api.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_document(doc_id: str, rag: Rag, _: CurrentUser) -> None:
        if not rag.delete_document(doc_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")

    # --- Questions and history (private to each user) ---------------------------

    def save_answer(db: Session, user_id: int, question: str, result: AskOut) -> None:
        db.add(
            ChatMessage(
                user_id=user_id,
                question=question,
                answer=result.answer,
                citations=[c.model_dump() for c in result.citations],
                grounded=result.grounded,
                mode=result.mode.value,
            )
        )
        db.commit()

    @api.post("/ask")
    def ask(body: AskIn, rag: Rag, user: CurrentUser, db: Db) -> AskOut:
        try:
            answer = rag.ask(body.question, body.top_k, body.mode)
        except LLMError as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
        result = AskOut(**asdict(answer))
        save_answer(db, user.id, body.question, result)
        return result

    @api.post("/ask/stream")
    def ask_stream(body: AskIn, rag: Rag, user: CurrentUser) -> StreamingResponse:
        """Newline-delimited JSON: {"type": "delta", "text"} lines while the model writes,
        then one {"type": "answer", ...AskOut} line, or {"type": "error", "detail"}."""
        user_id = user.id

        def events() -> Iterator[str]:
            try:
                for item in rag.ask_stream(body.question, body.top_k, body.mode):
                    if isinstance(item, str):
                        yield json.dumps({"type": "delta", "text": item}) + "\n"
                        continue
                    result = AskOut(**asdict(item))
                    # The request's own session is closed once streaming starts.
                    with session_factory() as db:
                        save_answer(db, user_id, body.question, result)
                    yield json.dumps({"type": "answer", **result.model_dump(mode="json")}) + "\n"
            except LLMError as exc:
                yield json.dumps({"type": "error", "detail": str(exc)}) + "\n"

        return StreamingResponse(events(), media_type="application/x-ndjson")

    @api.get("/history")
    def history(user: CurrentUser, db: Db, limit: int = 50) -> list[HistoryItemOut]:
        rows = db.scalars(
            select(ChatMessage)
            .where(ChatMessage.user_id == user.id)
            .order_by(ChatMessage.id.desc())
            .limit(min(max(limit, 1), 200))
        )
        return [HistoryItemOut.model_validate(row) for row in rows]

    @api.delete("/history", status_code=status.HTTP_204_NO_CONTENT)
    def clear_history(user: CurrentUser, db: Db) -> None:
        db.execute(delete(ChatMessage).where(ChatMessage.user_id == user.id))
        db.commit()

    # --- Evaluation results (public, for the dashboard) -------------------------

    @api.get("/evaluation")
    def evaluation() -> EvaluationOut:
        return EvaluationOut(
            retrieval=_read_json(settings.eval_results_dir / "retrieval.json"),
            answers=_read_json(settings.eval_results_dir / "answers.json"),
        )

    app.include_router(api)

    # The React app, when built. Mounted last so /api routes take precedence.
    if (settings.frontend_dist / "index.html").exists():
        app.mount("/", StaticFiles(directory=settings.frontend_dist, html=True), name="frontend")

    return app


app = create_app()
