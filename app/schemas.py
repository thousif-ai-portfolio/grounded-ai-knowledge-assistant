from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.retrieval import SearchMode


class RegisterIn(BaseModel):
    username: str = Field(min_length=3, max_length=32, pattern=r"^[A-Za-z0-9_]+$")
    password: str = Field(min_length=8, max_length=128)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


class DocumentOut(BaseModel):
    doc_id: str
    filename: str
    pages: int
    chunks: int


class UploadOut(BaseModel):
    doc_id: str
    filename: str
    chunks: int
    already_indexed: bool


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    mode: SearchMode | None = None


class CitationOut(BaseModel):
    number: int
    filename: str
    page: int
    text: str
    score: float


class AskOut(BaseModel):
    answer: str
    citations: list[CitationOut]
    grounded: bool
    mode: SearchMode
    retrieval_ms: float
    generation_ms: float
    input_tokens: int
    output_tokens: int


class HistoryItemOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    question: str
    answer: str
    citations: list[CitationOut]
    grounded: bool
    mode: SearchMode
    created_at: datetime


class EvaluationOut(BaseModel):
    """Raw JSON reports written by `python -m evaluation.run`; null if not run yet."""

    retrieval: dict | None
    answers: dict | None
