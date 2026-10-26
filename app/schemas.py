from pydantic import BaseModel, Field

from app.retrieval import SearchMode


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
