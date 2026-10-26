from pydantic import BaseModel, Field


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
