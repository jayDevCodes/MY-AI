from pydantic import BaseModel, Field


class Document(BaseModel):
    source: str = Field(min_length=1)
    text: str = Field(min_length=1)


class RetrievedChunk(BaseModel):
    source: str
    text: str
    score: float
    chunk_index: int
