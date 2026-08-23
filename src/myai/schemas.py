from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    conversation: list[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    text: str
    model: str
    version: str
