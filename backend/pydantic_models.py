from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ModelName(str, Enum):
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite"


class QueryInput(BaseModel):
    question: str = Field(..., description="Question to answer.")
    session_id: Optional[str] = Field(
        default=None,
        description="Session identifier. A new one is generated if omitted.",
    )
    model: ModelName = Field(
        default=ModelName.GEMINI_2_5_FLASH,
        description="Gemini model to use.",
    )


class QueryResponse(BaseModel):
    answer: str = Field(..., description="Generated answer.")
    session_id: str = Field(..., description="Session identifier.")
    model: ModelName = Field(..., description="Model used.")


class DocumentInfo(BaseModel):
    id: int = Field(..., description="Document ID.")
    filename: str = Field(..., description="Original file name.")
    upload_timestamp: datetime = Field(..., description="Upload time.")


class DeleteFileRequest(BaseModel):
    file_id: int = Field(..., description="Document ID to delete.")