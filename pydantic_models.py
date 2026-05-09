"""
pydantic_models.py
Request / response schemas shared between the FastAPI backend and any clients.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ModelName(str, Enum):
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite"


class QueryInput(BaseModel):
    question: str = Field(..., description="The question to be answered.")
    session_id: Optional[str] = Field(
        default=None,
        description="Session identifier. A new one is generated if omitted.",
    )
    model: ModelName = Field(
        default=ModelName.GEMINI_2_5_FLASH,
        description="The Gemini model to use.",
    )


class QueryResponse(BaseModel):
    answer: str = Field(..., description="The generated answer.")
    session_id: str = Field(..., description="Session identifier.")
    model: ModelName = Field(..., description="Model used.")


class DocumentInfo(BaseModel):
    id: int = Field(..., description="Unique document ID.")
    filename: str = Field(..., description="Original file name.")
    upload_timestamp: datetime = Field(..., description="Upload time.")


class DeleteFileRequest(BaseModel):
    file_id: int = Field(..., description="ID of the document to delete.")