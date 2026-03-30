"""
HTTP API для связки с Guard (Next.js): векторный плагиат + оценка AI-признаков.

Запуск: uvicorn api_server:app --host 0.0.0.0 --port 8765
Переменные окружения см. INTEGRATION.md
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from worker import AntiPlagiarismWorker

_worker: Optional[AntiPlagiarismWorker] = None


def _expected_api_key() -> Optional[str]:
    k = os.environ.get("ANALYSIS_API_KEY", "").strip()
    return k or None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _worker
    host = os.environ.get("QDRANT_HOST", "localhost")
    port = int(os.environ.get("QDRANT_PORT", "6333"))
    collection = os.environ.get("QDRANT_COLLECTION", "university_docs")
    _worker = AntiPlagiarismWorker(
        qdrant_host=host,
        qdrant_port=port,
        sqlite_db_path=os.environ.get("ANALYSIS_SQLITE_PATH", "showcase.db"),
        collection_name=collection,
    )
    yield
    _worker = None


app = FastAPI(
    title="Anti-plagiarism ML analysis",
    version="1.0.0",
    lifespan=lifespan,
)


class AnalyzeRequest(BaseModel):
    content: str = Field(..., min_length=1, description="Полный текст документа")
    filename: str = Field(default="document.txt", max_length=512)
    document_id: Optional[int] = Field(default=None, description="ID в БД Guard — только для логов")


class AnalyzeResponse(BaseModel):
    plagiarism_percent: float = Field(..., description="Доля чанков с совпадением в Qdrant, %")
    ai_percent: float = Field(..., description="Средняя оценка 'AI-признаков' по чанкам, %")


def _verify_api_key(x_api_key: Optional[str]) -> None:
    expected = _expected_api_key()
    if expected is None:
        return
    if not x_api_key or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/health")
def health():
    return {"status": "ok", "worker_loaded": _worker is not None}


@app.post("/v1/analyze", response_model=AnalyzeResponse)
def analyze(body: AnalyzeRequest, x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    _verify_api_key(x_api_key)
    if _worker is None:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")
    result = _worker.process_text(body.content, body.filename or "document.txt", verbose=False)
    return AnalyzeResponse(
        plagiarism_percent=float(result["plagiarism_percent"]),
        ai_percent=float(result["ai_percent"]),
    )
