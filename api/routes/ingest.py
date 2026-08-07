import os
import shutil
import tempfile
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from core.utils.errors import friendly_error
from services.rag.ingestion.service import IngestionService
from shared.schemas.ingest import IngestResponse

router = APIRouter(prefix="/ingest")

# FastAPI's Swagger "Try it out" UI pre-fills every Optional[str] Form field
# with the literal text "string" as its example value. Someone clicking
# Execute without clearing that field creates a real, permanent Milvus
# collection named "string" — invisible to chat, since chat searches the
# configured default collection. This happened once already (see
# project_context.md, 2026-08-07). Treat this one known placeholder value —
# and blank/whitespace-only input — as "not provided" rather than a real name.
_PLACEHOLDER_COLLECTION_NAMES = {"string"}


def _sanitize_collection_name(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    cleaned = raw.strip()
    if not cleaned or cleaned in _PLACEHOLDER_COLLECTION_NAMES:
        return None
    return cleaned


@router.post("/upload", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    collection_name: Optional[str] = Form(
        None,
        description=(
            "Target Milvus collection. Leave empty to use the server default "
            "(the one chat searches automatically) — only set this if you "
            "deliberately want to file this document under a separate, "
            "explicitly-named collection."
        ),
    ),
):
    """
    Ingest a document into the system.
    Parses, chunks, summarizes, embeds, and stores it in Milvus.
    """
    collection_name = _sanitize_collection_name(collection_name)

    # Save the uploaded file to a temporary location
    try:
        suffix = os.path.splitext(file.filename)[1] if file.filename else ".txt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")
    finally:
        file.file.close()

    try:
        service = IngestionService()
        stats = await service.ingest_file(
            file_path=tmp_path,
            collection_name=collection_name,
            source_name=file.filename,
        )
        return IngestResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=friendly_error(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
