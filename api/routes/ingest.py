import os
import shutil
import tempfile
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from core.utils.errors import friendly_error
from services.rag.ingestion.service import IngestionService
from shared.schemas.ingest import IngestResponse

router = APIRouter(prefix="/ingest")


@router.post("/upload", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    collection_name: Optional[str] = Form(None),
):
    """
    Ingest a document into the system.
    Parses, chunks, summarizes, embeds, and stores it in Milvus.
    """
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
