from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    collection_name: str = Field(..., description="The Milvus collection where data was inserted")
    file_name: str = Field(..., description="The original file name")
    inserted: int = Field(..., description="Number of chunks inserted")
    document_summary: str = Field(..., description="LLM-generated document summary")

    class Config:
        from_attributes = True
