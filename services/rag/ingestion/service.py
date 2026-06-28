import logging
from typing import Any, Dict, Optional

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from core.config.settings import get_settings
from core.providers import get_embeddings, get_llm
from services.rag.ingestion.parser.parser import NativeParser
from services.rag.ingestion.chunker.chunker import RecursiveChunker
from services.rag.ingestion.vector_store.writer import MilvusWriter

logger = logging.getLogger(__name__)

def _ensure_collection(collection_name: str, dim: int) -> Collection:
    """Create the Milvus collection if it does not exist with the standard schema."""
    if utility.has_collection(collection_name):
        return Collection(collection_name)

    logger.info("Creating Milvus collection: %s", collection_name)
    
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=256),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="document_summary", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="page", dtype=DataType.INT64),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
        FieldSchema(name="total_chunks", dtype=DataType.INT64),
        FieldSchema(name="chunk_size", dtype=DataType.INT64),
        FieldSchema(name="ingested_at", dtype=DataType.VARCHAR, max_length=64),
    ]
    
    schema = CollectionSchema(fields=fields, description="Retrieva Document Store", enable_dynamic_field=True)
    collection = Collection(name=collection_name, schema=schema)
    
    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 200},
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    logger.info("Created index for collection: %s", collection_name)
    
    return collection


class IngestionService:
    """Orchestrates file ingestion: parse → chunk → summarize → embed → store."""
    
    def __init__(self):
        self.settings = get_settings()
        self.llm = get_llm(self.settings)
        self.embeddings = get_embeddings(self.settings)
        
        # Connect to Milvus for schema creation (writer uses standard connection)
        connections.connect(
            alias="default", 
            uri=self.settings.milvus_uri
        )
        
        # Initialize pipeline tools
        self.parser = NativeParser(vision_llm=self.llm)
        self.chunker = RecursiveChunker(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

    async def ingest_file(
        self, 
        file_path: str, 
        collection_name: Optional[str] = None,
        source_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Ingest a file into Milvus.
        
        Args:
            file_path: Path to the local file.
            collection_name: Target Milvus collection (creates it if missing).
            source_name: Original file name or URI for metadata.
            
        Returns:
            Dict containing ingestion statistics.
        """
        target_collection = collection_name or self.settings.milvus_default_collection
        collection = _ensure_collection(target_collection, self.settings.embedding_dim)
        collection.load()
        
        # 1. Parse pages (returns list of strings, one per page/section)
        logger.info("Parsing file: %s", file_path)
        pages = self.parser.parse_pages(file_path)
        
        if not pages:
            logger.warning("No text extracted from %s", file_path)
            return {"inserted": 0, "collection_name": target_collection}
            
        # 2. Chunk
        logger.info("Chunking %d pages", len(pages))
        import uuid
        
        doc_id = str(uuid.uuid4())
        source = source_name or str(file_path)
        
        all_chunks = []
        for i, page_text in enumerate(pages):
            page_chunks = self.chunker.chunk(
                page_text, 
                document_metadata={
                    "document_id": doc_id,
                    "source": source,
                    "page": i + 1
                }
            )
            all_chunks.extend(page_chunks)
            
        logger.info("Generated %d chunks total", len(all_chunks))
            
        # 3. Summarize document
        # Join a sample of the text for summarization to avoid massive context
        sample_text = "\n\n".join(pages)[:20000]
        logger.info("Generating document summary")
        
        from langchain_core.messages import HumanMessage, SystemMessage
        prompt = [
            SystemMessage(content="You are an expert at concisely summarizing documents. Provide a short 2-3 sentence summary of the following document to help with retrieval and cataloging. Focus on the core topic and key findings."),
            HumanMessage(content=sample_text)
        ]
        response = await self.llm.ainvoke(prompt)
        doc_summary = str(response.content)
        logger.info("Document summary: %.100s...", doc_summary)
        
        # 4. Embed and Store
        logger.info("Writing to Milvus collection: %s", target_collection)
        writer = MilvusWriter(
            collection=collection,
            embedding_model=self.embeddings,
            batch_size=100
        )
        
        stats = writer.upsert(all_chunks, document_summary=doc_summary)
        stats["collection_name"] = target_collection
        stats["document_summary"] = doc_summary
        stats["file_name"] = source
        
        return stats
