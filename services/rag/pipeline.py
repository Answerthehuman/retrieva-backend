import json
from typing import Any

from ..core.config import Settings
from ..libs.sse import format_sse_event

try:
    from pymilvus import connections, utility, Collection
except ImportError:
    connections = None
    utility = None
    Collection = None


class RAGPipeline:
    def __init__(self, settings: Settings = Settings()):
        self.settings = settings
        self.collection_name = settings.milvus_default_collection
        self._connect_milvus()

    def _connect_milvus(self) -> None:
        if connections is None:
            return

        connections.connect(
            alias="default",
            host=self.settings.milvus_host,
            port=self.settings.milvus_port,
        )

    def _load_bm25_stats(self) -> Any:
        # Placeholder for hybrid BM25 stats loading from Redis
        return None

    def retrieve(self, query_text: str) -> list[dict]:
        if utility is None:
            return [
                {
                    "id": "demo-1",
                    "title": "Retrieva demo document",
                    "content": "Retrieva can answer questions by combining vector retrieval and generation.",
                    "source": "demo",
                }
            ]

        if not utility.has_collection(self.collection_name):
            return []

        collection = Collection(self.collection_name)
        search_params = {"metric_type": "L2", "params": {"nprobe": 8}}
        results = collection.search(
            data=[query_text],
            anns_field="embeddings",
            param=search_params,
            limit=3,
            expr=None,
            output_fields=["title", "content", "source"],
        )

        documents = []
        for hits in results:
            for hit in hits:
                documents.append({
                    "id": str(hit.id),
                    "title": hit.entity.get("title"),
                    "content": hit.entity.get("content"),
                    "source": hit.entity.get("source"),
                })
        return documents

    def rerank(self, documents: list[dict], query_text: str) -> list[dict]:
        # Placeholder for cross-encoder reranking.
        return documents

    def generate(self, query_text: str, documents: list[dict]) -> list[str]:
        prompt = self._build_prompt(query_text, documents)
        return [f"Retrieva has retrieved {len(documents)} document(s). ", f"Answering your question: {query_text}"]

    def _build_prompt(self, query_text: str, documents: list[dict]) -> str:
        docs_text = "\n\n".join([f"- {doc['title']}: {doc['content']}" for doc in documents])
        return f"Use retrieved passages to answer this user query:\n{query_text}\n\nRetrieved documents:\n{docs_text}"

    def query(self, query_text: str) -> dict:
        documents = self.retrieve(query_text)
        ranked_documents = self.rerank(documents, query_text)
        response = self.generate(query_text, ranked_documents)
        return {"documents": ranked_documents, "response": response}
