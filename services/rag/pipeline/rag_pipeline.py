from typing import List

try:
    from pymilvus import connections, utility, Collection
except ImportError:
    connections = None
    utility = None
    Collection = None


class RAGPipeline:
    def __init__(self, collection_name: str = "retrieva_docs"):
        self.collection_name = collection_name
        self._connect_milvus()

    def _connect_milvus(self) -> None:
        if connections is None:
            return
        connections.connect(
            alias="default",
            host="localhost",
            port="19530",
        )

    def retrieve(self, query_text: str) -> List[dict]:
        if utility is None:
            return [
                {
                    "id": "demo-1",
                    "title": "Retrieva demo document",
                    "content": "Retrieva can answer questions by combining vector retrieval and generation.",
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
            output_fields=["title", "content"],
        )

        documents = []
        for hits in results:
            for hit in hits:
                documents.append({
                    "id": str(hit.id),
                    "title": hit.entity.get("title"),
                    "content": hit.entity.get("content"),
                })
        return documents

    def rerank(self, documents: List[dict], query_text: str) -> List[dict]:
        return documents

    def generate(self, query_text: str, documents: List[dict]) -> List[str]:
        return [
            f"Retrieva retrieved {len(documents)} document(s).",
            f"Answer: {query_text}",
        ]

    def query(self, query_text: str) -> dict:
        documents = self.retrieve(query_text)
        ranked = self.rerank(documents, query_text)
        response = self.generate(query_text, ranked)
        return {"documents": ranked, "response": response}
