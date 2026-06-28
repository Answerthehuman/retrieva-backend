import json
import logging
from typing import AsyncGenerator, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage

from core.config.settings import get_settings
from core.providers import get_embeddings, get_llm, get_reranker, get_retriever

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Production-grade RAG pipeline integrating pyzo-ai-core retrieval and LLM generation."""

    def __init__(self):
        self.settings = get_settings()
        self.llm = get_llm(self.settings)
        self.retriever = get_retriever(self.settings)
        self.reranker = get_reranker(self.settings)
        self.embeddings = get_embeddings(self.settings)

    async def query(
        self,
        query_text: str,
        collection_name: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        filters: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        End-to-end RAG workflow, yielding SSE events.

        Args:
            query_text: The user's query.
            collection_name: Milvus collection to search.
            chat_history: Previous messages [{"role": "user", "content": "..."}, ...]
            filters: Optional metadata filters.

        Yields:
            JSON-encoded Server-Sent Events (SSE) strings.
        """
        from core.utils.sse import format_sse_event
        
        target_collection = collection_name or self.settings.milvus_default_collection
        
        # 1. Retrieval
        yield format_sse_event("event", {"format": "retrieval_start"})
        
        documents = await self.retrieve(query_text, target_collection, filters=filters)
        
        # Strip vectors from documents before sending to frontend
        clean_docs = []
        for d in documents:
            clean_d = {k: v for k, v in d.items() if k not in ("dense_vector", "sparse_vector", "embedding")}
            clean_docs.append(clean_d)
            
        yield format_sse_event("event", {"format": "retrieval_complete", "documents": clean_docs})

        # 2. Context Building
        context_str = self._build_context(clean_docs)

        # 3. Generation
        yield format_sse_event("event", {"format": "generation_start"})
        
        async for chunk in self.generate(query_text, context_str, chat_history):
            yield format_sse_event("token", {"format": "markdown", "content": chunk})
            
        yield format_sse_event("event", {"format": "generation_complete"})

    async def retrieve(self, query: str, collection_name: str, filters: Optional[str] = None) -> List[Dict]:
        """Execute parallel semantic/hybrid search and reranking."""
        from services.rag.retrieval import retrieve as internal_retrieve
        
        # Simple step generation for single collection
        steps = [{"query": query, "collection": collection_name}]
        
        # Use our internal retrieve function
        docs, _ = await internal_retrieve(
            retrieval_plan=steps,
            standalone_query=query,
            retriever=self.retriever,
            reranker=self.reranker,
            output_fields=["content", "document_summary", "source", "page", "id"],
            filters=filters,
        )
        return docs

    def _build_context(self, documents: List[Dict]) -> str:
        """Format documents into a context string."""
        if not documents:
            return "No context found."
            
        parts = []
        for i, doc in enumerate(documents):
            source = doc.get("source", "Unknown")
            page = doc.get("page", "")
            content = doc.get("content", "")
            
            page_str = f" (Page {page})" if page else ""
            parts.append(f"--- Document {i+1} [{source}{page_str}] ---\n{content}\n")
            
        return "\n".join(parts)

    async def generate(self, query: str, context: str, chat_history: Optional[List[Dict[str, str]]] = None) -> AsyncGenerator[str, None]:
        """Stream an LLM response with injected context."""
        system_prompt = (
            "You are a helpful assistant. Use the following retrieved context to answer the user's question.\n"
            "If the answer is not in the context, say so.\n"
            "Context:\n"
            "{rag_context}"
        )
        
        # Convert simple chat history to LangChain messages
        messages = []
        if chat_history:
            for msg in chat_history:
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    messages.append(AIMessage(content=msg.get("content", "")))
                    
        messages.append(HumanMessage(content=query))
        
        # Provide variables for template substitution
        variables = {
            "rag_context": context,
        }
        
        # We need an async generator that captures the stream
        # pyzo-ai-core's call_llm takes a state dict and streams via LangGraph's event system
        # Since we're not running LangGraph here, we'll invoke the LLM directly with streaming
        
        formatted_system = system_prompt.format(**variables)
        from langchain_core.messages import SystemMessage
        
        full_messages = [SystemMessage(content=formatted_system)] + messages
        
        # Standard langchain stream
        async for chunk in self.llm.astream(full_messages):
            if chunk.content:
                yield chunk.content
