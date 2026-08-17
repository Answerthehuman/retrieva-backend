"""Regression tests for bugs that silently corrupted retrieval.

Both bugs here shipped, ran in production, and reported success while
destroying data. Neither raised an error. They are the strongest argument in
the codebase for having tests at all, so they get first-class coverage.
"""

from core.utils.bm25.generator import SparseVectorGenerator


class TestBM25VocabularyResync:
    """`build_vocabulary()` renumbers every token id on each call.

    It re-sorts the *full* cumulative vocabulary alphabetically and renumbers
    from zero — it does not append new terms at new ids. So ingesting a second
    document whose vocabulary sorts earlier shifts the ids of terms already
    stored in Milvus, silently invalidating every previously-written sparse
    vector. No exception is raised; hybrid search just quietly degrades.
    """

    def test_token_ids_shift_when_vocabulary_grows(self):
        """Pin the underlying behaviour, so a future 'fix' that makes ids stable
        is noticed rather than silently changing the resync contract."""
        gen = SparseVectorGenerator()
        gen.build_vocabulary(["apple banana cherry"], incremental=True)
        before = dict(gen.vocab)

        # "aardvark" sorts before "apple", pushing every existing id up by one.
        gen.build_vocabulary(["aardvark date"], incremental=True)
        after = dict(gen.vocab)

        assert before["apple"] == 0
        assert after["apple"] == 1, "expected ids to shift; resync logic depends on this"
        assert after["aardvark"] == 0

    def test_stale_vector_no_longer_matches_vocabulary(self):
        """Demonstrate the corruption a resync must repair."""
        gen = SparseVectorGenerator()
        gen.build_vocabulary(["apple banana cherry"], incremental=True)
        stale = gen.generate_sparse_vector("apple banana cherry")

        gen.build_vocabulary(["aardvark date"], incremental=True)
        fresh = gen.generate_sparse_vector("apple banana cherry")

        assert set(stale.keys()) != set(fresh.keys()), (
            "a chunk embedded before the vocab grew must no longer match — "
            "this is precisely why ingestion re-embeds pre-existing chunks"
        )

    def test_regenerating_all_vectors_restores_consistency(self):
        """The fix: re-embed every chunk against the current vocabulary."""
        corpus = ["apple banana cherry", "aardvark date elephant"]
        gen = SparseVectorGenerator()
        for doc in corpus:
            gen.build_vocabulary([doc], incremental=True)

        # Every term in every document must map to a live vocabulary id.
        for doc in corpus:
            vector = gen.generate_sparse_vector(doc)
            assert vector, "regenerated vector should not be empty"
            assert all(tid in gen.idf_scores for tid in vector), (
                "regenerated vector references a token id with no IDF score"
            )

    def test_full_rebuild_is_self_consistent(self):
        """A non-incremental rebuild over the whole corpus is always coherent."""
        corpus = ["apple banana", "aardvark date", "zebra apple"]
        gen = SparseVectorGenerator()
        gen.build_vocabulary(corpus, incremental=False)

        for doc in corpus:
            vector = gen.generate_sparse_vector(doc)
            assert all(tid < len(gen.vocab) for tid in vector)


class TestChunkIdUniqueness:
    """Chunk ids were assigned per page, restarting at `chunk_0` each time.

    Milvus upserts by primary key, so page 2's `chunk_0` overwrote page 1's.
    A 3-page document reported `inserted: 10` while storing 4 rows — a silent
    60% data loss that looked like a successful ingest.
    """

    @staticmethod
    def _service(chunk_size=200, chunk_overlap=40):
        """IngestionService with a real chunker but no providers.

        __init__ resolves the LLM and embedding providers, which the
        credit-safety fixture has made unavailable — so bypass it and attach
        only the chunker, which is all chunk_pages touches.
        """
        from services.rag.ingestion.chunker.chunker import RecursiveChunker
        from services.rag.ingestion.service import IngestionService

        service = IngestionService.__new__(IngestionService)
        service.chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return service

    @staticmethod
    def _pages(n=3, repeat=60):
        return [f"Page {i} content. " * repeat for i in range(1, n + 1)]

    def test_ids_unique_across_pages(self):
        service = self._service()
        chunks = service.chunk_pages(self._pages(), doc_id="doc-abc", source="test.pdf")

        ids = [c["id"] for c in chunks]
        assert len(chunks) > 3, "need multiple chunks per page to exercise collisions"
        assert len(ids) == len(set(ids)), (
            f"duplicate chunk ids: {len(ids)} chunks but only {len(set(ids))} unique"
        )

    def test_chunk_index_is_document_global(self):
        service = self._service()
        chunks = service.chunk_pages(self._pages(), doc_id="doc-xyz", source="test.pdf")

        indices = [c["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks))), (
            "chunk_index must run 0..n-1 across the document, not restart per page"
        )

    def test_total_chunks_is_document_wide(self):
        service = self._service()
        chunks = service.chunk_pages(self._pages(), doc_id="d", source="test.pdf")

        assert {c["total_chunks"] for c in chunks} == {len(chunks)}

    def test_page_numbers_preserved(self):
        """Global re-keying must not lose which page a chunk came from."""
        service = self._service()
        chunks = service.chunk_pages(self._pages(n=3), doc_id="d", source="test.pdf")

        assert {c["page"] for c in chunks} == {1, 2, 3}

    def test_ids_namespaced_by_document(self):
        """Two documents must not collide with each other either."""
        service = self._service()
        a = service.chunk_pages(self._pages(n=2), doc_id="doc-a", source="a.pdf")
        b = service.chunk_pages(self._pages(n=2), doc_id="doc-b", source="b.pdf")

        assert not ({c["id"] for c in a} & {c["id"] for c in b})

    def test_single_page_document(self):
        service = self._service()
        chunks = service.chunk_pages(self._pages(n=1), doc_id="d", source="one.txt")

        assert len(chunks) >= 1
        assert len({c["id"] for c in chunks}) == len(chunks)

    def test_empty_pages_produce_no_chunks(self):
        service = self._service()
        assert service.chunk_pages([], doc_id="d", source="empty.txt") == []


class TestRerankerContentField:
    """`rerank()` read `doc["text"]`, but documents carry `content`.

    The resulting KeyError was swallowed by a blanket `except Exception`, so
    reranking silently returned the original order forever while appearing
    enabled.
    """

    async def test_reads_content_field(self):
        from services.rag.retrieval.reranker import Reranker

        class StubModel:
            def predict(self, pairs, batch_size=20):
                # Fails loudly if a pair's text is empty — i.e. if the wrong
                # key was read.
                assert all(text for _, text in pairs), "reranker read an empty/missing field"
                return [1.0 if "Milvus" in text else 0.5 for _, text in pairs]

        reranker = Reranker()
        reranker._model = StubModel()

        docs = [
            {"id": "a", "content": "Retrieva is a hybrid RAG system."},
            {"id": "b", "content": "It uses Milvus for vector search."},
        ]
        ranked = await reranker.rerank("vector search", docs)

        assert [d["id"] for d in ranked] == ["b", "a"]
        assert all("rerank_score" in d for d in ranked)

    async def test_missing_content_degrades_without_raising(self):
        """A malformed document must not take down the whole search."""
        from services.rag.retrieval.reranker import Reranker

        class StubModel:
            def predict(self, pairs, batch_size=20):
                return [0.5] * len(pairs)

        reranker = Reranker()
        reranker._model = StubModel()

        ranked = await reranker.rerank("q", [{"id": "a"}])  # no content key
        assert len(ranked) == 1

    async def test_empty_input_returns_empty(self):
        from services.rag.retrieval.reranker import Reranker

        assert await Reranker().rerank("q", []) == []
