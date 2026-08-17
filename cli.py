"""Retrieva CLI — ingest documents and query the RAG pipeline from the command line."""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Ensure the backend package is importable
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("retrieva.cli")

# File extensions the ingestion pipeline can handle
SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".doc",
    ".pptx",
    ".ppt",
    ".xlsx",
    ".xls",
    ".csv",
    ".txt",
    ".md",
    ".markdown",
}


# ── Ingest ────────────────────────────────────────────────────────────────────


async def _ingest(args: argparse.Namespace) -> None:
    from services.rag.ingestion.service import IngestionService

    service = IngestionService()
    target_collection = args.collection

    path = Path(args.path)
    if not path.exists():
        logger.error("Path does not exist: %s", path)
        sys.exit(1)

    # Collect files to ingest
    files: list[Path] = []
    if path.is_file():
        files.append(path)
    elif path.is_dir():
        for root, _dirs, filenames in os.walk(path):
            for fname in filenames:
                fpath = Path(root) / fname
                if fpath.suffix.lower() in SUPPORTED_EXTENSIONS:
                    files.append(fpath)
        if not files:
            logger.warning("No supported files found in %s", path)
            return
        logger.info("Found %d file(s) to ingest in %s", len(files), path)
    else:
        logger.error("Path is neither a file nor a directory: %s", path)
        sys.exit(1)

    total_inserted = 0
    for i, fpath in enumerate(files, 1):
        logger.info("─── [%d/%d] Ingesting: %s ───", i, len(files), fpath.name)
        try:
            stats = await service.ingest_file(
                file_path=str(fpath),
                collection_name=target_collection,
                source_name=args.source or fpath.name,
            )
            inserted = stats.get("inserted", 0)
            total_inserted += inserted
            logger.info(
                "  ✅ %d chunks inserted into '%s'  |  summary: %.80s…",
                inserted,
                stats.get("collection_name", target_collection),
                stats.get("document_summary", ""),
            )
        except Exception as e:
            logger.error("  ❌ Failed to ingest %s: %s", fpath.name, e, exc_info=True)

    logger.info(
        "═══ Ingestion complete: %d total chunks across %d file(s) ═══", total_inserted, len(files)
    )


# ── Query ─────────────────────────────────────────────────────────────────────


async def _query(args: argparse.Namespace) -> None:
    import json

    from services.rag.pipeline.rag_pipeline import RAGPipeline

    pipeline = RAGPipeline()
    query_text = args.query
    collection = args.collection

    logger.info("Querying collection '%s': %s", collection, query_text)

    async for sse_line in pipeline.query(
        query_text=query_text,
        collection_name=collection,
    ):
        # Each SSE line looks like: data: {"type": "token", "content": "..."}\n\n
        line = sse_line.strip()
        if not line.startswith("data: "):
            continue
        try:
            payload = json.loads(line[6:])
        except json.JSONDecodeError:
            continue

        event_type = payload.get("type", "")
        fmt = payload.get("format", "")

        if event_type == "event":
            if fmt == "retrieval_start":
                query = payload.get("query")
                suffix = f": {query!r}" if query else ""
                print(f"\n🔍 Retrieving documents{suffix}…", flush=True)
            elif fmt == "retrieval_complete":
                docs = payload.get("documents", [])
                print(f"📄 Retrieved {len(docs)} document(s)", flush=True)
                for j, doc in enumerate(docs, 1):
                    source = doc.get("source", "unknown")
                    page = doc.get("page", "")
                    score = doc.get("score", doc.get("hybrid_score", ""))
                    page_str = f" p.{page}" if page else ""
                    score_str = f"  score={score:.4f}" if isinstance(score, (int, float)) else ""
                    print(f"   {j}. {source}{page_str}{score_str}", flush=True)
            elif fmt == "generation_start":
                print("\n💬 Generating response…", flush=True)
            elif fmt == "generation_complete":
                print("\n", flush=True)
        elif event_type == "token":
            # Stream tokens directly to stdout
            content = payload.get("content", "")
            print(content, end="", flush=True)

    print()  # Final newline


# ── Arg parser ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="retrieva",
        description="Retrieva CLI — ingest documents and query the hybrid RAG pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── ingest sub-command ────────────────────────────────────────────────────
    p_ingest = subparsers.add_parser("ingest", help="Ingest documents into a Milvus collection")
    p_ingest.add_argument(
        "--path",
        required=True,
        help="Path to a file or directory containing documents to ingest.",
    )
    p_ingest.add_argument(
        "--collection",
        default=None,
        help="Target Milvus collection name (defaults to MILVUS_DEFAULT_COLLECTION from .env).",
    )
    p_ingest.add_argument(
        "--source",
        default=None,
        help="Source identifier / metadata label for the ingested document(s).",
    )

    # ── query sub-command ─────────────────────────────────────────────────────
    p_query = subparsers.add_parser("query", help="Query the RAG pipeline and stream the response")
    p_query.add_argument(
        "query",
        nargs="+",
        help="The question to ask the RAG pipeline.",
    )
    p_query.add_argument(
        "--collection",
        default=None,
        help="Milvus collection to search (defaults to MILVUS_DEFAULT_COLLECTION from .env).",
    )

    args = parser.parse_args()

    if args.command == "ingest":
        asyncio.run(_ingest(args))
    elif args.command == "query":
        # Join positional words into a single query string
        args.query = " ".join(args.query)
        asyncio.run(_query(args))


if __name__ == "__main__":
    main()
