"""Utility script to ingest knowledge-base documents into Pinecone."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from dotenv import load_dotenv
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError

from .app.main import EnhancedVectorStoreManager


logger = logging.getLogger("knowledge_base_ingest")


@dataclass
class ChunkPayload:
    """Represents a single text chunk to be embedded and indexed."""

    chunk_id: str
    text: str
    page_number: int
    chunk_index: int
    source_title: str
    doc_type: str
    jurisdiction: str
    relevance: str
    source_path: str

    def to_upsert_dict(self) -> dict:
        return {
            "id": self.chunk_id,
            "text": self.text,
            "metadata": {
                "source_title": self.source_title,
                "doc_type": self.doc_type,
                "jurisdiction": self.jurisdiction,
                "relevance_to_case_type": self.relevance,
                "page": self.page_number,
                "chunk_index": self.chunk_index,
                "source_path": self.source_path,
            },
        }


def normalise_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> Iterable[str]:
    words = text.split()
    if not words:
        return []

    step = max(chunk_size - overlap, 1)
    chunks: List[str] = []
    for start in range(0, len(words), step):
        segment = words[start : start + chunk_size]
        if not segment:
            continue
        chunks.append(" ".join(segment))
    return chunks


def infer_doc_type(file_path: Path) -> str:
    lower_name = file_path.stem.lower()
    if "petition" in lower_name:
        return "petition_sample"
    if "rules" in lower_name:
        return "rules"
    if "act" in lower_name or "code" in lower_name or "order" in lower_name:
        return "statute"
    return "knowledge_base"


def extract_pdf_chunks(
    file_path: Path,
    chunk_size: int,
    overlap: int,
    jurisdiction: str,
    relevance: str,
) -> List[ChunkPayload]:
    try:
        reader = PdfReader(str(file_path))
    except PdfReadError as exc:
        logger.warning("Skipping %s due to PDF read error: %s", file_path, exc)
        return []

    source_title = file_path.stem.replace("_", " ").replace("-", " ")
    doc_type = infer_doc_type(file_path)
    base_id = hashlib.sha256(str(file_path).encode("utf-8")).hexdigest()[:16]

    chunks: List[ChunkPayload] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            raw_text = page.extract_text() or ""
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to extract page %s from %s: %s", page_number, file_path, exc)
            continue

        cleaned = normalise_whitespace(raw_text)
        if not cleaned:
            continue

        page_chunks = chunk_text(cleaned, chunk_size=chunk_size, overlap=overlap)
        for chunk_index, chunk in enumerate(page_chunks, start=1):
            chunk_id = f"{base_id}-p{page_number:04d}-c{chunk_index:03d}"
            chunks.append(
                ChunkPayload(
                    chunk_id=chunk_id,
                    text=chunk,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    source_title=source_title,
                    doc_type=doc_type,
                    jurisdiction=jurisdiction,
                    relevance=relevance,
                    source_path=str(file_path),
                )
            )

    return chunks


async def ingest_directory(
    directory: Path,
    namespace: str,
    chunk_size: int,
    overlap: int,
    limit: int | None,
) -> None:
    load_dotenv()

    manager = EnhancedVectorStoreManager()
    if manager.index is None:
        raise RuntimeError("Pinecone index is not initialised. Check PINECONE_API_KEY.")

    pdf_files = sorted(directory.glob("**/*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found under %s", directory)
        return

    if limit:
        pdf_files = pdf_files[:limit]

    logger.info("Processing %s PDF files", len(pdf_files))

    for file_path in pdf_files:
        logger.info("Extracting chunks from %s", file_path.name)
        chunk_payloads = extract_pdf_chunks(
            file_path=file_path,
            chunk_size=chunk_size,
            overlap=overlap,
            jurisdiction="Pakistan",
            relevance="general",
        )

        if not chunk_payloads:
            logger.warning("No text extracted from %s", file_path)
            continue

        await manager.upsert_texts(
            [payload.to_upsert_dict() for payload in chunk_payloads],
            namespace=namespace,
        )


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed and ingest knowledge-base PDFs into Pinecone"
    )
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(os.getcwd(), "knowledge base"),
        help="Path to the knowledge base directory containing PDF files",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="knowledge-base",
        help="Pinecone namespace to target",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=400,
        help="Approximate number of words per chunk",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=60,
        help="Word overlap between successive chunks",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally limit the number of documents processed",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    directory = Path(args.directory).expanduser().resolve()
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    logger.info("Starting ingestion from %s", directory)
    asyncio.run(
        ingest_directory(
            directory=directory,
            namespace=args.namespace,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            limit=args.limit,
        )
    )


if __name__ == "__main__":
    main()


