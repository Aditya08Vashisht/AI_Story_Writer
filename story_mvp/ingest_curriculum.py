"""Curriculum PDF ingestion for StoryTutor-MM.

This module converts NCERT-style PDF folders into JSON chunks that the RAG
engine can index. It is intentionally offline and free-first.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from story_mvp.deva_quality import (
    UNUSABLE_THRESHOLD,
    best_text_extractor,
    normalize_devanagari,
)

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - handled at runtime for optional setup
    PdfReader = None

# Chunks shorter than this carry no retrievable meaning and only add index
# noise. The pre-fix corpus contained chunks as short as 3 characters.
MIN_CHUNK_CHARS = 200


LANGUAGE_ALIASES = {
    "english": "english",
    "hindi": "hindi",
    "marathi": "marathi",
}


@dataclass(frozen=True)
class PdfMetadata:
    class_level: str
    subject: str
    language: str
    chapter: str
    source_file: str


def ingest_curriculum_pdfs(
    source_dir: str,
    output_dir: str,
    chunk_chars: int = 1800,
    overlap_chars: int = 250,
    limit: int = 0,
    checkpoint_every: int = 10,
    workers: int = 1,
) -> Dict[str, object]:
    """Extract PDF text into processed curriculum JSON chunks."""
    if PdfReader is None:
        raise RuntimeError("pypdf is required. Install dependencies with `pip install -r requirements.txt`.")

    pdf_paths = list(_iter_pdf_paths(source_dir))
    # Refuse to "succeed" with nothing. Writing an empty corpus over a good one
    # is silent data loss: downstream steps then rebuild an empty index and an
    # empty golden set, and every failure looks like a different problem.
    # The usual cause is that the PDFs were never transferred (they are
    # gitignored at 2.5 GB), so say that rather than emitting zero chunks.
    if not pdf_paths:
        raise RuntimeError(
            f"No PDFs found under {os.path.abspath(source_dir)}.\n"
            "Refusing to overwrite the existing corpus with an empty one.\n"
            "The NCERT PDFs are gitignored (2.5 GB) and must be transferred "
            "separately, or the corpus rebuilt on a machine that has them."
        )
    if limit > 0:
        pdf_paths = pdf_paths[:limit]
    output_chunks = []
    os.makedirs(output_dir, exist_ok=True)
    chunks_path = os.path.join(output_dir, "curriculum_chunks.json")
    audit_path = os.path.join(output_dir, "ingestion_audit.json")
    audit = {
        "source_dir": os.path.abspath(source_dir),
        "output_dir": os.path.abspath(output_dir),
        "pdf_count": len(pdf_paths),
        "limited_run": limit > 0,
        "chunks_written": 0,
        "coverage": {},
        "errors": [],
    }

    work_items = [
        (pdf_path, _metadata_from_path(pdf_path), chunk_chars, overlap_chars)
        for pdf_path in pdf_paths
    ]
    if workers > 1:
        executor = ProcessPoolExecutor(max_workers=workers)
        extracted_items = executor.map(_extract_pdf_chunks, work_items)
    else:
        executor = None
        extracted_items = map(_extract_pdf_chunks, work_items)

    try:
        for pdf_number, (pdf_path, metadata, chunks, error) in enumerate(extracted_items, 1):
            coverage_key = f"class_{metadata.class_level}:{metadata.subject}:{metadata.language}"
            audit["coverage"][coverage_key] = audit["coverage"].get(coverage_key, 0) + 1

            if error:
                audit["errors"].append({"source_file": pdf_path, "error": error})
            else:
                output_chunks.extend(chunks)

            if pdf_number % max(checkpoint_every, 1) == 0:
                audit["chunks_written"] = len(output_chunks)
                audit["missing_expected_coverage"] = _missing_expected_coverage(audit["coverage"])
                _write_json(chunks_path, output_chunks)
                _write_json(audit_path, audit)
            print(f"Processed {len(output_chunks)} chunks from {coverage_key}")
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)

    audit["chunks_written"] = len(output_chunks)
    audit["missing_expected_coverage"] = _missing_expected_coverage(audit["coverage"])
    _write_json(chunks_path, output_chunks)
    _write_json(audit_path, audit)

    return audit


def _iter_pdf_paths(source_dir: str) -> Iterable[str]:
    for root, _, filenames in os.walk(source_dir):
        for filename in filenames:
            if filename.lower().endswith(".pdf"):
                yield os.path.join(root, filename)


def _metadata_from_path(pdf_path: str) -> PdfMetadata:
    normalized = pdf_path.replace("\\", "/")
    parts = [part.lower() for part in normalized.split("/")]
    folder_text = " ".join(parts)
    filename = os.path.basename(pdf_path)

    class_match = re.search(r"class\s*([6-8])|class([6-8])", folder_text)
    class_level = class_match.group(1) or class_match.group(2) if class_match else "unknown"

    if "sst" in folder_text or "social" in folder_text:
        subject = "social_science"
    elif "science" in folder_text:
        subject = "science"
    else:
        subject = "unknown"

    language = "unknown"
    for alias, canonical in LANGUAGE_ALIASES.items():
        if alias in folder_text:
            language = canonical
            break

    chapter_match = re.search(r"(?:ch-|ch|f[hm]?[a-z]*)(\d{2,3})", filename.lower())
    chapter = chapter_match.group(1).lstrip("0") if chapter_match else os.path.splitext(filename)[0]

    return PdfMetadata(
        class_level=class_level,
        subject=subject,
        language=language,
        chapter=chapter,
        source_file=os.path.abspath(pdf_path),
    )


def _chunks_from_pdf(
    pdf_path: str,
    metadata: PdfMetadata,
    chunk_chars: int,
    overlap_chars: int,
) -> List[Dict[str, object]]:
    # Pick the extractor that produces the least-damaged Devanagari for THIS
    # file. pypdf mangles most Hindi/Marathi NCERT PDFs (damage 80-240) where
    # PyMuPDF reads them cleanly (3-30) -- see story_mvp/deva_quality.py.
    pages, extractor, file_damage = best_text_extractor(pdf_path)
    unusable = file_damage >= UNUSABLE_THRESHOLD
    chunks = []

    for page_index, page_text in enumerate(pages, 1):
        text = _clean_text(page_text)
        # Drop fragments too short to carry meaning. The previous corpus had
        # chunks as short as 3 characters, which only added index noise.
        if len(text) < MIN_CHUNK_CHARS:
            continue
        for chunk_index, chunk_text in enumerate(_split_text(text, chunk_chars, overlap_chars), 1):
            if len(chunk_text) < MIN_CHUNK_CHARS:
                continue
            chunks.append(
                {
                    "id": _chunk_id(metadata, page_index, chunk_index),
                    "mode": "expand",
                    "genre": "thriller",
                    "tone": "cinematic",
                    "language": metadata.language,
                    "class_level": metadata.class_level,
                    "subject": metadata.subject,
                    "chapter": metadata.chapter,
                    "topic": "",
                    "chunk_type": "curriculum_page",
                    "page_start": page_index,
                    "page_end": page_index,
                    "source_file": metadata.source_file,
                    "text": chunk_text,
                    # Provenance for the extraction itself, so a bad chunk can
                    # be traced to the extractor that produced it.
                    "extractor": extractor,
                    "extraction_damage": file_damage,
                    "extraction_unusable": unusable,
                    "tags": [metadata.subject, metadata.language, f"class-{metadata.class_level}"],
                    "metadata": {
                        "class_level": metadata.class_level,
                        "subject": metadata.subject,
                        "language": metadata.language,
                        "chapter": metadata.chapter,
                        "source_file": metadata.source_file,
                    },
                }
            )

    return chunks


def _extract_pdf_chunks(
    work_item: Tuple[str, PdfMetadata, int, int],
) -> Tuple[str, PdfMetadata, List[Dict[str, object]], str]:
    pdf_path, metadata, chunk_chars, overlap_chars = work_item
    try:
        return pdf_path, metadata, _chunks_from_pdf(pdf_path, metadata, chunk_chars, overlap_chars), ""
    except Exception as exc:
        return pdf_path, metadata, [], str(exc)


def _split_text(text: str, chunk_chars: int, overlap_chars: int) -> Iterable[str]:
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        yield text[start:end].strip()
        if end == len(text):
            break
        start = max(0, end - overlap_chars)


def _clean_text(text: str) -> str:
    # Collapse the duplicated combining marks that glyph-level extraction
    # leaves behind before any downstream step sees the text.
    text = normalize_devanagari(str(text or ""))
    return re.sub(r"\s+", " ", text).strip()


def _chunk_id(metadata: PdfMetadata, page_index: int, chunk_index: int) -> str:
    source = os.path.splitext(os.path.basename(metadata.source_file))[0]
    return f"{metadata.class_level}_{metadata.subject}_{metadata.language}_{source}_p{page_index}_c{chunk_index}"


def _missing_expected_coverage(coverage: Dict[str, int]) -> List[str]:
    missing = []
    for class_level in ("6", "7", "8"):
        for subject in ("science", "social_science"):
            for language in ("english", "hindi", "marathi"):
                key = f"class_{class_level}:{subject}:{language}"
                if key not in coverage:
                    missing.append(key)
    return missing


def _write_json(path: str, payload, attempts: int = 5) -> None:
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # os.replace is atomic on POSIX, but on Windows it fails with
    # PermissionError when anything else holds the destination open -- an
    # antivirus scanner or search indexer sweeping a freshly written 25 MB
    # file is enough. Those locks are transient, so retry before giving up.
    last_error = None
    for attempt in range(attempts):
        try:
            os.replace(temp_path, path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.5 * (attempt + 1))

    raise RuntimeError(
        f"Could not replace {path} after {attempts} attempts: {last_error}. "
        f"The new data is intact at {temp_path} -- close any program holding "
        f"the file open (editor, viewer) and rename it manually."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest NCERT PDFs into StoryTutor-MM JSON chunks.")
    parser.add_argument("--source", default=os.path.join("story_datasets", "NCERT_6th-8th"))
    parser.add_argument("--output", default=os.path.join("knowledge_base", "processed"))
    parser.add_argument("--chunk-chars", type=int, default=1800)
    parser.add_argument("--overlap-chars", type=int, default=250)
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N PDFs for smoke testing.")
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Write resumable output after every N PDFs.")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(os.cpu_count() or 1, 4),
        help="Parallel PDF extraction workers (default: up to 4).",
    )
    args = parser.parse_args()

    audit = ingest_curriculum_pdfs(
        args.source,
        args.output,
        args.chunk_chars,
        args.overlap_chars,
        args.limit,
        args.checkpoint_every,
        args.workers,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
