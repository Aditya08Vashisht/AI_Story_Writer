"""Devanagari extraction-quality scoring.

A PDF stores glyph indices, not characters; a ToUnicode CMap maps them back.
For Latin that is trivially 1:1. For Devanagari it often is not, because the
renderer reorders matras (the vowel sign is drawn BEFORE its consonant but
encoded AFTER) and fuses consonants into conjunct ligatures. When that CMap is
incomplete or wrong, extraction returns glyphs in visual order with broken
clusters, and inter-glyph kerning is misread as word spacing.

The signals below are all structurally impossible in valid Devanagari, so a
nonzero rate means the text layer was damaged during extraction -- not that the
source was bad.

Measured on this corpus: pypdf scores 80-240 damage on most Hindi/Marathi
files; PyMuPDF scores 3-30 on the same files. Hence `best_text_extractor`.
"""

from __future__ import annotations

import re

CONSONANT = r"ऄ-हक़-य़ॸ-ॿ"
MATRA = r"ऺ-ौॎ-ॏॢ-ॣ"
VIRAMA = "्"
SIGNS = r"ऀ-ः़"
DEVA_ANY = r"ऀ-ॿ"

# A matra must attach to a consonant; preceded by whitespace the cluster broke.
ORPHAN_MATRA = re.compile(rf"(?:^|[\s]|[^{CONSONANT}{MATRA}{VIRAMA}{SIGNS}])[{MATRA}]")
# A virama must be followed by a consonant.
ORPHAN_VIRAMA = re.compile(rf"{VIRAMA}(?:\s|$|[^{CONSONANT}])")
# Real Devanagari words are rarely one character long.
SINGLETON = re.compile(rf"(?:^|\s)[{DEVA_ANY}](?=\s|$)")
DEVA_CHAR = re.compile(rf"[{DEVA_ANY}]")

# Above this, the text layer is unusable and the file needs OCR or exclusion.
UNUSABLE_THRESHOLD = 100.0


# PyMuPDF sometimes emits a combining mark twice for certain Devanagari fonts
# -- once for the glyph's visual position and once for its logical one -- so
# "पुरी" comes out as "पुुरी". In valid Devanagari the same matra, anusvara or
# virama never repeats consecutively, so collapsing runs is always safe.
_DOUBLED_MARK = re.compile(rf"([{MATRA}{VIRAMA}{SIGNS}])\1+")


def normalize_devanagari(text: str) -> str:
    """Collapse repeated combining marks left behind by glyph-level extraction."""
    return _DOUBLED_MARK.sub(r"\1", text)


def score_text(text: str) -> dict:
    """Damage score per 1000 Devanagari characters. Clean text scores near 0."""
    deva_chars = len(DEVA_CHAR.findall(text))
    if deva_chars < 20:
        return {"deva_chars": deva_chars, "scored": False, "damage": 0.0}

    orphan_matra = len(ORPHAN_MATRA.findall(text))
    orphan_virama = len(ORPHAN_VIRAMA.findall(text))
    singleton = len(SINGLETON.findall(text))
    per_1k = 1000.0 / deva_chars

    return {
        "deva_chars": deva_chars,
        "scored": True,
        "orphan_matra_per_1k": round(orphan_matra * per_1k, 2),
        "orphan_virama_per_1k": round(orphan_virama * per_1k, 2),
        "singleton_per_1k": round(singleton * per_1k, 2),
        "damage": round((orphan_matra + orphan_virama + singleton) * per_1k, 2),
    }


def damage(text: str) -> float:
    return score_text(text).get("damage", 0.0)


def extract_pages_pymupdf(pdf_path: str) -> list:
    import pymupdf

    with pymupdf.open(pdf_path) as doc:
        return [page.get_text() or "" for page in doc]


def extract_pages_pypdf(pdf_path: str) -> list:
    from pypdf import PdfReader

    return [page.extract_text() or "" for page in PdfReader(pdf_path).pages]


def best_text_extractor(pdf_path: str, sample_pages: int = 10) -> tuple:
    """Pick the extractor that produces the least-damaged Devanagari.

    PyMuPDF wins on 22 of 24 sampled files and never loses, but the two ties
    are catastrophic for both, so this stays adaptive rather than hardcoded --
    and a Latin-only PDF scores 0 either way, where the first successful
    extractor is simply kept.

    Returns (pages, extractor_name, damage_score).
    """
    candidates = []
    for name, fn in (("pymupdf", extract_pages_pymupdf), ("pypdf", extract_pages_pypdf)):
        try:
            pages = fn(pdf_path)
        except Exception:
            continue
        if not pages:
            continue
        sample = " ".join(pages[:sample_pages])
        candidates.append((damage(sample), name, pages))

    if not candidates:
        raise RuntimeError(f"no extractor could read {pdf_path}")

    candidates.sort(key=lambda c: c[0])
    best_damage, best_name, best_pages = candidates[0]
    return best_pages, best_name, best_damage
