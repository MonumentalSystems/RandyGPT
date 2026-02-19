#!/usr/bin/env python3
"""
clean_gutenberg_v2.py — Improved Gutenberg cleaner.

Changes vs v1:
  - Excludes known problematic books (Ulysses — extreme stream-of-consciousness)
  - Strips bible/scripture verse references (135:026:010, Gen. 1:1, etc.)
  - Strips lines that are purely numeric / numbering artifacts
  - Strips table of contents / index lines (Chapter I., Act II., etc.)
  - Strips footnote markers ([1], [*], {35}, etc.)
  - Strips lines of only punctuation/symbols
  - Strips ALL-CAPS headers/titles (common in plays, legal texts)
  - Strips lines with excessive digit density (page refs, catalogue numbers)
  - Strips pipe-table lines (| col | col |)
  - Strips email addresses
  - Strips /c envelope markers
  - Strips extended repeated characters (Steeeeeeeee, Frseeeeefronnnng)
  - Normalises italic stage direction markup [_Exit_] → [Exit]
  - Strips backtick OCR artifacts (sce`ne → scene)
  - Normalises tab characters to single space
  - Normalises multiple blank lines to max 2
  - All v1 cleaning retained

Usage:
    python3 scripts/clean_gutenberg_v2.py gutenberg_train.txt train.txt

    # With extra books to replace excluded ones:
    cat gutenberg_train.txt moby_dick.txt > combined.txt
    python3 scripts/clean_gutenberg_v2.py combined.txt train.txt
"""

import re
import sys
import hashlib
from pathlib import Path


# ── Gutenberg markers ─────────────────────────────────────────────────────────

START_RE = re.compile(r'^\*\*\* START OF (THIS|THE) PROJECT GUTENBERG EBOOK', re.I)
END_RE   = re.compile(r'^\*\*\* END OF (THIS|THE) PROJECT GUTENBERG EBOOK',   re.I)

# Books to exclude by title substring (matched against first 2000 chars of book)
EXCLUDED_BOOKS = [
    "ulysses",          # Joyce — extreme stream-of-consciousness, phonetic artifacts
]


# ── Per-line noise patterns (drop entire line) ────────────────────────────────

NOISE_PATTERNS = [
    # v1 patterns
    re.compile(r'^\s*This eBook is for the use of', re.I),
    re.compile(r'^\s*Produced by', re.I),
    re.compile(r'^\s*Transcribed by', re.I),
    re.compile(r'^\s*Prepared by', re.I),
    re.compile(r'^\s*Updated editions will replace', re.I),
    re.compile(r'^\s*www\.gutenberg\.org', re.I),
    re.compile(r'^\s*\[Illustration', re.I),

    # Bible / scripture verse references: 1:2, 001:002:003, Gen. 3:16
    re.compile(r'^\s*\d{1,3}:\d{2,3}(:\d{2,3})?\s*$'),
    re.compile(r'^\s*[A-Z][a-z]{1,5}\.\s+\d+:\d+'),

    # Pure number lines (page numbers, line numbers, catalogue refs)
    re.compile(r'^\s*[\d\s\.\-,]+\s*$'),

    # Lines that are only punctuation / underscores / symbols
    re.compile(r'^\s*[^a-zA-Z0-9]{4,}\s*$'),

    # Footnote / reference markers standing alone: [1], [*], {35}, (a)
    re.compile(r'^\s*[\[\({][0-9a-zA-Z\*†‡§¶]{1,4}[\]\)}]\.?\s*$'),

    # ALL-CAPS lines (headers, scene headings, chapter titles in plays)
    # Only drop if short enough to be a heading (<=60 chars)
    re.compile(r'^\s*[A-Z][A-Z\s\.\,\-\'\!\?]{4,59}\s*$'),

    # Table of contents / index patterns
    re.compile(r'^\s*(Chapter|CHAPTER|Act|ACT|Scene|SCENE|Book|BOOK|Part|PART)'
               r'\s+[IVXLCDM\d]+[\.\s]', re.I),
    re.compile(r'^\s*\d+\.\s+(Chapter|Section|Part|Book)\b', re.I),

    # Trailing page/section refs like "...... 23" or "_________ 7"
    re.compile(r'[\.\-_]{4,}\s*\d+\s*$'),

    # Pipe-table lines: | col | col |
    re.compile(r'^\s*\|.*\|.*\|\s*$'),

    # Email addresses
    re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'),

    # /c envelope markers
    re.compile(r'^\s*/?c/?$'),
    re.compile(r'^\s*c/\s*$'),
]

# Inline patterns: applied to line content (substitution, not full drop)
INLINE_SUBS = [
    # Footnote markers inline: [1], [*], {35}  (but NOT stage directions like [Exit])
    (re.compile(r'[\[\({][0-9\*†‡§¶]{1,4}[\]\)}]'), ''),
    # Scripture refs inline: "see Gen. 3:16" → "see Gen."
    (re.compile(r'\b\d{1,3}:\d{2,3}(:\d{2,3})?'), ''),
    # Gutenberg project references
    (re.compile(r'Project Gutenberg[^\n]{0,80}', re.I), ''),
    # Italic stage direction markup: [_Exit_] → [Exit], _word_ → word
    (re.compile(r'\[_(.*?)_\]'), r'[\1]'),
    (re.compile(r'\b_([\w\s,;:]+?)_\b'), r'\1'),
    # Backtick OCR artifacts: sce`ne → scene
    (re.compile(r'`'), ''),
    # Extended repeated characters: Steeeeeee → Ste, frseeeee → frse
    (re.compile(r'(.)\1{3,}'), r'\1\1'),
    # Tab characters → single space
    (re.compile(r'\t+'), ' '),
    # Email addresses inline
    (re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'), ''),
]


def _digit_density(line: str) -> float:
    if not line:
        return 0.0
    digits = sum(c.isdigit() for c in line)
    return digits / len(line)


def _is_excluded(book_text: str) -> bool:
    sample = book_text[:2000].lower()
    return any(title in sample for title in EXCLUDED_BOOKS)


def extract_books(text: str):
    books, current, inside = [], [], False
    for line in text.splitlines():
        if START_RE.match(line):
            inside, current = True, []
            continue
        if END_RE.match(line):
            if inside and current:
                books.append('\n'.join(current))
            inside, current = False, []
            continue
        if inside:
            current.append(line)
    if inside and current:
        books.append('\n'.join(current))
    return books


def clean_book(text: str) -> str:
    lines = []
    for line in text.splitlines():
        # Drop high digit-density lines (page refs, verse numbers)
        if _digit_density(line.strip()) > 0.30 and len(line.strip()) > 3:
            continue
        # Drop noise pattern lines
        if any(p.search(line) for p in NOISE_PATTERNS):
            continue
        # Apply inline substitutions
        for pattern, replacement in INLINE_SUBS:
            line = pattern.sub(replacement, line)
        line = line.rstrip()
        lines.append(line)

    # Collapse runs of >2 blank lines to 2
    result, blank_run = [], 0
    for line in lines:
        if line.strip() == '':
            blank_run += 1
            if blank_run <= 2:
                result.append(line)
        else:
            blank_run = 0
            result.append(line)

    return '\n'.join(result).strip()


def deduplicate_paragraphs(books):
    seen, result, total_dupes = set(), [], 0
    for book in books:
        paragraphs = re.split(r'\n{2,}', book)
        unique_paras = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(para) >= 60:
                h = hashlib.md5(para.lower().encode()).hexdigest()
                if h in seen:
                    total_dupes += 1
                    continue
                seen.add(h)
            unique_paras.append(para)
        if unique_paras:
            result.append('\n\n'.join(unique_paras))
    print(f"  Duplicate paragraphs removed: {total_dupes}", file=sys.stderr)
    return result


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input> <output>", file=sys.stderr)
        sys.exit(1)

    inp  = Path(sys.argv[1])
    outp = Path(sys.argv[2])

    print(f"Reading {inp} ({inp.stat().st_size / 1e6:.1f} MB)...", file=sys.stderr)
    text = inp.read_text(encoding='utf-8', errors='replace')

    print("Extracting books...", file=sys.stderr)
    books = extract_books(text)
    print(f"  Found {len(books)} books", file=sys.stderr)

    print("Filtering excluded books...", file=sys.stderr)
    before = len(books)
    books = [b for b in books if not _is_excluded(b)]
    excluded = before - len(books)
    print(f"  Excluded {excluded} book(s) ({before} → {len(books)})", file=sys.stderr)

    print("Cleaning noise lines (v2)...", file=sys.stderr)
    books = [clean_book(b) for b in books]
    books = [b for b in books if len(b) > 500]
    print(f"  {len(books)} books after size filter", file=sys.stderr)

    print("Deduplicating paragraphs...", file=sys.stderr)
    books = deduplicate_paragraphs(books)

    total_chars = sum(len(b) for b in books)
    print(f"  Total content: {total_chars / 1e6:.1f} MB across {len(books)} books",
          file=sys.stderr)

    separator = '\n\n<|eos|>\n\n'
    output = separator.join(books)

    outp.write_text(output, encoding='utf-8')
    print(f"Wrote {outp} ({outp.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)


if __name__ == '__main__':
    main()
