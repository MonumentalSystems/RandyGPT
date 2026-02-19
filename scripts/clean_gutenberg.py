#!/usr/bin/env python3
"""
clean_gutenberg.py — Strip Gutenberg headers/footers, deduplicate paragraphs,
and insert <|eos|> document separators for boundary-aware sampling.

Usage:
    python3 scripts/clean_gutenberg.py gutenberg_train.txt train.txt

Output:
    Clean text with books separated by <|eos|> token.
    Duplicate paragraphs (across all books) are dropped.
"""

import re
import sys
import hashlib
from pathlib import Path


# Lines within a book that are likely license/boilerplate to drop even inside content
NOISE_PATTERNS = [
    re.compile(r'^\s*This eBook is for the use of', re.I),
    re.compile(r'^\s*Produced by', re.I),
    re.compile(r'^\s*Transcribed by', re.I),
    re.compile(r'^\s*Prepared by', re.I),
    re.compile(r'^\s*Updated editions will replace', re.I),
    re.compile(r'^\s*www\.gutenberg\.org', re.I),
    re.compile(r'^\s*\[Illustration', re.I),
]

START_RE = re.compile(r'^\*\*\* START OF (THIS|THE) PROJECT GUTENBERG EBOOK', re.I)
END_RE   = re.compile(r'^\*\*\* END OF (THIS|THE) PROJECT GUTENBERG EBOOK',   re.I)


def extract_books(text):
    """Split flat Gutenberg file into list of raw book content strings."""
    books = []
    current = []
    inside = False

    for line in text.splitlines():
        if START_RE.match(line):
            inside = True
            current = []
            continue
        if END_RE.match(line):
            if inside and current:
                books.append('\n'.join(current))
            inside = False
            current = []
            continue
        if inside:
            current.append(line)

    # Handle file that ends without END marker
    if inside and current:
        books.append('\n'.join(current))

    return books


def clean_book(text):
    """Remove per-line noise from within a book's content."""
    lines = []
    for line in text.splitlines():
        if any(p.match(line) for p in NOISE_PATTERNS):
            continue
        lines.append(line)
    return '\n'.join(lines).strip()


def deduplicate_paragraphs(books):
    """
    Remove duplicate paragraphs across all books.
    Returns list of cleaned book strings with dupes removed.
    Paragraphs shorter than 60 chars are excluded from the seen set
    (they're too short to reliably identify duplicates vs common phrases).
    """
    seen = set()
    result = []
    total_dupes = 0

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

    print("Cleaning noise lines...", file=sys.stderr)
    books = [clean_book(b) for b in books]
    books = [b for b in books if len(b) > 500]  # drop tiny fragments
    print(f"  {len(books)} books after size filter", file=sys.stderr)

    print("Deduplicating paragraphs...", file=sys.stderr)
    books = deduplicate_paragraphs(books)

    total_chars = sum(len(b) for b in books)
    print(f"  Total content: {total_chars / 1e6:.1f} MB across {len(books)} books",
          file=sys.stderr)

    # Join with <|eos|> document separator on its own line
    # The BPE tokenizer already has <|eos|> as token index 1
    separator = '\n\n<|eos|>\n\n'
    output = separator.join(books)

    outp.write_text(output, encoding='utf-8')
    print(f"Wrote {outp} ({outp.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)


if __name__ == '__main__':
    main()
