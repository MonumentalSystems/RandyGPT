#!/usr/bin/env python3
"""
clean_gutenberg_v3.py — Further improved Gutenberg cleaner.

Changes vs v2:
  - Strips image filename references (p003.jpg, bookcover.jpg (152K), etc.)
  - Strips "Full Size" HTML image link artifacts
  - Fixes footnote ref matching for 3-digit refs ([116], [261], etc.)
  - Fixes "End of the Project Gutenberg EBook" variant not caught by v2
  - Reduces repeated chars to 1 instead of 2 (seeeee → se, hissss → his)
  - Strips acute/prime backtick variants (′ ´ `) more thoroughly
  - Strips standalone dash/dot/underscore lines more reliably
  - Strips lines with angle-bracket HTML artifacts (<i>, </div>, etc.)
  - Strips transcription notes (Transcriber's note:, Editor's note:)
  - All v2 cleaning retained

Usage:
    python3 scripts/clean_gutenberg_v3.py gutenberg_train.txt train_v3.txt
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

    # v3: image filename references (.jpg, .png, etc.) with optional size
    re.compile(r'^\s*\S+\.(jpg|jpeg|png|gif|svg|webp|bmp|tiff?)\s*(\([0-9.]+\s*[KMG]?\))?\s*$', re.I),

    # v3: "Full Size" HTML image link artifacts
    re.compile(r'^\s*Full\s+Size\s*$', re.I),

    # v3: HTML tag lines (<i>, </div>, <br/>, etc.)
    re.compile(r'^\s*<[a-z/][^>]{0,40}>\s*$', re.I),

    # v3: Transcriber/Editor notes
    re.compile(r'^\s*(Transcriber|Editor|Translator)\'?s?\s+[Nn]ote', re.I),

    # v3: "End of the Project Gutenberg EBook" — catches variant v2 missed
    re.compile(r'^\s*End of (the )?Project Gutenberg', re.I),

    # Bible / scripture verse references: 1:2, 001:002:003, Gen. 3:16
    re.compile(r'^\s*\d{1,3}:\d{2,3}(:\d{2,3})?\s*$'),
    re.compile(r'^\s*[A-Z][a-z]{1,5}\.\s+\d+:\d+'),

    # Pure number lines (page numbers, line numbers, catalogue refs)
    re.compile(r'^\s*[\d\s\.\-,]+\s*$'),

    # Lines that are only punctuation / underscores / symbols
    re.compile(r'^\s*[^a-zA-Z0-9]{4,}\s*$'),

    # v3: standalone repeated-punctuation dividers (catches spaced variants)
    re.compile(r'^\s*[\.\-_\*]{2,}(\s*[\.\-_\*]{1,}){2,}\s*$'),

    # Footnote/reference markers — any-length numeric [1], [116], [2048], symbol [*], [†]
    re.compile(r'^\s*[\[\({][0-9]+[\]\)}]\.?\s*$'),
    re.compile(r'^\s*[\[\({][*†‡§¶]{1,3}[\]\)}]\.?\s*$'),

    # ALL-CAPS lines (headers, scene headings, chapter titles in plays)
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
    # Footnote markers inline — any-length numeric [1], [116], [2048]; symbols [*], [†]
    # Does NOT match word refs like [Exit] or [Chorus] (those contain letters)
    (re.compile(r'[\[\({][0-9]+[\]\)}]'), ''),
    (re.compile(r'[\[\({][*†‡§¶]{1,3}[\]\)}]'), ''),
    # Scripture refs inline: "see Gen. 3:16" → "see Gen."
    (re.compile(r'\b\d{1,3}:\d{2,3}(:\d{2,3})?'), ''),
    # Gutenberg project references
    (re.compile(r'(End of )?(the )?Project Gutenberg[^\n]{0,80}', re.I), ''),
    # Italic stage direction markup: [_Exit_] → [Exit], _word_ → word
    (re.compile(r'\[_(.*?)_\]'), r'[\1]'),
    (re.compile(r'\b_([\w\s,;:]+?)_\b'), r'\1'),
    # v3: backtick and variant OCR artifacts (`, ′, ´)
    (re.compile(r'[`′´]'), ''),
    # v3: reduce repeated chars to max 2 (seeeee → se, hissss → his)
    # Vowels: keep 2 (seee → see is natural, seeee → see)
    (re.compile(r'([aeiouAEIOU])\1{2,}'), r'\1\1'),
    # Consonants: keep 1 (frssss → frs, nnnn → n)
    (re.compile(r'([^aeiouAEIOU\s])\1{3,}'), r'\1'),
    # Tab characters → single space
    (re.compile(r'\t+'), ' '),
    # Email addresses inline
    (re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'), ''),
    # v3: inline HTML tags
    (re.compile(r'<[a-z/][^>]{0,40}>', re.I), ''),
]


# ── Unicode → ASCII character normalization ────────────────────────────────────
# Applied per-line before any pattern matching so downstream regexes only
# see plain ASCII punctuation.
_UNICODE_XLAT = str.maketrans({
    # Smart double quotes → "
    '\u201c': '"',  # "  left double quotation mark
    '\u201d': '"',  # "  right double quotation mark
    '\u201e': '"',  # „  double low-9 quotation mark
    '\u201f': '"',  # ‟  double high-reversed-9 quotation mark
    '\u00ab': '"',  # «  left-pointing double angle quotation mark
    '\u00bb': '"',  # »  right-pointing double angle quotation mark
    # Smart single quotes / apostrophes → '
    '\u2018': "'",  # '  left single quotation mark
    '\u2019': "'",  # '  right single quotation mark
    '\u201a': "'",  # ‚  single low-9 quotation mark
    '\u201b': "'",  # ‛  single high-reversed-9 quotation mark
    '\u2039': "'",  # ‹  single left-pointing angle quotation mark
    '\u203a': "'",  # ›  single right-pointing angle quotation mark
    '\u02bc': "'",  # ʼ  modifier letter apostrophe
    '\u02bb': "'",  # ʻ  modifier letter turned comma
    # Dashes
    '\u2013': '-',   # –  en dash
    '\u2014': '--',  # —  em dash
    '\u2015': '--',  # ―  horizontal bar
    # Ellipsis
    '\u2026': '...',  # …  horizontal ellipsis
    # Whitespace variants
    '\u00a0': ' ',   # non-breaking space
    '\u202f': ' ',   # narrow no-break space
    '\u2009': ' ',   # thin space
    # Miscellaneous typographic symbols
    '\u2022': '*',   # •  bullet
    '\u2032': "'",   # ′  prime (often misused as apostrophe)
    '\u2033': '"',   # ″  double prime
})


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
        # Normalize Unicode punctuation to plain ASCII
        line = line.translate(_UNICODE_XLAT)
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

    print("Cleaning noise lines (v3)...", file=sys.stderr)
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
