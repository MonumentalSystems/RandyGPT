#!/usr/bin/env python3
"""
fetch_gists.py — Fetch Rust gists from a GitHub user and build a training corpus.

Fetches all public gists, filters to .rs files, wraps each in <|bos|>/<|eos|>,
and writes to a single training file.

Usage:
    python scripts/fetch_gists.py --user RandyMcMillan --output train_rust.txt
    python scripts/fetch_gists.py --user RandyMcMillan --output train_rust.txt --token ghp_xxx
"""

import argparse
import json
import time
import urllib.request
import urllib.error
from pathlib import Path


MIN_SIZE =   150   # bytes — skip stubs / one-liners
MAX_SIZE = 60_000  # bytes — skip giant files


def fetch_json(url: str, token: str = None) -> dict:
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "randyGPT-fetch-gists/1.0")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def fetch_text(url: str, token: str = None) -> str:
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "randyGPT-fetch-gists/1.0")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", errors="replace")


def fetch_all_gists(user: str, token: str = None) -> list:
    """Fetch all gist metadata via paginated API."""
    gists = []
    page = 1
    while True:
        url = f"https://api.github.com/users/{user}/gists?per_page=100&page={page}"
        batch = fetch_json(url, token)
        if not batch:
            break
        gists.extend(batch)
        print(f"  Page {page}: {len(batch)} gists (total so far: {len(gists)})")
        if len(batch) < 100:
            break
        page += 1
        time.sleep(0.5)  # be polite to the API
    return gists


def extract_rust_files(gists: list) -> list:
    """Extract Rust file metadata from gist list."""
    rust_files = []
    for gist in gists:
        for fname, fmeta in gist["files"].items():
            if fmeta.get("language") == "Rust" or fname.endswith(".rs"):
                size = fmeta.get("size", 0)
                if MIN_SIZE <= size <= MAX_SIZE:
                    rust_files.append({
                        "filename":    fname,
                        "size":        size,
                        "raw_url":     fmeta["raw_url"],
                        "description": gist.get("description", ""),
                        "gist_id":     gist["id"],
                        "html_url":    gist["html_url"],
                    })
    return rust_files


def fetch_corpus(rust_files: list, token: str = None, delay: float = 0.3) -> list:
    """Fetch raw content for each Rust file."""
    samples = []
    for i, f in enumerate(rust_files):
        try:
            content = fetch_text(f["raw_url"], token)
            # Basic sanity: must contain 'fn ' somewhere
            if "fn " not in content and "impl " not in content and "struct " not in content:
                print(f"  [{i+1}/{len(rust_files)}] SKIP (no Rust keywords): {f['filename']}")
                continue
            samples.append({**f, "content": content})
            print(f"  [{i+1}/{len(rust_files)}] OK  {f['filename']} ({f['size']} bytes)")
        except Exception as e:
            print(f"  [{i+1}/{len(rust_files)}] ERR {f['filename']}: {e}")
        time.sleep(delay)
    return samples


def write_corpus(samples: list, output_path: str, bos: str = "<|bos|>", eos: str = "<|eos|>") -> None:
    """Write all samples to a single training file, wrapped in bos/eos."""
    out = Path(output_path)
    total_chars = 0
    with out.open("w", encoding="utf-8") as f:
        for i, s in enumerate(samples):
            content = s["content"].strip()
            # Write header comment + content wrapped in bos/eos
            if s["description"]:
                block = f"{bos}\n// {s['description']}\n// {s['filename']}\n{content}\n{eos}\n"
            else:
                block = f"{bos}\n// {s['filename']}\n{content}\n{eos}\n"
            f.write(block)
            total_chars += len(block)

    size_kb = out.stat().st_size / 1024
    print(f"\nWrote {len(samples)} samples → {output_path} ({size_kb:.1f} KB)")
    print(f"Avg sample size: {total_chars // max(1, len(samples))} chars")


def main():
    parser = argparse.ArgumentParser(description="Fetch Rust gists and build training corpus")
    parser.add_argument("--user",   default="RandyMcMillan",
                        help="GitHub username (default: RandyMcMillan)")
    parser.add_argument("--output", default="train_rust.txt",
                        help="Output training file (default: train_rust.txt)")
    parser.add_argument("--token",  default=None,
                        help="GitHub personal access token (optional, raises rate limit)")
    parser.add_argument("--min-size", type=int, default=MIN_SIZE,
                        help=f"Min file size in bytes (default: {MIN_SIZE})")
    parser.add_argument("--max-size", type=int, default=MAX_SIZE,
                        help=f"Max file size in bytes (default: {MAX_SIZE})")
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Delay between raw file fetches in seconds (default: 0.3)")
    args = parser.parse_args()

    min_size = args.min_size
    max_size = args.max_size

    print(f"Fetching gists for {args.user} ...")
    gists = fetch_all_gists(args.user, args.token)
    print(f"Total gists: {len(gists)}")

    rust_files = [
        f for f in extract_rust_files(gists)
        if min_size <= f["size"] <= max_size
    ]
    print(f"\nRust files in range [{MIN_SIZE}–{MAX_SIZE} bytes]: {len(rust_files)}")
    for f in rust_files[:5]:
        print(f"  {f['filename']} ({f['size']} bytes)")
    if len(rust_files) > 5:
        print(f"  ... and {len(rust_files) - 5} more")

    print(f"\nFetching raw content ...")
    samples = fetch_corpus(rust_files, args.token, args.delay)

    write_corpus(samples, args.output)

    # Summary stats
    langs = {}
    for g in gists:
        for fmeta in g["files"].values():
            lang = fmeta.get("language") or "unknown"
            langs[lang] = langs.get(lang, 0) + 1
    print("\nAll languages in gists:")
    for lang, count in sorted(langs.items(), key=lambda x: -x[1])[:10]:
        print(f"  {lang:20s} {count}")


if __name__ == "__main__":
    main()
