#!/usr/bin/env python3
"""
build_bpe_vocab.py — Build a BPE vocab from a training text file.

Produces vocab.json compatible with RandyGPTTokenizer and the Rust BPE tokenizer.

Usage:
    python scripts/build_bpe_vocab.py --input train_rust.txt --output vocab_rust.json --size 800
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path


def build_vocab(text: str, target_size: int, verbose: bool = True) -> tuple:
    """
    Train BPE from scratch. Returns (vocab, merges).
    vocab:  list of token strings (index = token id)
    merges: list of [left, right] pairs in merge order
    """
    # ── Seed vocab: all unique characters + special tokens ────────────────────
    chars = sorted(set(text))
    vocab = ["<|bos|>", "<|eos|>"] + chars
    token_to_id = {t: i for i, t in enumerate(vocab)}

    if verbose:
        print(f"Seed vocab: {len(vocab)} tokens ({len(chars)} unique chars)")

    # ── Tokenize corpus into char ids ─────────────────────────────────────────
    # Split on newlines to mirror Rust encode behavior
    lines = text.split("\n")
    nl_id = token_to_id.get("\n")

    corpus = []  # list of token-id lists, one per "word" (line chunk)
    for i, line in enumerate(lines):
        if line:
            corpus.append([token_to_id[c] for c in line if c in token_to_id])
        if i < len(lines) - 1 and nl_id is not None:
            corpus.append([nl_id])

    # Remove empty
    corpus = [seq for seq in corpus if seq]

    merges = []
    t0 = time.time()

    # ── BPE merge loop ────────────────────────────────────────────────────────
    while len(vocab) < target_size:
        # Count all adjacent pairs
        pair_counts = defaultdict(int)
        for seq in corpus:
            for i in range(len(seq) - 1):
                pair_counts[(seq[i], seq[i + 1])] += 1

        if not pair_counts:
            break

        # Pick most frequent pair
        best_pair = max(pair_counts, key=pair_counts.__getitem__)
        best_count = pair_counts[best_pair]

        if best_count < 2:
            break

        # Create new token
        left_str  = vocab[best_pair[0]]
        right_str = vocab[best_pair[1]]
        new_str   = left_str + right_str
        new_id    = len(vocab)
        vocab.append(new_str)
        token_to_id[new_str] = new_id
        merges.append([left_str, right_str])

        # Apply merge across corpus
        new_corpus = []
        for seq in corpus:
            new_seq = []
            i = 0
            while i < len(seq):
                if i + 1 < len(seq) and seq[i] == best_pair[0] and seq[i+1] == best_pair[1]:
                    new_seq.append(new_id)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            new_corpus.append(new_seq)
        corpus = new_corpus

        if verbose and len(vocab) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {len(vocab):4d} tokens | last merge: {repr(new_str):20s} "
                  f"(count={best_count:6d}) | {elapsed:.1f}s")

    if verbose:
        print(f"Final vocab: {len(vocab)} tokens in {time.time()-t0:.1f}s")

    return vocab, merges


def main():
    parser = argparse.ArgumentParser(description="Build BPE vocab from training text")
    parser.add_argument("--input",  required=True, help="Input training text file")
    parser.add_argument("--output", required=True, help="Output vocab.json path")
    parser.add_argument("--size",   type=int, default=800,
                        help="Target vocab size (default: 800)")
    parser.add_argument("--quiet",  action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    print(f"Reading {args.input} ...")
    text = Path(args.input).read_text(encoding="utf-8")
    print(f"  {len(text):,} chars")

    vocab, merges = build_vocab(text, args.size, verbose=not args.quiet)

    out = {"vocab": vocab, "merges": merges}
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"Wrote {args.output} ({len(vocab)} tokens, {len(merges)} merges)")

    # Quick coverage check
    chars_in_text = set(text)
    vocab_chars   = set(t for t in vocab if len(t) == 1)
    missing = chars_in_text - vocab_chars
    if missing:
        print(f"WARNING: {len(missing)} chars in text not in vocab: {sorted(missing)[:10]}")
    else:
        print("Coverage: all chars in text covered by vocab")


if __name__ == "__main__":
    main()
