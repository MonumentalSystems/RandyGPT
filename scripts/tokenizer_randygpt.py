"""
tokenizer_randygpt.py — Python port of the randyGPT BPE tokenizer.

Matches the Rust BpeTokenizer encode/decode exactly:
  - Encode: apply merges in priority order (lowest merged_id first)
  - Decode: concatenate vocab[id] for each token, skipping bos/eos
  - Splits on newlines before encoding (same as Rust rayon parallel path)

Usage:
    from tokenizer_randygpt import RandyGPTTokenizer

    tok = RandyGPTTokenizer.from_file("vocab.json")
    ids = tok.encode("Once upon a time")
    txt = tok.decode(ids)
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple


class RandyGPTTokenizer:
    def __init__(
        self,
        vocab:     List[str],
        merges:    List[Tuple[str, str]],
    ):
        self.vocab      = vocab
        self.merges     = merges
        self.token_to_id: Dict[str, int] = {s: i for i, s in enumerate(vocab)}
        self.bos_id     = self.token_to_id.get("<|bos|>", 0)
        self.eos_id     = self.token_to_id.get("<|eos|>", 1)
        self.vocab_size = len(vocab)

        # (left_id, right_id) → merged_id — first insertion wins (highest priority)
        self.merge_map: Dict[Tuple[int, int], int] = {}
        for left, right in merges:
            l = self.token_to_id.get(left)
            r = self.token_to_id.get(right)
            merged = self.token_to_id.get(left + right)
            if l is not None and r is not None and merged is not None:
                self.merge_map.setdefault((l, r), merged)

    @classmethod
    def from_file(cls, path: str) -> "RandyGPTTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        vocab  = data["vocab"]
        merges = [tuple(m) for m in data["merges"]]
        return cls(vocab, merges)

    def _encode_chunk(self, text: str) -> List[int]:
        """Encode a single line (no newlines). Mirrors Rust encode_chunk."""
        tokens = [self.token_to_id[c] for c in text if c in self.token_to_id]
        if len(tokens) < 2:
            return tokens

        # Repeatedly apply the highest-priority merge (lowest merged_id)
        while True:
            best_merged_id = None
            for i in range(len(tokens) - 1):
                m = self.merge_map.get((tokens[i], tokens[i + 1]))
                if m is not None:
                    if best_merged_id is None or m < best_merged_id:
                        best_merged_id = m

            if best_merged_id is None:
                break

            out = []
            i = 0
            while i < len(tokens):
                if (i + 1 < len(tokens) and
                        self.merge_map.get((tokens[i], tokens[i + 1])) == best_merged_id):
                    out.append(best_merged_id)
                    i += 2
                else:
                    out.append(tokens[i])
                    i += 1
            tokens = out

        return tokens

    def encode(self, text: str) -> List[int]:
        """Encode text. Splits on newlines matching Rust parallel encode path."""
        nl_id = self.token_to_id.get("\n")
        lines = text.split("\n")
        result = []
        for i, line in enumerate(lines):
            result.extend(self._encode_chunk(line))
            if i < len(lines) - 1 and nl_id is not None:
                result.append(nl_id)
        return result

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Decode token ids back to text."""
        parts = []
        for i in ids:
            if skip_special and i in (self.bos_id, self.eos_id):
                continue
            if 0 <= i < len(self.vocab):
                parts.append(self.vocab[i])
        return "".join(parts)

    def __len__(self) -> int:
        return self.vocab_size
