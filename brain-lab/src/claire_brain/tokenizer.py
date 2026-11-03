"""Byte-pair tokenizer trained from the supplied corpus."""

from __future__ import annotations

from collections import Counter

EOS = 256
PAD = 257
BASE_VOCAB_SIZE = 258


class BytePairTokenizer:
    """Simple BPE implementation; its merges are learned only from local data."""

    def __init__(self, merges: list[tuple[int, int]] | None = None) -> None:
        self.merges = list(merges or [])
        self.ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.token_bytes = [bytes([value]) for value in range(256)] + [b"", b""]
        for left, right in self.merges:
            self.token_bytes.append(self.token_bytes[left] + self.token_bytes[right])

    @property
    def vocab_size(self) -> int:
        return BASE_VOCAB_SIZE + len(self.merges)

    @classmethod
    def train(cls, texts: list[str], *, vocab_size: int = 512) -> "BytePairTokenizer":
        if vocab_size < BASE_VOCAB_SIZE:
            raise ValueError(f"vocab_size must be at least {BASE_VOCAB_SIZE}")
        sequences = [list(text.encode("utf-8")) for text in texts]
        merges: list[tuple[int, int]] = []
        while BASE_VOCAB_SIZE + len(merges) < vocab_size:
            counts: Counter[tuple[int, int]] = Counter()
            for sequence in sequences:
                counts.update(zip(sequence, sequence[1:]))
            if not counts:
                break
            pair, frequency = counts.most_common(1)[0]
            if frequency < 2:
                break
            token_id = BASE_VOCAB_SIZE + len(merges)
            merges.append(pair)
            for index, sequence in enumerate(sequences):
                merged: list[int] = []
                cursor = 0
                while cursor < len(sequence):
                    if cursor + 1 < len(sequence) and (sequence[cursor], sequence[cursor + 1]) == pair:
                        merged.append(token_id)
                        cursor += 2
                    else:
                        merged.append(sequence[cursor])
                        cursor += 1
                sequences[index] = merged
        return cls(merges)

    def encode(self, text: str, *, append_eos: bool = True) -> list[int]:
        sequence = list(text.encode("utf-8"))
        while len(sequence) > 1:
            candidates = [
                (self.ranks[(sequence[index], sequence[index + 1])], index)
                for index in range(len(sequence) - 1)
                if (sequence[index], sequence[index + 1]) in self.ranks
            ]
            if not candidates:
                break
            rank, _ = min(candidates)
            pair = self.merges[rank]
            token_id = BASE_VOCAB_SIZE + rank
            merged: list[int] = []
            cursor = 0
            while cursor < len(sequence):
                if cursor + 1 < len(sequence) and (sequence[cursor], sequence[cursor + 1]) == pair:
                    merged.append(token_id)
                    cursor += 2
                else:
                    merged.append(sequence[cursor])
                    cursor += 1
            sequence = merged
        return sequence + ([EOS] if append_eos else [])

    def decode(self, tokens: list[int]) -> str:
        raw = b"".join(
            self.token_bytes[token]
            for token in tokens
            if 0 <= token < len(self.token_bytes) and token not in {EOS, PAD}
        )
        return raw.decode("utf-8", errors="replace")
