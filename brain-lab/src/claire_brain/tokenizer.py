"""Byte-pair tokenizer trained from the supplied corpus.

Both halves of this module do the same thing over and over: find every
occurrence of one pair and merge it. The naive way to do that rescans the whole
text once per merge, which makes the cost `merges x corpus` and meant a 5 MB
corpus spent longer tokenising than training. Here a sequence is a linked list
with an index of adjacent pairs, so a merge costs only the number of places that
pair actually appears.

The merge order is unchanged: ranks are applied in ascending order, and
occurrences within a rank are merged left to right. That is exactly what the
previous implementation did, so a checkpoint trained with the old tokenizer still
encodes to the same tokens with this one.
"""

from __future__ import annotations

from array import array
from collections import Counter

EOS = 256
PAD = 257
BASE_VOCAB_SIZE = 258

# Positions and token ids both stay far below the int32 limit; using arrays keeps
# a multi-megabyte corpus to tens of megabytes instead of hundreds.
_TYPE = "i"


class _Sequence:
    """One token sequence as a linked list, plus an index of adjacent pairs."""

    __slots__ = ("index", "next", "previous", "alive", "pairs")

    def __init__(self, data: list[int], counts: dict[tuple[int, int], int] | None = None) -> None:
        size = len(data)
        self.index = array(_TYPE, data)
        self.next = array(_TYPE, range(1, size + 1))
        if size:
            self.next[-1] = -1
        self.previous = array(_TYPE, range(-1, size - 1))
        self.alive = bytearray(b"\x01") * size
        self.pairs: dict[tuple[int, int], array] = {}
        for position in range(size - 1):
            pair = (data[position], data[position + 1])
            slot = self.pairs.get(pair)
            if slot is None:
                slot = self.pairs[pair] = array(_TYPE)
            slot.append(position)
            if counts is not None:
                counts[pair] = counts.get(pair, 0) + 1

    def merge(self, pair: tuple[int, int], token: int, counts: dict[tuple[int, int], int] | None = None) -> int:
        """Merge every occurrence of ``pair`` into ``token``; return how many.

        Positions are visited in ascending order so overlapping runs resolve the
        same way they did before: for ``aaa`` exactly one merge happens, at the
        left, leaving ``Xa``. Entries that merging has invalidated are skipped.
        """
        positions = self.pairs.pop(pair, None)
        if not positions:
            return 0
        index, nxt, previous, alive = self.index, self.next, self.previous, self.alive
        left_id, right_id = pair
        merged = 0
        for i in sorted(positions):
            if not alive[i]:
                continue
            j = nxt[i]
            if j == -1 or not alive[j]:
                continue
            if index[i] != left_id or index[j] != right_id:
                continue
            left, right = previous[i], nxt[j]
            if counts is not None:
                if left != -1:
                    _bump(counts, (index[left], index[i]), -1)
                _bump(counts, pair, -1)
                if right != -1:
                    _bump(counts, (index[j], index[right]), -1)
            # Fold j into i and unlink j.
            index[i] = token
            alive[j] = 0
            nxt[i] = right
            if right != -1:
                previous[right] = i
            if counts is not None:
                if left != -1:
                    _bump(counts, (index[left], token), 1)
                if right != -1:
                    _bump(counts, (token, index[right]), 1)
            if left != -1:
                self._remember((index[left], token), left)
            if right != -1:
                self._remember((token, index[right]), i)
            merged += 1
        return merged

    def _remember(self, pair: tuple[int, int], position: int) -> None:
        slot = self.pairs.get(pair)
        if slot is None:
            slot = self.pairs[pair] = array(_TYPE)
        slot.append(position)

    def first_occurrence(self, pair: tuple[int, int]) -> int | None:
        """Leftmost live occurrence of ``pair``, for deterministic tie-breaking."""
        positions = self.pairs.get(pair)
        if not positions:
            return None
        index, nxt, alive = self.index, self.next, self.alive
        for position in sorted(positions):
            if not alive[position]:
                continue
            following = nxt[position]
            if following != -1 and alive[following] and (index[position], index[following]) == pair:
                return position
        return None

    def flatten(self) -> list[int]:
        out: list[int] = []
        index, nxt = self.index, self.next
        position = 0
        while position != -1:
            out.append(index[position])
            position = nxt[position]
        return out


def _bump(counts: dict[tuple[int, int], int], pair: tuple[int, int], delta: int) -> None:
    updated = counts.get(pair, 0) + delta
    if updated > 0:
        counts[pair] = updated
    else:
        counts.pop(pair, None)


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
        counts: dict[tuple[int, int], int] = {}
        sequences = [_Sequence(list(text.encode("utf-8")), counts) for text in texts]
        merges: list[tuple[int, int]] = []
        while BASE_VOCAB_SIZE + len(merges) < vocab_size:
            if not counts:
                break
            best = max(counts.values())
            if best < 2:
                break
            candidates = [pair for pair, count in counts.items() if count == best]
            pair = candidates[0] if len(candidates) == 1 else _earliest(sequences, candidates)
            if pair is None:
                break
            token = BASE_VOCAB_SIZE + len(merges)
            merges.append(pair)
            for sequence in sequences:
                sequence.merge(pair, token, counts)
        return cls(merges)

    def encode(self, text: str, *, append_eos: bool = True) -> list[int]:
        data = list(text.encode("utf-8"))
        if len(data) < 2 or not self.merges:
            return data + ([EOS] if append_eos else [])
        sequence = _Sequence(data)
        for rank, pair in enumerate(self.merges):
            sequence.merge(pair, BASE_VOCAB_SIZE + rank)
        return sequence.flatten() + ([EOS] if append_eos else [])

    def decode(self, tokens: list[int]) -> str:
        raw = b"".join(
            self.token_bytes[token]
            for token in tokens
            if 0 <= token < len(self.token_bytes) and token not in {EOS, PAD}
        )
        return raw.decode("utf-8", errors="replace")


def _earliest(sequences: list[_Sequence], candidates: list[tuple[int, int]]) -> tuple[int, int] | None:
    """Which tied pair comes first in the corpus.

    ``Counter.most_common`` breaks ties by insertion order, which for a freshly
    rebuilt counter means "first seen in the corpus". Reproducing that keeps the
    learned merges identical to the previous implementation.
    """
    best_pair: tuple[int, int] | None = None
    best_key: tuple[int, int] | None = None
    for pair in candidates:
        for order, sequence in enumerate(sequences):
            position = sequence.first_occurrence(pair)
            if position is None:
                continue
            key = (order, position)
            if best_key is None or key < best_key:
                best_key, best_pair = key, pair
            break
    return best_pair
