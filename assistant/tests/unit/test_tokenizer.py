"""The tokenizer must stay compatible: same merges, same tokens, only faster.

The reference implementations at the top of this file are the previous, naive
algorithms, kept verbatim. They are slow but obviously correct, so any difference
in the new code shows up as a failing comparison rather than as a silently
different tokenization that would quietly invalidate a trained checkpoint.
"""

from __future__ import annotations

import pytest

from claire_brain.tokenizer import BASE_VOCAB_SIZE, EOS, PAD, BytePairTokenizer

TRICKY = [
    "",
    "a",
    "aa",
    "aaa",
    "aaaa",
    "aaaaa",
    "abab",
    "abababab",
    "abracadabra abracadabra",
    "the the the theory of the theatre",
    "banana banana bandana",
    "héllo wörld — accents and dashes",
    "日本語のテキスト",
    "x" * 500,
    " ".join(["the quick brown fox jumps over the lazy dog"] * 12),
]


# -- reference implementations (the previous algorithm, unchanged) ----------


def reference_train(texts: list[str], vocab_size: int) -> list[tuple[int, int]]:
    sequences = [list(text.encode("utf-8")) for text in texts]
    merges: list[tuple[int, int]] = []
    while BASE_VOCAB_SIZE + len(merges) < vocab_size:
        counts: Counter = Counter()
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
    return merges


def reference_encode(merges: list[tuple[int, int]], text: str, *, append_eos: bool = True) -> list[int]:
    ranks = {pair: rank for rank, pair in enumerate(merges)}
    sequence = list(text.encode("utf-8"))
    while len(sequence) > 1:
        candidates = [
            (ranks[(sequence[index], sequence[index + 1])], index)
            for index in range(len(sequence) - 1)
            if (sequence[index], sequence[index + 1]) in ranks
        ]
        if not candidates:
            break
        rank, _ = min(candidates)
        pair = merges[rank]
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


from collections import Counter  # noqa: E402  (used by the reference above)


# -- equivalence -----------------------------------------------------------


@pytest.mark.parametrize("vocab_size", [BASE_VOCAB_SIZE, 260, 280, 320])
def test_training_learns_exactly_the_same_merges_as_before(vocab_size: int) -> None:
    texts = [" ".join(TRICKY), TRICKY[14], TRICKY[9]]

    learned = BytePairTokenizer.train(texts, vocab_size=vocab_size)

    assert learned.merges == reference_train(texts, vocab_size)
    assert learned.vocab_size == vocab_size


def test_training_stops_when_no_pair_repeats() -> None:
    # Each pair occurs once, so there is nothing worth merging.
    learned = BytePairTokenizer.train(["abc"], vocab_size=512)
    assert learned.merges == reference_train(["abc"], 512)


@pytest.mark.parametrize("text", TRICKY)
def test_encoding_produces_exactly_the_same_tokens_as_before(text: str) -> None:
    tokenizer = BytePairTokenizer.train([" ".join(TRICKY)], vocab_size=320)

    assert tokenizer.encode(text) == reference_encode(tokenizer.merges, text)


def test_overlapping_runs_resolve_left_to_right() -> None:
    """``aaa`` merges once, at the left. This is where a naive rewrite drifts."""
    tokenizer = BytePairTokenizer.train(["a" * 40], vocab_size=259)
    assert tokenizer.encode("aaa", append_eos=False) == [BASE_VOCAB_SIZE, ord("a")]


def test_a_trained_tokenizer_round_trips() -> None:
    tokenizer = BytePairTokenizer.train([" ".join(TRICKY)], vocab_size=300)
    for text in TRICKY:
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_eos_is_appended_and_never_decoded() -> None:
    tokenizer = BytePairTokenizer.train([" ".join(TRICKY)], vocab_size=280)
    encoded = tokenizer.encode("banana")
    assert encoded[-1] == EOS
    assert PAD not in encoded
    assert tokenizer.decode(encoded) == "banana"


def test_an_empty_tokenizer_still_encodes() -> None:
    tokenizer = BytePairTokenizer()
    assert tokenizer.vocab_size == BASE_VOCAB_SIZE
    assert tokenizer.encode("hi", append_eos=False) == list(b"hi")
    assert tokenizer.encode("") == [EOS]


def test_a_bad_vocabulary_is_refused() -> None:
    with pytest.raises(ValueError):
        BytePairTokenizer.train(["text"], vocab_size=BASE_VOCAB_SIZE - 1)
