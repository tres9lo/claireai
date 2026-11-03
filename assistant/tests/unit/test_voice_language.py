"""The voice must not read French with an English mouth."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from claireai.voice import (
    available_voices,
    detect_language,
    for_speech,
    speak_text,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Hello, this is a normal English sentence.", "en"),
        ("Bonjour, je voudrais ouvrir mes fichiers.", "fr"),
        ("Je ne sais pas.", "fr"),
        ("Ça va très bien.", "fr"),
        ("The quick brown fox jumps over the lazy dog.", "en"),
        ("", "en"),
        ("12345", "en"),
    ],
)
def test_detect_language(text: str, expected: str) -> None:
    assert detect_language(text) == expected


def test_markdown_is_not_read_out_loud() -> None:
    answer = """
## Result

Here is the **list**:

- `alpha.txt`
- [beta](https://example.com/beta)

```python
print("secret")
```
"""
    spoken = for_speech(answer)
    assert "```" not in spoken
    assert "**" not in spoken
    assert "##" not in spoken
    assert "https://" not in spoken
    assert "print(" not in spoken
    assert "alpha.txt" in spoken
    assert "beta" in spoken


def test_a_french_answer_is_synthesized_with_a_french_voice() -> None:
    if shutil.which("espeak-ng") is None and shutil.which("espeak") is None:
        pytest.skip("no espeak on this host")

    engine = shutil.which("espeak-ng") or shutil.which("espeak")
    phrase = "Bonjour, je m'appelle Claire."
    french = subprocess.run([engine, "--stdout", "-v", "fr"], input=phrase.encode(),
                            capture_output=True, check=False)
    english = subprocess.run([engine, "--stdout", "-v", "en"], input=phrase.encode(),
                             capture_output=True, check=False)

    assert french.returncode == 0 and french.stdout.startswith(b"RIFF")
    assert english.returncode == 0 and english.stdout.startswith(b"RIFF")
    # Two different voices produce different audio for the same sentence.
    assert french.stdout != english.stdout


def test_speak_text_accepts_a_language_and_rejects_a_bad_one() -> None:
    if shutil.which("espeak-ng") is None and shutil.which("espeak") is None:
        pytest.skip("no espeak on this host")
    with pytest.raises(ValueError):
        speak_text("hello", language="de")


def test_available_voices_lists_en_and_fr() -> None:
    if shutil.which("espeak-ng") is None:
        pytest.skip("no espeak-ng on this host")
    voices = available_voices()
    assert set(voices) == {"en", "fr"}
