"""Local image understanding, and screen capture.

A CLIP model compares an image against a list of phrases and returns which phrase
it is closest to. That is the honest description of what this does: it ranks
candidate descriptions, it does not invent captions. With the default label set
that answers "what is in this picture / on this screen"; with caller-supplied
labels it becomes a zero-shot classifier.

Nothing leaves the machine: the weights are local, inference is CPU-only, and
capture reads the local framebuffer. Two consequences worth stating plainly:

* The model must be installed (``open-clip-torch``) and its weights downloaded
  once. Without them this module raises :class:`VisionUnavailable` rather than
  pretending to see.
* Screen capture only works where the compositor allows it. On a session that
  blocks capture the tools return black frames; those are detected and refused
  as a failure, because silently describing a black rectangle is worse than
  saying no.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

MAX_IMAGE_BYTES = 12 * 1024 * 1024
CAPTURE_TIMEOUT_SECONDS = 20

# The smallest standard CLIP: ~600 MB of weights, and fast enough on four CPU
# threads. Hugging Face's transformers is used rather than open_clip because
# open_clip imports torchvision unconditionally, and the torchvision wheel that
# installs here does not register its operators against torch 2.15.
MODEL_NAME = "ViT-B-32"
PRETRAINED = "clip-vit-base-patch32"
MODEL_ID = "openai/clip-vit-base-patch32"

# Broad enough to be useful on photos, screenshots and documents without being
# so long that every call pays for it. Callers can pass their own labels.
DEFAULT_LABELS: tuple[str, ...] = (
    "a screenshot of a computer desktop",
    "a screenshot of a web page",
    "a screenshot of a code editor or terminal",
    "a screenshot of a chat conversation",
    "a photo of a person",
    "a photo of a group of people",
    "a photo of a cat",
    "a photo of a dog",
    "a photo of another animal",
    "a photo of a landscape",
    "a photo of a building",
    "a photo of food",
    "a photo of a car or other vehicle",
    "a photo of an indoor room",
    "a photo of an object on a plain background",
    "a photograph of an animal in a zoo",
    "a scanned document with text",
    "a chart, graph or diagram",
    "a table of numbers",
    "a map",
    "a logo or icon",
    "abstract art or a pattern",
    "an error message or dialog box",
    "a photograph that is blurry or hard to read",
    "a photo of a bird",
    "a photo of an insect",
    "a photo of a flower or plant",
    "a photo of a toy",
    "a photo of an airplane",
    "a photo of a boat or ship",
    "a photo of a bicycle or motorcycle",
    "a photo of furniture",
    "a photo of a computer or phone",
    "a photo of a classroom or office",
    "a photo of a street scene",
    "a photo of water, sea or a beach",
    "a photo of a mountain or sky",
    "a hand-written note",
    "a photograph of a textile or clothing",
    "a photograph that is mostly one solid colour",
)


class VisionUnavailable(RuntimeError):
    """The vision model or the screen could not be used, and we say so."""


_model: tuple | None = None


def model_status() -> dict[str, object]:
    """What is installed, without loading any weights."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return {"installed": False, "loaded": False, "detail": "transformers is not installed"}
    return {"installed": True, "loaded": _model is not None, "detail": f"{MODEL_NAME} / {PRETRAINED}"}


def _load():
    """Load the model once and keep it; CPU inference is the only path here."""
    global _model
    if _model is not None:
        return _model
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as error:
        raise VisionUnavailable(
            "local vision model is not installed; install it with "
            "`uv pip install transformers`"
        ) from error
    try:
        model = CLIPModel.from_pretrained(MODEL_ID)
        processor = CLIPProcessor.from_pretrained(MODEL_ID)
    except Exception as error:  # noqa: BLE001 - download/format failures are all "unavailable"
        raise VisionUnavailable(f"could not load the local vision model: {type(error).__name__}") from error
    model.eval()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    _model = (model, processor, torch)
    return _model


def open_image(data: bytes):
    """Decode image bytes, with a size limit, into a PIL image."""
    if not data:
        raise VisionUnavailable("the image is empty")
    if len(data) > MAX_IMAGE_BYTES:
        raise VisionUnavailable(f"the image is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB")
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - Pillow is a hard dependency here
        raise VisionUnavailable("Pillow is not installed") from error
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as error:  # noqa: BLE001 - any decoder failure means "not an image"
        raise VisionUnavailable(f"that file is not a readable image: {type(error).__name__}") from error
    return image.convert("RGB")


def looks_blank(image) -> bool:
    """True for a uniform image, which is what a blocked screen capture returns."""
    red, green, blue = image.convert("RGB").getextrema()
    return all(low == high for low, high in (red, green, blue))


def _normalised(output) -> "object":
    """Pull the embedding out of whatever the model returned, then normalise it.

    Depending on the transformers version, ``get_image_features`` hands back a
    tensor or a model-output object whose ``pooler_output`` is the projected CLIP
    space. Both are accepted, and an unrecognised shape is refused rather than
    silently compared against the wrong thing.
    """
    tensor = getattr(output, "pooler_output", None)
    if tensor is None:
        tensor = getattr(output, "image_embeds", None)
    if tensor is None:
        tensor = getattr(output, "text_embeds", None)
    if tensor is None:
        tensor = output
    if not hasattr(tensor, "norm"):
        raise VisionUnavailable("the vision model returned an unrecognised output")
    return tensor / tensor.norm(dim=-1, keepdim=True)


def image_features(images: Sequence) -> "object":
    """Normalised CLIP embeddings for a batch of PIL images."""
    model, processor, torch = _load()
    inputs = processor(images=list(images), return_tensors="pt")
    with torch.inference_mode():
        return _normalised(model.get_image_features(**inputs))


def text_features(texts: Sequence[str]) -> "object":
    """Normalised CLIP embeddings for a batch of phrases."""
    model, processor, torch = _load()
    inputs = processor(text=list(texts), return_tensors="pt", padding=True, truncation=True)
    with torch.inference_mode():
        return _normalised(model.get_text_features(**inputs))


def rank(image, labels: Sequence[str], *, limit: int = 5) -> list[tuple[str, float]]:
    """Best-matching labels for one image, highest probability first."""
    if not labels:
        raise VisionUnavailable("at least one label is required")
    import torch as torch_module

    with torch_module.inference_mode():
        scores = (100.0 * image_features([image]) @ text_features(labels).T).softmax(dim=-1)[0]
    pairs = sorted(zip(labels, scores.tolist()), key=lambda item: item[1], reverse=True)
    return pairs[: max(1, limit)]


def describe(data: bytes, labels: Sequence[str] | None = None, *, limit: int = 3) -> list[tuple[str, float]]:
    """Rank what is in these image bytes against labels (or the default set)."""
    image = open_image(data)
    if looks_blank(image):
        raise VisionUnavailable("the image is a single solid colour, so there is nothing to identify")
    return rank(image, labels or DEFAULT_LABELS, limit=limit)


# -- screen capture --------------------------------------------------------


def capture_commands() -> list[tuple[str, list[str]]]:
    """Capture programs to try, in order, as (name, argv). The path is appended."""
    return [
        ("grim", ["grim", "-t", "png"]),
        ("gnome-screenshot", ["gnome-screenshot", "-f"]),
        ("spectacle", ["spectacle", "-b", "-n", "-o"]),
        ("scrot", ["scrot", "-o"]),
        ("import", ["import", "-window", "root"]),
    ]


def available_capture_tools() -> list[str]:
    return [name for name, _argv in capture_commands() if shutil.which(name)]


def capture_screen() -> bytes:
    """PNG bytes of the whole screen, from the first capture tool that works.

    Raises :class:`VisionUnavailable` with every backend's reason when none does,
    and treats a uniform result as a refusal rather than a valid screenshot.
    """
    tools = available_capture_tools()
    if not tools:
        raise VisionUnavailable(
            "no screen capture tool is installed (looked for grim, gnome-screenshot, "
            "spectacle, scrot and ImageMagick's import)"
        )

    problems: list[str] = []
    for name, command in capture_commands():
        if not shutil.which(name):
            continue
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            target = Path(handle.name)
        try:
            try:
                subprocess.run(
                    [*command, str(target)],
                    capture_output=True,
                    timeout=CAPTURE_TIMEOUT_SECONDS,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                problems.append(f"{name}: {type(error).__name__}")
                continue
            data = target.read_bytes()
            if not data:
                problems.append(f"{name}: produced no file (the display may be blocked)")
                continue
            try:
                image = open_image(data)
            except VisionUnavailable as error:
                problems.append(f"{name}: {error}")
                continue
            if looks_blank(image):
                problems.append(f"{name}: returned a blank frame (the compositor blocks capture)")
                continue
            return data
        finally:
            target.unlink(missing_ok=True)

    raise VisionUnavailable("no capture tool produced a usable screenshot — " + "; ".join(problems))
