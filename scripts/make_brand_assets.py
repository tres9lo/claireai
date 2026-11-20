"""Render the ClaireAI mark and its PNG sizes from one geometry.

The browser gets the SVG in ``desktop/presence/logo.svg``; Tk cannot decode SVG,
so the same shapes are drawn here with Pillow and committed as PNGs. Run this
again after changing the mark:

    assistant/.venv/bin/python scripts/make_brand_assets.py

Pillow is a development dependency; nothing at runtime needs it.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assistant" / "src" / "claireai" / "desktop" / "assets"
SIZES = (32, 48, 64, 128, 256, 512)

TOP = (155, 140, 255)
BOTTOM = (75, 60, 240)
SUPERSAMPLE = 4


def rounded_mask(size: int, radius: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size - 1, size - 1), radius=radius, fill=255)
    return mask


def gradient(size: int) -> Image.Image:
    image = Image.new("RGB", (size, size))
    pixels = image.load()
    for y in range(size):
        blend = y / max(1, size - 1)
        colour = tuple(round(TOP[i] + (BOTTOM[i] - TOP[i]) * blend) for i in range(3))
        for x in range(size):
            pixels[x, y] = colour
    return image


def mark(size: int) -> Image.Image:
    """The app icon: a rounded violet square with an open ring and a spark."""
    canvas = size * SUPERSAMPLE
    plate = gradient(canvas)
    plate.putalpha(rounded_mask(canvas, round(canvas * 0.22)))

    # A soft highlight across the top third keeps the square from looking flat.
    highlight = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    ImageDraw.Draw(highlight).ellipse(
        (-canvas * 0.25, -canvas * 0.55, canvas * 1.25, canvas * 0.55),
        fill=(255, 255, 255, 26),
    )
    plate = Image.alpha_composite(plate, highlight)

    art = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    draw = ImageDraw.Draw(art)
    ring = canvas * 0.242
    width = max(2, round(canvas * 0.086))
    draw.arc(
        (canvas / 2 - ring, canvas / 2 - ring, canvas / 2 + ring, canvas / 2 + ring),
        # The long way round: from lower-right, through the left, to
        # upper-right. The 90-degree gap on the right is the opening,
        # which is what makes the ring read as a C rather than a bracket.
        start=45,
        end=315,
        fill=(255, 255, 255, 255),
        width=width,
    )
    dot = canvas * 0.041
    centre = canvas / 2 + ring * 0.92
    draw.ellipse(
        (centre - dot, canvas / 2 - dot, centre + dot, canvas / 2 + dot),
        fill=(255, 255, 255, 255),
    )
    plate = Image.alpha_composite(plate, art)
    return plate.resize((size, size), Image.LANCZOS)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for size in SIZES:
        mark(size).save(OUTPUT / f"logo-{size}.png")
        print(f"wrote {OUTPUT / f'logo-{size}.png'}")


if __name__ == "__main__":
    main()
