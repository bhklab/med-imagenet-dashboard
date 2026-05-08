#!/usr/bin/env python3
"""Compose modality + sources poster PNGs side by side (equal height, aligned top).

Reads default paths under assets/poster-panels/. Run after:

  pixi run python scripts/render_home_poster.py --panel modality --panel sources

(or ``--panel all``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def combine_modality_sources_row(
    modality_png: Path,
    sources_png: Path,
    out_png: Path,
    *,
    gap_px: int = 24,
    bg_rgb: tuple[int, int, int] = (255, 255, 255),
) -> Path:
    """Resize both panels to the same pixel height (preserving aspect), paste left/right."""
    left = Image.open(modality_png).convert("RGBA")
    right = Image.open(sources_png).convert("RGBA")

    target_h = max(left.height, right.height)
    if target_h <= 0:
        raise ValueError("empty image height")

    def _to_height(im: Image.Image, h: int) -> Image.Image:
        if im.height == h:
            return im
        new_w = max(1, round(im.width * h / im.height))
        return im.resize((new_w, h), Image.Resampling.LANCZOS)

    left_r = _to_height(left, target_h)
    right_r = _to_height(right, target_h)

    total_w = left_r.width + gap_px + right_r.width
    canvas = Image.new("RGBA", (total_w, target_h), (*bg_rgb, 255))
    canvas.paste(left_r, (0, 0), left_r)
    canvas.paste(right_r, (left_r.width + gap_px, 0), right_r)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    rgb = canvas.convert("RGB")
    rgb.save(out_png, format="PNG", optimize=True)
    return out_png


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    default_dir = root / "assets" / "poster-panels"

    p = argparse.ArgumentParser(description="Side-by-side modality + sources PNG (equal height).")
    p.add_argument("--modality", type=Path, default=default_dir / "03-modality.png")
    p.add_argument("--sources", type=Path, default=default_dir / "04-sources.png")
    p.add_argument(
        "--output",
        type=Path,
        default=default_dir / "03-04-modality-sources.png",
        help="Output PNG path.",
    )
    p.add_argument("--gap", type=int, default=24, help="Horizontal gap in pixels between panels.")
    args = p.parse_args()

    if not args.modality.is_file():
        raise SystemExit(f"Missing modality PNG: {args.modality}")
    if not args.sources.is_file():
        raise SystemExit(f"Missing sources PNG: {args.sources}")

    out = combine_modality_sources_row(args.modality, args.sources, args.output, gap_px=args.gap)
    print(f"Wrote {out.resolve()} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
