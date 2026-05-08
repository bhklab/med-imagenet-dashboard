#!/usr/bin/env python3
"""Write a QR code PNG (black on white), print-ready.

Default encodes the public Med-ImageNet repo URL. Override with --url.

Examples:
  pixi run python scripts/render_qr_png.py
  pixi run python scripts/render_qr_png.py --url https://example.org --out assets/my-qr.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import qrcode


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a QR code PNG.")
    p.add_argument(
        "--url",
        default="https://github.com/bhklab/med-imagenet",
        help="Payload (URL or text) to encode",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("assets/poster-panels/med-imagenet-qr.png"),
        help="Output PNG path",
    )
    p.add_argument("--box-size", type=int, default=14, help="Pixels per module")
    p.add_argument("--border", type=int, default=3, help="Quiet-zone modules (border)")
    args = p.parse_args()

    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=args.box_size,
        border=args.border,
    )
    qr.add_data(args.url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#111827", back_color="#FFFFFF")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out)
    print(f"Wrote {args.out.resolve()} ({args.url!r})")


if __name__ == "__main__":
    main()
