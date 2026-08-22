#!/usr/bin/env python3
"""Layered cyberpunk photo edit — preserves original pixels, adds local effects only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def screen_blend(base: np.ndarray, overlay: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Screen blend with alpha mask (0..1)."""
    a = alpha[..., None]
    out = 1.0 - (1.0 - base) * (1.0 - overlay)
    return np.clip(base * (1.0 - a) + out * a, 0.0, 1.0)


def add_neon_trim(img: np.ndarray, region_mask: np.ndarray) -> np.ndarray:
    """Neon lines along building edges in masked region."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    edges = edges.astype(np.float32) / 255.0
    edges *= region_mask

    overlay = np.zeros_like(img)
    # Cyan + magenta alternating bands by vertical position
    y_idx = np.linspace(0, 1, h)[:, None]
    cyan_w = np.clip(1.0 - np.abs(y_idx - 0.25) * 4, 0, 1)
    magenta_w = np.clip(1.0 - np.abs(y_idx - 0.55) * 4, 0, 1)
    overlay[..., 0] = edges * (0.15 + 0.85 * magenta_w)  # R
    overlay[..., 1] = edges * (0.10 + 0.70 * cyan_w)     # G
    overlay[..., 2] = edges * (0.55 + 0.45 * cyan_w)     # B

    glow = cv2.GaussianBlur((edges * 255).astype(np.uint8), (0, 0), 3).astype(np.float32) / 255.0
    glow = glow[..., None] * np.array([0.9, 0.2, 1.0])
    overlay = np.clip(overlay + glow * 0.35, 0, 1)

    alpha = np.clip(edges * 0.85 + glow[..., 0] * 0.25, 0, 1) * region_mask
    return screen_blend(img, overlay, alpha)


def boost_sign_neon(img: np.ndarray, sign_mask: np.ndarray) -> np.ndarray:
    """Enhance existing green sign glow."""
    overlay = np.zeros_like(img)
    overlay[..., 1] = 0.95  # green neon
    overlay[..., 2] = 0.35
    alpha = sign_mask * 0.55
    return screen_blend(img, overlay, alpha)


def delorean_car(img: np.ndarray, car_mask: np.ndarray) -> np.ndarray:
    """Brushed steel / chrome feel on car body."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    streak = (np.sin(xx / 18.0 + yy / 90.0) * 0.5 + 0.5).astype(np.float32)

    steel = img.copy()
    gray = steel.mean(axis=2, keepdims=True)
    steel = np.clip(gray * 0.55 + steel * 0.45 + (streak[..., None] - 0.5) * 0.08, 0, 1)

    chrome = np.zeros_like(img)
    chrome[..., 0] = 0.75
    chrome[..., 1] = 0.82
    chrome[..., 2] = 0.95
    alpha = car_mask * (0.35 + streak * 0.25)
    out = screen_blend(img, chrome, alpha * 0.22)
    out = out * (1 - car_mask[..., None] * 0.35) + steel * (car_mask[..., None] * 0.35)

    # Underglow
    under = np.zeros_like(out)
    under[..., 2] = 1.0
    under[..., 1] = 0.45
    glow_mask = car_mask.copy()
    glow_mask[int(h * 0.55):, :] *= 0.6
    out = screen_blend(out, under, glow_mask * 0.18)
    return out


def cyber_implants(img: np.ndarray, person_mask: np.ndarray) -> np.ndarray:
    """Subtle glowing implant lines on exposed skin area."""
    h, w = img.shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.float32)
    alpha = np.zeros((h, w), dtype=np.float32)

    # Temple chip (right side of frame where person stands)
    cx, cy = int(w * 0.915), int(h * 0.22)
    draw_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(draw_mask, (cx, cy), max(3, w // 120), 255, -1)
    cv2.line(draw_mask, (cx - w // 25, cy + h // 30), (cx + w // 40, cy + h // 18), 255, max(1, w // 200))

    # Forearm line
    x1, y1 = int(w * 0.88), int(h * 0.42)
    x2, y2 = int(w * 0.93), int(h * 0.52)
    cv2.line(draw_mask, (x1, y1), (x2, y2), 255, max(1, w // 180))

    glow = cv2.GaussianBlur(draw_mask, (0, 0), 2).astype(np.float32) / 255.0
    glow *= person_mask
    overlay[..., 0] = glow * 0.35
    overlay[..., 1] = glow * 0.95
    overlay[..., 2] = glow * 1.0
    alpha = glow * 0.65
    return screen_blend(img, overlay, alpha)


def wet_reflection(img: np.ndarray, neon_layer: np.ndarray, ground_mask: np.ndarray) -> np.ndarray:
    """Soft neon reflection on pavement."""
    h, w = img.shape[:2]
    refl = cv2.flip(neon_layer, 0)
    fade = np.linspace(1, 0, h)[:, None]
    fade = np.clip(fade * 1.4 - 0.2, 0, 1)
    alpha = ground_mask * fade * 0.28
    return screen_blend(img, refl, alpha)


def color_grade(img: np.ndarray) -> np.ndarray:
    """Light cyberpunk grade without crushing exposure."""
    out = img.copy()
    shadows = np.clip(1.0 - out, 0, 1)
    highlights = out
    out = np.clip(out + shadows * np.array([0.02, 0.00, 0.05]), 0, 1)
    out = np.clip(out + highlights * np.array([0.04, -0.01, 0.02]), 0, 1)
    # Slight vignette
    h, w = out.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt(((xx - cx) / w) ** 2 + ((yy - cy) / h) ** 2)
    vignette = np.clip(1.0 - dist * 0.35, 0.75, 1.0)
    out *= vignette[..., None]
    return out


def auto_masks(img: np.ndarray) -> dict[str, np.ndarray]:
    """Heuristic masks tuned for street photo: building top, car center, person right."""
    h, w = img.shape[:2]
    masks = {}

    yy, xx = np.mgrid[0:h, 0:w]
    yy = yy.astype(np.float32) / h
    xx = xx.astype(np.float32) / w

    building = np.clip(1.0 - (yy - 0.05) / 0.45, 0, 1)
    building = cv2.GaussianBlur(building, (0, 0), 8)
    masks["building"] = building

    # Green sign band (upper center)
    hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    green = ((hsv[..., 0] > 35) & (hsv[..., 0] < 95) & (hsv[..., 1] > 40)).astype(np.float32)
    green = cv2.GaussianBlur(green, (0, 0), 5)
    green *= (yy < 0.42).astype(np.float32)
    masks["sign"] = np.clip(green, 0, 1)

    # Car: lower-middle, dark blob detection
    car_region = ((yy > 0.35) & (yy < 0.92) & (xx > 0.08) & (xx < 0.78)).astype(np.float32)
    lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    dark = (lab[..., 0] < 95).astype(np.float32)
    car = car_region * dark
    car = cv2.GaussianBlur(car, (0, 0), 12)
    masks["car"] = np.clip(car, 0, 1)

    # Person: right edge vertical strip
    person = np.clip((xx - 0.78) / 0.22, 0, 1)
    person *= np.clip(1.0 - np.abs(yy - 0.55) / 0.45, 0, 1)
    person = cv2.GaussianBlur(person, (0, 0), 10)
    masks["person"] = np.clip(person, 0, 1)

    ground = np.clip((yy - 0.62) / 0.38, 0, 1)
    ground = cv2.GaussianBlur(ground, (0, 0), 15)
    masks["ground"] = ground

    return masks


def process(input_path: Path, output_path: Path) -> None:
    pil = Image.open(input_path).convert("RGB")
    img = np.asarray(pil, dtype=np.float32) / 255.0
    masks = auto_masks(img)

    neon_preview = add_neon_trim(img.copy(), masks["building"])
    result = boost_sign_neon(img, masks["sign"])
    result = add_neon_trim(result, masks["building"])
    result = delorean_car(result, masks["car"])
    result = cyber_implants(result, masks["person"])
    result = wet_reflection(result, neon_preview, masks["ground"])
    result = color_grade(result)

    out = (np.clip(result, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(out).save(output_path, quality=95)
    print(f"Saved: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Cyberpunk overlay edit (preserves original photo)")
    parser.add_argument("input", type=Path, help="Original photo path")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("/opt/cursor/artifacts/assets/cyberpunk-edited-original.png"),
        help="Output path",
    )
    args = parser.parse_args()
    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    process(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
