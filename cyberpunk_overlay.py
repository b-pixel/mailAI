#!/usr/bin/env python3
"""Layered cyberpunk photo edit — preserves original pixels, adds local effects only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def screen_blend(base: np.ndarray, overlay: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha[..., None]
    out = 1.0 - (1.0 - base) * (1.0 - overlay)
    return np.clip(base * (1.0 - a) + out * a, 0.0, 1.0)


def add_neon_trim(img: np.ndarray, region_mask: np.ndarray, strength: float = 0.55) -> np.ndarray:
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 120)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1).astype(np.float32) / 255.0
    edges *= region_mask

    h = img.shape[0]
    y_idx = np.linspace(0, 1, h)[:, None]
    overlay = np.zeros_like(img)
    overlay[..., 0] = edges * (0.35 + 0.5 * np.clip(1 - np.abs(y_idx - 0.35) * 3, 0, 1))
    overlay[..., 1] = edges * 0.25
    overlay[..., 2] = edges * (0.65 + 0.35 * np.clip(1 - np.abs(y_idx - 0.30) * 3, 0, 1))

    glow = cv2.GaussianBlur((edges * 255).astype(np.uint8), (0, 0), 4).astype(np.float32) / 255.0
    alpha = np.clip(edges * strength + glow * 0.18, 0, 1) * region_mask
    return screen_blend(img, overlay, alpha)


def boost_sign_neon(img: np.ndarray, sign_mask: np.ndarray) -> np.ndarray:
    overlay = np.zeros_like(img)
    overlay[..., 1] = 1.0
    overlay[..., 2] = 0.25
    return screen_blend(img, overlay, sign_mask * 0.45)


def delorean_car(img: np.ndarray, car_mask: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    streak = (np.sin(xx / 16.0 + yy / 75.0) * 0.5 + 0.5).astype(np.float32)

    gray = img.mean(axis=2, keepdims=True)
    steel = np.clip(gray * 0.62 + img * 0.38 + (streak[..., None] - 0.5) * 0.06, 0, 1)

    chrome = np.zeros_like(img)
    chrome[..., :] = np.array([0.78, 0.84, 0.95])
    out = screen_blend(img, chrome, car_mask * (0.12 + streak * 0.12))
    out = out * (1 - car_mask[..., None] * 0.28) + steel * (car_mask[..., None] * 0.28)

    under = np.zeros_like(out)
    under[..., 1], under[..., 2] = 0.35, 0.95
    out = screen_blend(out, under, car_mask * 0.10)
    return out


def cyber_implants(img: np.ndarray, person_mask: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    draw = np.zeros((h, w), dtype=np.uint8)

    # Woman on far right: temple (hair bun) + forearm (coffee cup hand)
    temple = (int(w * 0.905), int(h * 0.19))
    forearm = (int(w * 0.885), int(h * 0.46))
    wrist = (int(w * 0.905), int(h * 0.52))

    cv2.circle(draw, temple, max(2, w // 140), 255, -1)
    cv2.line(draw, (temple[0] - w // 35, temple[1] + h // 40), (temple[0] + w // 50, temple[1] + h // 25), 255, max(1, w // 220))
    cv2.line(draw, forearm, wrist, 255, max(1, w // 200))

    glow = cv2.GaussianBlur(draw, (0, 0), 2).astype(np.float32) / 255.0
    glow *= person_mask
    overlay = np.zeros_like(img)
    overlay[..., 1] = glow * 0.95
    overlay[..., 2] = glow * 1.0
    overlay[..., 0] = glow * 0.25
    return screen_blend(img, overlay, glow * 0.55)


def soft_reflection(img: np.ndarray, glow_source: np.ndarray, ground_mask: np.ndarray) -> np.ndarray:
    h = img.shape[0]
    refl = cv2.flip(glow_source, 0)
    fade = np.clip(np.linspace(1.1, 0.0, h)[:, None] - 0.15, 0, 1)
    return screen_blend(img, refl, ground_mask * fade * 0.15)


def color_grade(img: np.ndarray) -> np.ndarray:
    """Subtle grade — keeps daytime exposure."""
    out = img.copy()
    out = np.clip(out + out * np.array([0.015, -0.005, 0.025]), 0, 1)
    shadows = np.clip(1.0 - out, 0, 1)
    out = np.clip(out + shadows * np.array([0.01, 0.0, 0.02]), 0, 1)
    return out


def auto_masks(img: np.ndarray) -> dict[str, np.ndarray]:
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    yy = yy.astype(np.float32) / h
    xx = xx.astype(np.float32) / w

    building = np.clip(1.0 - (yy - 0.02) / 0.50, 0, 1)
    building = cv2.GaussianBlur(building, (0, 0), 10)

    hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    green = ((hsv[..., 0] > 30) & (hsv[..., 0] < 95) & (hsv[..., 1] > 35)).astype(np.float32)
    green = cv2.GaussianBlur(green, (0, 0), 6)
    green *= (yy < 0.40).astype(np.float32)

    lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    car_region = ((yy > 0.32) & (yy < 0.90) & (xx > 0.05) & (xx < 0.78)).astype(np.float32)
    car = car_region * (lab[..., 0] < 100).astype(np.float32)
    car = cv2.GaussianBlur(car, (0, 0), 14)

    # Far-right silhouette (girl with coffee cup)
    person = np.clip((xx - 0.80) / 0.20, 0, 1)
    person *= np.clip(1.0 - np.abs(yy - 0.52) / 0.42, 0, 1)
    person = cv2.GaussianBlur(person, (0, 0), 8)

    ground = np.clip((yy - 0.58) / 0.42, 0, 1)
    ground = cv2.GaussianBlur(ground, (0, 0), 12)

    return {
        "building": building,
        "sign": np.clip(green, 0, 1),
        "car": np.clip(car, 0, 1),
        "person": np.clip(person, 0, 1),
        "ground": ground,
    }


def process(input_path: Path, output_path: Path) -> None:
    img = np.asarray(Image.open(input_path).convert("RGB"), dtype=np.float32) / 255.0
    masks = auto_masks(img)

    neon_layer = add_neon_trim(img.copy(), masks["building"], strength=0.0)  # build glow source
    neon_layer = boost_sign_neon(img, masks["sign"])

    result = boost_sign_neon(img, masks["sign"])
    result = add_neon_trim(result, masks["building"], strength=0.50)
    result = delorean_car(result, masks["car"])
    result = cyber_implants(result, masks["person"])
    result = soft_reflection(result, neon_layer, masks["ground"])
    result = color_grade(result)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(result, 0, 1) * 255).astype(np.uint8)).save(output_path, quality=95)
    print(f"Saved: {output_path} ({img.shape[1]}x{img.shape[0]})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Cyberpunk overlay edit (preserves original photo)")
    parser.add_argument("input", type=Path, help="Original photo path")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("/opt/cursor/artifacts/assets/cyberpunk-edited-original.png"),
    )
    args = parser.parse_args()
    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 1
    process(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
