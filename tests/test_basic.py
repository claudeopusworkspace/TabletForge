"""Basic tests for TabletForge."""

from pathlib import Path

import numpy as np
import pytest

FONT_DIR = Path(__file__).parent / "fixtures" / "font"


def test_font_loading():
    from tabletforge.font import load_font
    font = load_font(FONT_DIR)
    assert len(font) == 26
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        assert letter in font
        assert len(font[letter].polygons) > 0


def test_glyph_rasterize():
    from tabletforge.font import load_font, rasterize_glyph
    font = load_font(FONT_DIR)
    arr = rasterize_glyph(font["A"], (100, 100))
    assert arr.shape == (100, 100)
    assert arr.max() > 0  # glyph has content


def test_generate_basic():
    from tabletforge import generate
    img = generate("TEST", font_dir=str(FONT_DIR), height=128, seed=42)
    assert img.mode == "RGBA"
    assert img.size[1] == 128
    assert img.size[0] > 0


def test_generate_single_char():
    from tabletforge import generate
    img = generate("A", font_dir=str(FONT_DIR), height=64, seed=1)
    assert img.mode == "RGBA"
    assert img.size[1] == 64


def test_generate_with_spaces():
    from tabletforge import generate
    img = generate("A B", font_dir=str(FONT_DIR), height=64, seed=1)
    assert img.mode == "RGBA"


def test_reproducibility():
    from tabletforge import generate
    img1 = generate("HI", font_dir=str(FONT_DIR), height=64, seed=99)
    img2 = generate("HI", font_dir=str(FONT_DIR), height=64, seed=99)
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    np.testing.assert_array_equal(arr1, arr2)


def test_different_seeds_differ():
    from tabletforge import generate
    img1 = generate("HI", font_dir=str(FONT_DIR), height=64, seed=1)
    img2 = generate("HI", font_dir=str(FONT_DIR), height=64, seed=2)
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    assert not np.array_equal(arr1, arr2)
