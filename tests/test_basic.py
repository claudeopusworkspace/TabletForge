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


@pytest.mark.parametrize("seed", [1, 7, 42, 99, 200])
def test_polygon_seam_continuity(seed):
    """The displaced polygon must close smoothly — no large gap
    between the last point and the first point compared to the
    typical spacing between consecutive points."""
    from tabletforge.renderer import _build_tablet_polygon, TabletConfig

    config = TabletConfig()
    rng = np.random.RandomState(seed)
    poly = _build_tablet_polygon(w=800, h=256, config=config, rng=rng)
    assert len(poly) >= 3

    pts = np.array(poly)

    # Distances between consecutive points (including wrap)
    diffs = np.diff(pts, axis=0, append=pts[:1])
    step_dists = np.hypot(diffs[:, 0], diffs[:, 1])

    # The closing step (last → first) is the last entry
    closing_dist = step_dists[-1]
    median_dist = np.median(step_dists)

    # The closing gap must not be more than 4x the median step.
    # A well-blended seam will be comparable to nearby steps.
    assert closing_dist < median_dist * 4, (
        f"seed {seed}: closing gap {closing_dist:.1f} is >"
        f" 4x median step {median_dist:.1f}"
    )
