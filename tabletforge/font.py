"""Font loading from SVG glyph files."""

import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class Glyph:
    """A single glyph loaded from SVG."""
    label: str
    polygons: list  # List of list of (x, y) tuples, normalized to 0-1
    viewbox_w: float
    viewbox_h: float


def load_glyph(svg_path):
    """Load a single glyph from an SVG file.

    Parses <polygon> elements and normalizes coordinates to [0, 1].
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # Strip namespace prefixes for simpler querying
    for elem in root.iter():
        if '}' in elem.tag:
            elem.tag = elem.tag.split('}', 1)[1]
        for key in list(elem.attrib):
            if '}' in key:
                elem.attrib[key.split('}', 1)[1]] = elem.attrib.pop(key)

    # Parse viewBox
    viewbox = root.get('viewBox', '0 0 200 200')
    parts = viewbox.split()
    vb_w = float(parts[2])
    vb_h = float(parts[3])

    # Parse all polygon elements
    polygons = []
    for poly in root.iter('polygon'):
        points_str = poly.get('points', '')
        points = []
        for pair in points_str.strip().split():
            coords = pair.split(',')
            if len(coords) == 2:
                x = float(coords[0]) / vb_w
                y = float(coords[1]) / vb_h
                points.append((x, y))
        if len(points) >= 3:
            polygons.append(points)

    # Extract label from filename (glyph_a.svg -> A)
    label = Path(svg_path).stem
    if label.startswith('glyph_'):
        label = label[6:]
    label = label.upper()

    return Glyph(label=label, polygons=polygons,
                 viewbox_w=vb_w, viewbox_h=vb_h)


def load_font(font_dir):
    """Load all glyphs from a directory of SVG files.

    Looks for files named glyph_*.svg. Returns a dict mapping
    uppercase letter to Glyph object.
    """
    font_dir = Path(font_dir)
    glyphs = {}

    for svg_file in sorted(font_dir.glob('glyph_*.svg')):
        glyph = load_glyph(svg_file)
        glyphs[glyph.label] = glyph

    return glyphs


def rasterize_glyph(glyph, size):
    """Rasterize a glyph to a grayscale alpha mask.

    Args:
        glyph: Glyph object with normalized polygon coordinates.
        size: (width, height) tuple for the output image.

    Returns:
        numpy array of shape (height, width), values 0-255.
    """
    w, h = size
    img = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(img)

    for polygon in glyph.polygons:
        scaled = [(x * w, y * h) for x, y in polygon]
        if len(scaled) >= 3:
            draw.polygon(scaled, fill=255)

    return np.array(img)
