# TabletForge

Procedural stone tablet image generator. Takes SVG glyph fonts (like those from GlyphForge) and renders text carved into weathered stone tablets.

## Architecture

- `tabletforge/noise.py` - Value noise and fractal Brownian motion
- `tabletforge/font.py` - SVG font loading and glyph rasterization
- `tabletforge/renderer.py` - Main rendering pipeline (tablet shape, stone texture, carving, weathering, lighting)
- `tabletforge/__init__.py` - Public API (`generate()`)
- `tabletforge/__main__.py` - CLI entry point

## Conventions

- All randomness is seeded via numpy RandomState for reproducibility
- Image dimensions are height-driven: width is calculated from text length
- Each character occupies a square cell (height x height) within the text area
- Effects scale relative to image height so output looks consistent at any resolution
- Font format: directory of `glyph_*.svg` files with `<polygon>` elements (GlyphForge-compatible)

## Testing

```bash
python -m pytest tests/
```

## Usage

```bash
python -m tabletforge "HELLO" --font tests/fixtures/font --height 256 --seed 42 --output tablet.png
```
