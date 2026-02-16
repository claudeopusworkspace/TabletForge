"""TabletForge - Generate ancient stone tablet images with custom fonts."""

from .renderer import render, TabletConfig

__version__ = "0.1.0"
__all__ = ["generate", "render", "TabletConfig"]


def generate(text, font_dir, height=256, seed=None, **kwargs):
    """Generate a stone tablet image with carved text.

    Args:
        text: Text to carve onto the tablet (A-Z and spaces).
        font_dir: Path to directory containing glyph_*.svg files.
        height: Output image height in pixels. Width is calculated
            automatically from text length (each character occupies
            a square cell).
        seed: Random seed for reproducible tablet generation.
        **kwargs: Additional TabletConfig parameters (stone_color,
            carve_depth, crack_density, edge_roughness, etc.).

    Returns:
        PIL Image in RGBA mode.
    """
    config = TabletConfig(**kwargs)
    return render(text, font_dir, height, seed=seed, config=config)
