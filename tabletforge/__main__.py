"""CLI entry point for TabletForge."""

import argparse
import sys
from pathlib import Path

from . import generate


def main():
    parser = argparse.ArgumentParser(
        description="Generate ancient stone tablet images with custom fonts"
    )
    parser.add_argument("text", help="Text to render on the tablet (A-Z, spaces)")
    parser.add_argument(
        "--font", "-f", required=True,
        help="Path to font directory containing glyph_*.svg files"
    )
    parser.add_argument(
        "--height", "-H", type=int, default=256,
        help="Output image height in pixels (default: 256)"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=None,
        help="Random seed for reproducible generation"
    )
    parser.add_argument(
        "--output", "-o", default="tablet.png",
        help="Output file path (default: tablet.png)"
    )
    parser.add_argument(
        "--stone-color", nargs=3, type=int, default=None,
        metavar=("R", "G", "B"),
        help="Base stone color as R G B (default: 180 165 140)"
    )
    parser.add_argument(
        "--weathering", "-w", type=float, default=None,
        help="Overall weathering intensity 0.0-1.0"
    )
    parser.add_argument(
        "--carve-depth", type=float, default=None,
        help="Carving depth 0.0-1.0 (default: 0.7)"
    )
    parser.add_argument(
        "--edge-roughness", type=float, default=None,
        help="Edge irregularity 0.0-1.0 (default: 0.4)"
    )

    args = parser.parse_args()

    kwargs = {}
    if args.stone_color:
        kwargs["stone_color"] = tuple(args.stone_color)
    if args.weathering is not None:
        kwargs["crack_density"] = args.weathering
        kwargs["pit_density"] = args.weathering
        kwargs["wear"] = args.weathering
        kwargs["edge_roughness"] = min(args.weathering, 0.8)
    if args.carve_depth is not None:
        kwargs["carve_depth"] = args.carve_depth
    if args.edge_roughness is not None:
        kwargs["edge_roughness"] = args.edge_roughness

    image = generate(
        text=args.text,
        font_dir=args.font,
        height=args.height,
        seed=args.seed,
        **kwargs,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(output))
    print(f"Saved tablet ({image.size[0]}x{image.size[1]}) to {output}")


if __name__ == "__main__":
    main()
