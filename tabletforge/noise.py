"""Procedural noise generation for texture and shape effects."""

import numpy as np


def _fade(t):
    """Perlin fade function: 6t^5 - 15t^4 + 10t^3."""
    return t * t * t * (t * (t * 6 - 15) + 10)


def value_noise_2d(height, width, scale=1.0, rng=None):
    """Generate 2D value noise with smooth interpolation.

    Args:
        height: Output height in pixels.
        width: Output width in pixels.
        scale: Noise cell size in pixels (larger = smoother).
        rng: numpy RandomState for reproducibility.

    Returns:
        Array of shape (height, width) with values in [0, 1].
    """
    if rng is None:
        rng = np.random.RandomState()

    scale = max(scale, 1.0)

    # Grid dimensions (number of random lattice points)
    gh = int(np.ceil(height / scale)) + 2
    gw = int(np.ceil(width / scale)) + 2

    # Random values at lattice points
    grid = rng.random_sample((gh, gw))

    # Continuous sample coordinates in grid space
    y_coords = np.linspace(0, (height - 1) / scale, height)
    x_coords = np.linspace(0, (width - 1) / scale, width)

    # Integer and fractional parts
    yi = np.floor(y_coords).astype(int)
    xi = np.floor(x_coords).astype(int)
    yf = _fade(y_coords - yi)
    xf = _fade(x_coords - xi)

    # Build meshgrids for vectorized lookup
    yi_m, xi_m = np.meshgrid(yi, xi, indexing='ij')
    uy_m, ux_m = np.meshgrid(yf, xf, indexing='ij')

    # Bilinear interpolation of lattice values
    v00 = grid[yi_m, xi_m]
    v10 = grid[yi_m + 1, xi_m]
    v01 = grid[yi_m, xi_m + 1]
    v11 = grid[yi_m + 1, xi_m + 1]

    v0 = v00 + ux_m * (v01 - v00)
    v1 = v10 + ux_m * (v11 - v10)
    result = v0 + uy_m * (v1 - v0)

    return result


def fbm_2d(height, width, octaves=6, base_scale=64.0, persistence=0.5,
           lacunarity=2.0, rng=None):
    """Generate fractal Brownian motion noise (layered value noise).

    Args:
        height: Output height in pixels.
        width: Output width in pixels.
        octaves: Number of noise layers.
        base_scale: Scale of the coarsest octave.
        persistence: Amplitude decay per octave (0-1).
        lacunarity: Frequency multiplier per octave.
        rng: numpy RandomState for reproducibility.

    Returns:
        Array of shape (height, width) with values in [0, 1].
    """
    if rng is None:
        rng = np.random.RandomState()

    result = np.zeros((height, width), dtype=np.float64)
    amplitude = 1.0
    total_amplitude = 0.0
    scale = base_scale

    for _ in range(octaves):
        noise = value_noise_2d(height, width, scale=scale, rng=rng)
        result += amplitude * noise
        total_amplitude += amplitude
        amplitude *= persistence
        scale /= lacunarity

    return result / total_amplitude
