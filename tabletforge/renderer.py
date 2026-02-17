"""Main tablet rendering pipeline.

Generates a stone tablet image with carved text, surface texture,
weathering effects, and directional lighting.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from dataclasses import dataclass

from .noise import fbm_2d, value_noise_2d
from .font import load_font, rasterize_glyph


@dataclass
class TabletConfig:
    """Configuration for tablet generation."""

    # Tablet shape
    padding: float = 0.15
    edge_roughness: float = 0.4

    # Stone appearance
    stone_color: tuple = (180, 165, 140)
    color_variation: float = 0.15
    grain_intensity: float = 0.08

    # Carving
    carve_depth: float = 0.7
    carve_roughness: float = 0.3

    # Weathering
    crack_density: float = 0.4
    pit_density: float = 0.3
    wear: float = 0.5

    # Edge bevel
    edge_bevel: float = 0.6

    # Lighting
    light_angle: float = 315.0
    light_intensity: float = 0.6
    ambient: float = 0.4

    # Mineral veins
    vein_count: int = 3
    vein_intensity: float = 0.1


def render(text, font_dir, height, seed=None, config=None):
    """Render text on a weathered stone tablet.

    Args:
        text: Text string to carve (A-Z and spaces).
        font_dir: Path to directory of glyph_*.svg files.
        height: Output image height in pixels.
        seed: Random seed for reproducible generation.
        config: TabletConfig instance (defaults used if None).

    Returns:
        PIL Image in RGBA mode.
    """
    if config is None:
        config = TabletConfig()
    if seed is None:
        seed = np.random.randint(0, 2**31)

    rng = np.random.RandomState(seed)
    text = text.upper()

    # Load font glyphs
    font = load_font(font_dir)

    # Calculate dimensions
    pad_px = int(height * config.padding)
    text_height = height - 2 * pad_px
    cell_size = max(1, text_height)

    # Total text width
    text_width = 0
    for ch in text:
        if ch == ' ':
            text_width += cell_size // 2
        else:
            text_width += cell_size

    canvas_w = text_width + 2 * pad_px
    canvas_h = height

    # Ensure minimum dimensions
    canvas_w = max(canvas_w, cell_size)
    canvas_h = max(canvas_h, cell_size)

    # --- Pipeline ---

    # 1. Tablet shape mask
    tablet_mask = _generate_tablet_mask(canvas_w, canvas_h, config, rng)

    # 2. Stone surface texture
    stone = _generate_stone_texture(canvas_w, canvas_h, config, rng)

    # 3. Mineral veins
    stone = _apply_veins(stone, canvas_w, canvas_h, config, rng)

    # 4. Text mask
    text_mask = _render_text(text, font, cell_size, pad_px,
                             canvas_w, canvas_h, config, rng)

    # 5. Carving effect
    stone = _apply_carving(stone, text_mask, config, canvas_h)

    # 6. Weathering (cracks – may cut into the tablet mask)
    stone, tablet_mask = _apply_weathering(stone, tablet_mask, config, rng)

    # 7. Edge bevel (3D raised-slab look)
    stone = _apply_edge_bevel(stone, tablet_mask, config, canvas_h)

    # 8. Edge darkening
    stone = _apply_edge_darkening(stone, tablet_mask, canvas_h)

    # 9. Final composite
    return _composite(stone, tablet_mask)


# ---------------------------------------------------------------------------
# Internal pipeline stages
# ---------------------------------------------------------------------------

def _build_tablet_polygon(w, h, config, rng):
    """Build the displaced polygon for a crumbly tablet outline.

    Returns a list of (x, y) tuples tracing the tablet perimeter.
    Separated from mask rendering so it can be tested directly.
    """
    # Random corner radius, scaled to image size
    corner_r = rng.uniform(h * 0.02, h * 0.15)
    corner_r = min(corner_r, min(w, h) * 0.2)

    # Margin so outward displacement stays on canvas
    margin = max(4, int(h * 0.12 * config.edge_roughness))

    # Base rounded-rect bounds
    x0, y0 = float(margin), float(margin)
    x1, y1 = float(w - 1 - margin), float(h - 1 - margin)
    r = min(corner_r, (x1 - x0) / 2, (y1 - y0) / 2)

    # --- Trace perimeter (clockwise) as (x, y, nx, ny) ---
    step = max(1.0, h * 0.004)
    perimeter = []

    def _arc(cx, cy, start_angle, end_angle):
        """Append arc points with radial outward normals."""
        n_arc = max(4, int(r * abs(end_angle - start_angle) / step))
        for a in np.linspace(start_angle, end_angle, n_arc):
            perimeter.append((cx + r * np.cos(a), cy + r * np.sin(a),
                              np.cos(a), np.sin(a)))

    # Top edge (left -> right)
    for x in np.arange(x0 + r, x1 - r + 0.5, step):
        perimeter.append((x, y0, 0.0, -1.0))
    # Top-right corner
    _arc(x1 - r, y0 + r, -np.pi / 2, 0)
    # Right edge (top -> bottom)
    for y in np.arange(y0 + r, y1 - r + 0.5, step):
        perimeter.append((x1, y, 1.0, 0.0))
    # Bottom-right corner
    _arc(x1 - r, y1 - r, 0, np.pi / 2)
    # Bottom edge (right -> left)
    for x in np.arange(x1 - r, x0 + r - 0.5, -step):
        perimeter.append((x, y1, 0.0, 1.0))
    # Bottom-left corner
    _arc(x0 + r, y1 - r, np.pi / 2, np.pi)
    # Left edge (bottom -> top)
    for y in np.arange(y1 - r, y0 + r - 0.5, -step):
        perimeter.append((x0, y, -1.0, 0.0))
    # Top-left corner
    _arc(x0 + r, y0 + r, np.pi, 3 * np.pi / 2)

    n_pts = len(perimeter)
    if n_pts < 3:
        return []

    # --- 1D fractal noise along the perimeter ---
    amp_large = h * 0.20 * config.edge_roughness

    # Fine amplitude varies along the perimeter (some zones smooth,
    # some jagged) using a slow modulation signal
    amp_fine_mod = fbm_2d(1, n_pts, octaves=2,
                          base_scale=max(2, n_pts * 0.15),
                          persistence=0.5, rng=rng)[0]
    amp_fine = h * (0.10 + amp_fine_mod * 0.30) * config.edge_roughness

    lg = fbm_2d(1, n_pts, octaves=3,
                base_scale=max(2, n_pts * 0.25),
                persistence=0.5, rng=rng)[0]
    fn = fbm_2d(1, n_pts, octaves=6,
                base_scale=max(2, n_pts * 0.05),
                persistence=0.55, rng=rng)[0]
    disp = (lg - 0.5) * amp_large + (fn - 0.5) * amp_fine

    # --- Random bite indentations along the perimeter ---
    n_bites = max(0, rng.poisson(6))
    indices = np.arange(n_pts, dtype=np.float64)

    for _ in range(n_bites):
        center = rng.randint(0, n_pts)
        width = rng.uniform(n_pts * 0.01, n_pts * 0.06)
        depth = h * rng.uniform(0.02, 0.10) * config.edge_roughness

        dist = np.abs(indices - center)
        dist = np.minimum(dist, n_pts - dist)

        bite = depth * np.exp(-0.5 * (dist / max(1.0, width)) ** 2)

        if rng.random() > 0.4:
            jag = fbm_2d(1, n_pts, octaves=4,
                         base_scale=max(2, n_pts * 0.03),
                         persistence=0.5, rng=rng)[0]
            jag_amp = depth * rng.uniform(0.2, 0.6)
            bite += (jag - 0.5) * jag_amp * (bite / max(depth, 1e-6))

        disp -= bite

    # Blend the seam: smoothly ramp the last blend_n displacement
    # values toward disp[0] so the polygon closes without a step.
    blend_n = max(8, n_pts // 20)
    for i in range(blend_n):
        t = (i + 1) / (blend_n + 1)
        idx = n_pts - blend_n + i
        disp[idx] = disp[idx] * (1.0 - t) + disp[0] * t

    # Displace each point along its outward normal
    return [(px + nx * d, py + ny * d)
            for (px, py, nx, ny), d in zip(perimeter, disp)]


def _generate_tablet_mask(w, h, config, rng):
    """Generate the tablet silhouette with crumbly, irregular edges.

    Traces the perimeter of a rounded rectangle (with a random corner
    radius driven by the seed) and displaces each point along its
    outward normal using fractal noise.
    """
    if config.edge_roughness <= 0:
        corner_r = rng.uniform(h * 0.02, h * 0.15)
        corner_r = min(corner_r, min(w, h) * 0.2)
        img = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle([0, 0, w - 1, h - 1],
                               radius=int(corner_r), fill=255)
        return img

    poly = _build_tablet_polygon(w, h, config, rng)
    if len(poly) < 3:
        return Image.new('L', (w, h), 0)

    img = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(img)
    draw.polygon(poly, fill=255)

    # Minimal anti-alias blur (just 1px to soften staircase)
    img = img.filter(ImageFilter.GaussianBlur(radius=1))

    return img


def _generate_stone_texture(w, h, config, rng):
    """Build the base stone surface as an (h, w, 3) float64 array in [0,1]."""
    base_r, base_g, base_b = [c / 255.0 for c in config.stone_color]

    # Large-scale colour variation (blotches)
    variation = fbm_2d(h, w, octaves=4, base_scale=h * 0.5,
                       persistence=0.5, rng=rng)
    variation = (variation - 0.5) * config.color_variation * 2

    # Medium-scale grain
    grain = fbm_2d(h, w, octaves=3, base_scale=max(1, h * 0.05),
                   persistence=0.4, rng=rng)
    grain = (grain - 0.5) * config.grain_intensity * 2

    # Very fine pitting / micro-texture
    fine = value_noise_2d(h, w, scale=3.0, rng=rng)
    fine = (fine - 0.5) * 0.04

    combined = variation + grain + fine

    # Per-channel assembly with subtle colour drift in the grain
    r = np.clip(base_r + combined + grain * 0.02, 0, 1)
    g = np.clip(base_g + combined, 0, 1)
    b = np.clip(base_b + combined - grain * 0.01, 0, 1)

    return np.stack([r, g, b], axis=-1)


def _render_text(text, font, cell_size, pad, canvas_w, canvas_h, config, rng):
    """Rasterize text string to a float64 mask (1 = carved)."""
    mask = np.zeros((canvas_h, canvas_w), dtype=np.float64)

    x_offset = pad
    y_offset = pad

    for ch in text:
        if ch == ' ':
            x_offset += cell_size // 2
            continue
        if ch not in font:
            x_offset += cell_size
            continue

        glyph = font[ch]
        glyph_arr = rasterize_glyph(glyph, (cell_size, cell_size))
        glyph_mask = glyph_arr.astype(np.float64) / 255.0

        # Compute paste region (clipped to canvas)
        y1, y2 = y_offset, y_offset + cell_size
        x1, x2 = x_offset, x_offset + cell_size
        gy1, gx1 = max(0, -y1), max(0, -x1)
        gy2 = cell_size - max(0, y2 - canvas_h)
        gx2 = cell_size - max(0, x2 - canvas_w)
        cy1, cx1 = max(0, y1), max(0, x1)
        cy2, cx2 = min(canvas_h, y2), min(canvas_w, x2)

        if cy2 > cy1 and cx2 > cx1:
            mask[cy1:cy2, cx1:cx2] = np.maximum(
                mask[cy1:cy2, cx1:cx2],
                glyph_mask[gy1:gy2, gx1:gx2]
            )

        x_offset += cell_size

    # Apply carve roughness – erode text edges with noise
    if config.carve_roughness > 0:
        blur_r = max(1, int(cell_size * 0.015 * (1 + config.carve_roughness)))
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=blur_r))
        mask = np.array(mask_img, dtype=np.float64) / 255.0

        edge_noise = fbm_2d(canvas_h, canvas_w, octaves=3,
                            base_scale=max(1, cell_size * 0.1),
                            persistence=0.5, rng=rng)
        threshold = 0.5 - config.carve_roughness * 0.15
        noise_mod = (edge_noise - 0.5) * config.carve_roughness * 0.3
        mask = (mask > (threshold + noise_mod)).astype(np.float64)

    return mask


def _apply_carving(stone, text_mask, config, height):
    """Emboss carved text into the stone surface."""
    if text_mask.max() < 0.01:
        return stone

    # Light direction (shared by both passes)
    rad = np.radians(config.light_angle)
    light_x = np.cos(rad)
    light_y = -np.sin(rad)

    # --- Sharp emboss (tight blur for crisp edge highlights) ---
    sharp_r = max(1, int(height * 0.006 * config.carve_depth))
    sharp_img = Image.fromarray((text_mask * 255).astype(np.uint8))
    sharp_depth = np.array(
        sharp_img.filter(ImageFilter.GaussianBlur(radius=sharp_r)),
        dtype=np.float64
    ) / 255.0

    dy_s = np.zeros_like(sharp_depth)
    dx_s = np.zeros_like(sharp_depth)
    dy_s[1:, :] = sharp_depth[1:, :] - sharp_depth[:-1, :]
    dx_s[:, 1:] = sharp_depth[:, 1:] - sharp_depth[:, :-1]
    sharp_emboss = dx_s * light_x + dy_s * light_y
    max_s = np.abs(sharp_emboss).max()
    if max_s > 0:
        sharp_emboss /= max_s

    # --- Wide emboss (broader blur for ambient depth illusion) ---
    wide_r = max(2, int(height * 0.02 * config.carve_depth))
    wide_img = Image.fromarray((text_mask * 255).astype(np.uint8))
    wide_depth = np.array(
        wide_img.filter(ImageFilter.GaussianBlur(radius=wide_r)),
        dtype=np.float64
    ) / 255.0

    dy_w = np.zeros_like(wide_depth)
    dx_w = np.zeros_like(wide_depth)
    dy_w[1:, :] = wide_depth[1:, :] - wide_depth[:-1, :]
    dx_w[:, 1:] = wide_depth[:, 1:] - wide_depth[:, :-1]
    wide_emboss = dx_w * light_x + dy_w * light_y
    max_w = np.abs(wide_emboss).max()
    if max_w > 0:
        wide_emboss /= max_w

    # Combine both emboss passes (negate for incised/carved look)
    emboss = -(sharp_emboss * 0.6 + wide_emboss * 0.4)
    emboss *= config.light_intensity * config.carve_depth

    # 1. Darken carved areas (interior of carved text is in shadow)
    darken = 1.0 - text_mask * 0.35 * config.carve_depth
    stone = stone * darken[:, :, np.newaxis]

    # 2. Inner shadow: darkened halo near carved edges
    inner_r = max(1, int(height * 0.01))
    inner_img = Image.fromarray((text_mask * 255).astype(np.uint8))
    inner_blur = np.array(
        inner_img.filter(ImageFilter.GaussianBlur(radius=inner_r)),
        dtype=np.float64
    ) / 255.0
    # Shadow is strongest at edges (where blur < mask)
    inner_shadow = np.clip(text_mask - inner_blur * 0.6, 0, 1)
    stone *= (1.0 - inner_shadow * 0.15 * config.carve_depth)[:, :, np.newaxis]

    # 3. Emboss highlights and shadows
    stone += emboss[:, :, np.newaxis] * 0.6

    # 4. Subtle colour shift in carved areas (cooler)
    stone[:, :, 0] -= text_mask * 0.03
    stone[:, :, 2] += text_mask * 0.015

    return np.clip(stone, 0, 1)


def _apply_veins(stone, w, h, config, rng):
    """Draw faint mineral veins across the stone surface."""
    if config.vein_count <= 0 or config.vein_intensity <= 0:
        return stone

    vein_img = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(vein_img)
    step = max(2, h // 50)

    for _ in range(config.vein_count):
        # Start from a random edge point
        if rng.random() > 0.5:
            x, y = 0.0, float(rng.randint(0, h))
            angle = rng.uniform(-0.3, 0.3)
        else:
            x, y = float(rng.randint(0, w)), 0.0
            angle = rng.uniform(1.2, 1.9)

        points = [(int(x), int(y))]
        for _ in range(300):
            x += np.cos(angle) * step
            y += np.sin(angle) * step
            angle += rng.uniform(-0.15, 0.15)
            if x < 0 or x >= w or y < 0 or y >= h:
                break
            points.append((int(x), int(y)))

        if len(points) > 1:
            lw = max(1, int(h * 0.003))
            draw.line(points, fill=255, width=lw)

    # Soften veins
    blur_r = max(1, h // 80)
    vein_mask = np.array(
        vein_img.filter(ImageFilter.GaussianBlur(radius=blur_r)),
        dtype=np.float64
    ) / 255.0

    effect = vein_mask * config.vein_intensity
    for c in range(3):
        stone[:, :, c] = np.clip(
            stone[:, :, c] + effect * (0.8 + c * 0.1), 0, 1
        )

    return stone


def _apply_weathering(stone, tablet_mask, config, rng):
    """Add cracks and pitting to the stone surface.

    Cracks originate from the tablet perimeter and taper inward:
      - First 20%: wide V-shaped bite, very rough/jagged edges
      - Middle 60%: slightly wider than 1px
      - Last 20%: tapers to 1px at the tip

    Where a crack is wider than 10 px, the inner portion (beyond a
    5 px rim on each side) is punched through the tablet mask so
    the background shows through the fissure.

    Returns (stone, tablet_mask) — the mask may be modified.
    """
    h, w = stone.shape[:2]
    mask_arr = np.array(tablet_mask, dtype=np.float64) / 255.0

    if config.crack_density > 0:
        crack_img = Image.new('L', (w, h), 0)
        crack_draw = ImageDraw.Draw(crack_img)
        cutout_img = Image.new('L', (w, h), 0)
        cutout_draw = ImageDraw.Draw(cutout_img)

        n_cracks = max(0, int(config.crack_density * 8 * rng.poisson(2)))

        if n_cracks > 0:
            # Find edge pixels (interior with at least one exterior neighbor)
            interior = mask_arr > 0.5
            fully_interior = interior.copy()
            fully_interior[1:, :] &= interior[:-1, :]
            fully_interior[:-1, :] &= interior[1:, :]
            fully_interior[:, 1:] &= interior[:, :-1]
            fully_interior[:, :-1] &= interior[:, 1:]
            edge_pixels = interior & ~fully_interior
            edge_ys, edge_xs = np.where(edge_pixels)

            if len(edge_ys) > 0:
                # Blurred mask gradient → inward direction
                blur_r = max(3, h // 20)
                dist_arr = np.array(
                    tablet_mask.filter(
                        ImageFilter.GaussianBlur(radius=blur_r)),
                    dtype=np.float64
                ) / 255.0
                grad_y = np.zeros_like(dist_arr)
                grad_x = np.zeros_like(dist_arr)
                grad_y[1:-1, :] = (dist_arr[2:, :] - dist_arr[:-2, :]) / 2
                grad_x[:, 1:-1] = (dist_arr[:, 2:] - dist_arr[:, :-2]) / 2

                for _ in range(n_cracks):
                    idx = rng.randint(0, len(edge_ys))
                    sx, sy = float(edge_xs[idx]), float(edge_ys[idx])

                    # Walk direction: generally inward + randomness
                    iy_c = min(h - 1, max(0, int(sy)))
                    ix_c = min(w - 1, max(0, int(sx)))
                    gx = grad_x[iy_c, ix_c]
                    gy = grad_y[iy_c, ix_c]
                    if abs(gx) + abs(gy) > 0.001:
                        angle = np.arctan2(gy, gx) + rng.uniform(-0.5, 0.5)
                    else:
                        angle = rng.uniform(0, 2 * np.pi)

                    crack_len = int(h * rng.uniform(0.05, 0.25))
                    pts = [(sx, sy)]

                    for _ in range(crack_len):
                        sx += np.cos(angle) * 1.5
                        sy += np.sin(angle) * 1.5
                        angle += rng.uniform(-0.3, 0.3)
                        ix, iy = int(sx), int(sy)
                        if ix < 0 or ix >= w or iy < 0 or iy >= h:
                            break
                        if mask_arr[iy, ix] < 0.5:
                            break
                        pts.append((sx, sy))

                    if len(pts) < 3:
                        continue

                    # --- Build variable-width polygon along the crack ---
                    total = len(pts)
                    v_width = h * rng.uniform(0.015, 0.04)
                    mid_width = h * rng.uniform(0.004, 0.010)

                    left_side = []
                    right_side = []
                    left_hws = []
                    right_hws = []
                    norms = []

                    for i, (px, py) in enumerate(pts):
                        t = i / max(1, total - 1)

                        # Width profile
                        if t < 0.2:
                            frac = t / 0.2
                            base_hw = (v_width * (1 - frac)
                                       + mid_width * frac) / 2
                        elif t < 0.8:
                            base_hw = mid_width / 2
                        else:
                            frac = (t - 0.8) / 0.2
                            base_hw = (mid_width * (1 - frac)
                                       + 1.0 * frac) / 2

                        # Jaggedness (independent per side)
                        if t < 0.25:
                            jag_amp = base_hw * 0.6
                        elif t < 0.8:
                            jag_amp = base_hw * 0.15
                        else:
                            jag_amp = 0.0

                        lhw = max(0.3, base_hw
                                  + rng.uniform(-jag_amp, jag_amp))
                        rhw = max(0.3, base_hw
                                  + rng.uniform(-jag_amp, jag_amp))

                        # Perpendicular direction
                        if i < total - 1:
                            dx = pts[i + 1][0] - px
                            dy = pts[i + 1][1] - py
                        else:
                            dx = px - pts[i - 1][0]
                            dy = py - pts[i - 1][1]
                        mag = max(0.001, np.sqrt(dx * dx + dy * dy))
                        nx, ny = -dy / mag, dx / mag

                        left_hws.append(lhw)
                        right_hws.append(rhw)
                        norms.append((nx, ny))
                        left_side.append(
                            (px + nx * lhw, py + ny * lhw))
                        right_side.append(
                            (px - nx * rhw, py - ny * rhw))

                    # Draw full crack on the darkening layer
                    poly = left_side + right_side[::-1]
                    if len(poly) >= 3:
                        crack_draw.polygon(
                            [(int(round(x)), int(round(y)))
                             for x, y in poly],
                            fill=200,
                        )

                    # Cutout: transparent middle where crack is
                    # wider than 2*rim.  5 px of stone stays visible
                    # on each side; the interior is punched out.
                    rim = 5.0
                    co_left = []
                    co_right = []
                    has_cutout = False

                    for i, (px, py) in enumerate(pts):
                        nx, ny = norms[i]
                        inner_l = max(0.0, left_hws[i] - rim)
                        inner_r = max(0.0, right_hws[i] - rim)
                        if inner_l > 0 or inner_r > 0:
                            has_cutout = True
                        co_left.append(
                            (px + nx * inner_l, py + ny * inner_l))
                        co_right.append(
                            (px - nx * inner_r, py - ny * inner_r))

                    if has_cutout:
                        co_poly = co_left + co_right[::-1]
                        if len(co_poly) >= 3:
                            cutout_draw.polygon(
                                [(int(round(x)), int(round(y)))
                                 for x, y in co_poly],
                                fill=255,
                            )

        # Darken stone along cracks
        crack_mask = np.array(
            crack_img.filter(ImageFilter.GaussianBlur(radius=0.7)),
            dtype=np.float64
        ) / 255.0
        stone *= (1.0 - crack_mask * 0.4)[:, :, np.newaxis]

        # Punch through the interior of wide cracks (soft edge)
        cutout_arr = np.array(
            cutout_img.filter(ImageFilter.GaussianBlur(radius=1.5)),
            dtype=np.float64
        ) / 255.0
        mask_mod = mask_arr - cutout_arr
        np.clip(mask_mod, 0, 1, out=mask_mod)
        tablet_mask = Image.fromarray(
            (mask_mod * 255).astype(np.uint8), mode='L')

    # --- Pitting (small surface imperfections) ---
    if config.pit_density > 0:
        mask_arr = np.array(tablet_mask, dtype=np.float64) / 255.0
        n_pits = int(config.pit_density * h * w / 500)
        px = rng.randint(0, w, size=n_pits)
        py = rng.randint(0, h, size=n_pits)
        pr = np.maximum(1, rng.exponential(1.5, size=n_pits).astype(int))
        pd = rng.uniform(0.85, 0.95, size=n_pits)

        pit_factor = np.ones((h, w), dtype=np.float64)
        for i in range(n_pits):
            if mask_arr[py[i], px[i]] < 0.5:
                continue
            y1 = max(0, py[i] - pr[i])
            y2 = min(h, py[i] + pr[i] + 1)
            x1 = max(0, px[i] - pr[i])
            x2 = min(w, px[i] + pr[i] + 1)
            pit_factor[y1:y2, x1:x2] = np.minimum(
                pit_factor[y1:y2, x1:x2], pd[i]
            )

        stone *= pit_factor[:, :, np.newaxis]

    return np.clip(stone, 0, 1), tablet_mask


def _apply_edge_bevel(stone, tablet_mask, config, height):
    """Emboss the tablet edges to give the slab a 3D raised appearance."""
    if config.edge_bevel <= 0:
        return stone

    h, w = stone.shape[:2]
    mask_arr = np.array(tablet_mask, dtype=np.float64) / 255.0

    # Blur the tablet mask to create an edge depth ramp
    bevel_r = max(2, int(height * 0.025 * config.edge_bevel))
    bevel_depth = np.array(
        tablet_mask.filter(ImageFilter.GaussianBlur(radius=bevel_r)),
        dtype=np.float64
    ) / 255.0

    # Compute gradients of the depth ramp
    dy = np.zeros_like(bevel_depth)
    dx = np.zeros_like(bevel_depth)
    dy[1:, :] = bevel_depth[1:, :] - bevel_depth[:-1, :]
    dx[:, 1:] = bevel_depth[:, 1:] - bevel_depth[:, :-1]

    # Same light direction as carving
    rad = np.radians(config.light_angle)
    light_x = np.cos(rad)
    light_y = -np.sin(rad)

    emboss = dx * light_x + dy * light_y
    max_val = np.abs(emboss).max()
    if max_val > 0:
        emboss /= max_val

    # Only apply in the edge zone (where mask is between 0 and ~1)
    edge_zone = (mask_arr > 0.01) & (bevel_depth < 0.95)
    emboss *= edge_zone.astype(np.float64)
    emboss *= config.edge_bevel * config.light_intensity

    stone += emboss[:, :, np.newaxis] * 0.45

    return np.clip(stone, 0, 1)


def _apply_edge_darkening(stone, tablet_mask, height):
    """Vignette: darken near the tablet edges."""
    h, w = stone.shape[:2]
    blur_r = max(3, int(height * 0.08))
    dist = np.array(
        tablet_mask.filter(ImageFilter.GaussianBlur(radius=blur_r)),
        dtype=np.float64
    ) / 255.0

    edge_factor = 0.85 + 0.15 * dist
    stone *= edge_factor[:, :, np.newaxis]
    return np.clip(stone, 0, 1)


def _composite(stone, tablet_mask):
    """Assemble the final RGBA image."""
    h, w = stone.shape[:2]
    mask_arr = np.array(tablet_mask, dtype=np.uint8)

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = np.clip(stone[:, :, 0] * 255, 0, 255).astype(np.uint8)
    rgba[:, :, 1] = np.clip(stone[:, :, 1] * 255, 0, 255).astype(np.uint8)
    rgba[:, :, 2] = np.clip(stone[:, :, 2] * 255, 0, 255).astype(np.uint8)
    rgba[:, :, 3] = mask_arr

    return Image.fromarray(rgba, 'RGBA')
