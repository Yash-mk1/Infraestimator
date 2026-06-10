"""
scanner/image_validator.py — NEW FILE

Heuristic-based image relevance checker.
Rejects obviously wrong inputs (people, animals, food, colourful objects)
before running the expensive crack detection pipeline.

Checks:
  1. Colour saturation  — structural surfaces are mostly grey/beige (low sat)
  2. Colour diversity   — walls have limited colour range
  3. Texture uniformity — surfaces have consistent local texture
  4. Edge density       — walls have moderate edges, not chaotic
  5. Brightness range   — structural photos aren't pure black or blown out

Returns: (is_valid: bool, reason: str, confidence: float)
"""

import cv2
import numpy as np


# ── Thresholds ────────────────────────────────────────────────────────────────
# Each check produces a score 0-1. Final confidence = weighted average.
# If confidence < ACCEPT_THRESHOLD → reject.

ACCEPT_THRESHOLD = 0.42   # below this → reject
HARD_REJECT      = 0.25   # below this → reject with strong message


def validate_image(image_bgr: np.ndarray) -> tuple:
    """
    Returns (is_valid, reason, confidence)
      is_valid   : bool
      reason     : string shown to user if rejected
      confidence : float 0-1 (how likely it's a structural surface)
    """
    if image_bgr is None or image_bgr.size == 0:
        return False, "Could not read image. Please upload a valid file.", 0.0

    h, w = image_bgr.shape[:2]
    if h < 50 or w < 50:
        return False, "Image is too small. Please upload a larger photo.", 0.0

    scores = {}

    # ── Check 1: Saturation ───────────────────────────────────────────────────
    # Structural surfaces: concrete, brick, asphalt, stone = low saturation
    # People, animals, food, nature = higher saturation
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)

    mean_sat   = float(np.mean(sat))
    # % of pixels with HIGH saturation (colourful)
    high_sat_pct = float(np.mean(sat > 80)) * 100

    # Low mean saturation → good (structural)
    if mean_sat < 25:
        sat_score = 1.0
    elif mean_sat < 50:
        sat_score = 0.8
    elif mean_sat < 80:
        sat_score = 0.5
    elif mean_sat < 110:
        sat_score = 0.25
    else:
        sat_score = 0.0

    # Penalise if large % of image is very colourful
    if high_sat_pct > 40:
        sat_score *= 0.4
    elif high_sat_pct > 25:
        sat_score *= 0.65

    scores['saturation'] = sat_score

    # ── Check 2: Dominant colour range ───────────────────────────────────────
    # Structural surfaces cluster around grey/beige/brown in hue
    # Hue 0-30 (red/orange) and 90-150 (green/teal) = non-structural
    hue = hsv[:, :, 0].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    # Only check hue for pixels that aren't near-grey (sat > 20)
    coloured_pixels = sat > 20
    if np.sum(coloured_pixels) > 100:
        hue_coloured = hue[coloured_pixels]
        # Green/teal range (very unlikely in structural surfaces)
        green_pct = float(np.mean((hue_coloured > 60) & (hue_coloured < 150))) * 100
        # Vivid blue range
        blue_pct  = float(np.mean((hue_coloured > 100) & (hue_coloured < 130))) * 100

        if green_pct > 30:
            colour_score = 0.1   # likely nature / vegetation
        elif green_pct > 15:
            colour_score = 0.4
        elif blue_pct > 40:
            colour_score = 0.3   # likely sky / water
        else:
            colour_score = 0.85
    else:
        colour_score = 0.95   # mostly grey → very likely structural

    scores['colour'] = colour_score

    # ── Check 3: Texture uniformity ───────────────────────────────────────────
    # Structural surfaces have consistent local texture (not chaotic like fur/foliage)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Compute local standard deviation in 16x16 blocks
    block = 16
    h_blocks = h // block
    w_blocks = w // block
    local_stds = []

    for i in range(h_blocks):
        for j in range(w_blocks):
            patch = gray[i*block:(i+1)*block, j*block:(j+1)*block]
            local_stds.append(float(np.std(patch)))

    if local_stds:
        std_of_stds = float(np.std(local_stds))
        mean_std    = float(np.mean(local_stds))

        # Low variation in local texture std = uniform surface = structural
        # High variation = chaotic texture = animal fur, foliage, fabric
        texture_uniformity = std_of_stds / (mean_std + 1e-6)

        if texture_uniformity < 0.5:
            texture_score = 0.95
        elif texture_uniformity < 0.8:
            texture_score = 0.80
        elif texture_uniformity < 1.2:
            texture_score = 0.60
        elif texture_uniformity < 1.8:
            texture_score = 0.35
        else:
            texture_score = 0.15
    else:
        texture_score = 0.5

    scores['texture'] = texture_score

    # ── Check 4: Edge density ─────────────────────────────────────────────────
    # Structural surfaces have moderate edge density
    # Too few edges = blank/blurry photo
    # Too many chaotic edges = foliage, crowd, complex scene
    edges = cv2.Canny(
        cv2.GaussianBlur(gray, (5, 5), 1), 30, 100
    )
    edge_density = float(np.mean(edges > 0)) * 100   # % of edge pixels

    if 1.0 <= edge_density <= 20.0:
        edge_score = 0.90    # typical structural surface range
    elif edge_density < 1.0:
        edge_score = 0.50    # very little texture — possibly blank/blurry
    elif edge_density <= 35.0:
        edge_score = 0.65    # moderate-high edges — could still be valid
    else:
        edge_score = 0.20    # very chaotic edges — likely complex scene

    scores['edges'] = edge_score

    # ── Check 5: Brightness distribution ─────────────────────────────────────
    # Structural photos usually have mid-range brightness
    # Pure black (night photo / lens cap) or pure white (overexposed) = bad
    mean_brightness = float(np.mean(val))
    dark_pct  = float(np.mean(val < 20))  * 100
    white_pct = float(np.mean(val > 240)) * 100

    if dark_pct > 60:
        bright_score = 0.2    # nearly all black — unusable
    elif white_pct > 60:
        bright_score = 0.2    # nearly all white — overexposed
    elif 40 <= mean_brightness <= 220:
        bright_score = 0.95   # good range
    elif 20 <= mean_brightness < 40:
        bright_score = 0.6    # a bit dark but usable
    else:
        bright_score = 0.5

    scores['brightness'] = bright_score

    # ── Weighted final confidence ─────────────────────────────────────────────
    weights = {
        'saturation': 0.35,   # most discriminative for non-structural images
        'colour':     0.25,   # green/blue = nature/sky
        'texture':    0.20,   # uniformity
        'edges':      0.12,
        'brightness': 0.08,
    }

    confidence = sum(scores[k] * weights[k] for k in weights)

    # ── Decision ──────────────────────────────────────────────────────────────
    is_valid = confidence >= ACCEPT_THRESHOLD

    # Build a reason string
    if is_valid:
        reason = "Image accepted as a structural surface."
    else:
        # Give specific feedback based on which check failed hardest
        worst = min(scores, key=scores.get)
        reasons = {
            'saturation': (
                "Image appears too colourful for a structural surface. "
                "Please upload a photo of a wall, floor, road, or building surface."
            ),
            'colour': (
                "Image contains too much green or blue — likely a natural scene or sky. "
                "Please upload a photo of a structural surface."
            ),
            'texture': (
                "Image texture looks too complex or organic — possibly an animal, "
                "plant, or fabric. Please upload a structural surface photo."
            ),
            'edges': (
                "Image has too many chaotic edges — possibly a complex scene or crowd. "
                "Please upload a clear photo of a wall, road, or structure."
            ),
            'brightness': (
                "Image is too dark or overexposed. "
                "Please take a well-lit photo of the surface."
            ),
        }
        reason = reasons.get(worst, "Image does not appear to be a structural surface.")

    return is_valid, reason, round(confidence, 3)


def get_validation_details(image_bgr: np.ndarray) -> dict:
    """
    Returns detailed scores for debugging.
    Not shown to users — useful during development.
    """
    is_valid, reason, confidence = validate_image(image_bgr)
    return {
        'is_valid':   is_valid,
        'confidence': confidence,
        'reason':     reason,
    }