"""
scanner/image_validator.py — REVISED

Smarter validation that:
  1. Accepts coloured/painted real walls (blue, yellow, red)
  2. Rejects anime, illustrations, cartoons, digital art
  3. Rejects obvious non-structural (pure nature, sky, people)
  4. Uses photo-realism detection as primary filter
"""

import cv2
import numpy as np

ACCEPT_THRESHOLD = 0.36
HARD_REJECT      = 0.20


def validate_image(image_bgr: np.ndarray) -> tuple:
    if image_bgr is None or image_bgr.size == 0:
        return False, "Could not read image. Please upload a valid file.", 0.0

    h, w = image_bgr.shape[:2]
    if h < 50 or w < 50:
        return False, "Image is too small. Please upload a larger photo.", 0.0

    scores = {}

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    # ── Check 1: Photo realism (most important) ───────────────────────────────
    # Real photos have natural noise and grain — illustrations are too smooth
    # Measure local noise in flat regions (low edge areas)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5,5), 1), 30, 100)
    flat_mask = (edges == 0).astype(np.uint8)

    if np.sum(flat_mask) > 1000:
        flat_gray = gray[flat_mask > 0].astype(np.float32)
        # Real photos: noise std > 3 in flat regions
        # Illustrations: noise std < 2 (too clean/smooth)
        noise_std = float(np.std(flat_gray))
        if noise_std > 8:
            realism_score = 1.0    # very noisy = real photo
        elif noise_std > 4:
            realism_score = 0.85
        elif noise_std > 2:
            realism_score = 0.60   # borderline
        elif noise_std > 1:
            realism_score = 0.25   # too smooth — likely illustration
        else:
            realism_score = 0.05   # perfectly smooth = digital art
    else:
        realism_score = 0.5

    scores['realism'] = realism_score

    # ── Check 2: Colour smoothness (illustration detection) ───────────────────
    # Anime/illustrations have very smooth colour transitions and
    # sharp hard edges between flat colour regions (cel shading)
    # Real surfaces have gradual, noisy colour variation

    # Measure colour variance in local 8x8 blocks
    block = 8
    h_blocks = h // block
    w_blocks = w // block
    block_vars = []

    for i in range(h_blocks):
        for j in range(w_blocks):
            patch_s = s_ch[i*block:(i+1)*block, j*block:(j+1)*block]
            patch_v = v_ch[i*block:(i+1)*block, j*block:(j+1)*block]
            # Combined variance of saturation and value
            var = float(np.var(patch_s)) + float(np.var(patch_v))
            block_vars.append(var)

    if block_vars:
        mean_var = float(np.mean(block_vars))
        # Illustrations: many blocks with near-zero variance (flat colour fill)
        # Real surfaces: consistent moderate variance everywhere
        zero_var_pct = float(np.mean([v < 5 for v in block_vars]))

        if zero_var_pct > 0.6:
            smoothness_score = 0.05   # >60% perfectly flat blocks = illustration
        elif zero_var_pct > 0.4:
            smoothness_score = 0.2
        elif zero_var_pct > 0.25:
            smoothness_score = 0.5
        else:
            smoothness_score = 0.9    # varied blocks = real photo
    else:
        smoothness_score = 0.5

    scores['smoothness'] = smoothness_score

    # ── Check 3: Saturation — but wall-aware ──────────────────────────────────
    # Coloured walls (blue, yellow, red) have UNIFORM saturation
    # Non-structural colourful objects have VARIED saturation
    mean_sat = float(np.mean(s_ch))
    sat_std  = float(np.std(s_ch))

    # High sat + uniform (low std) = painted wall = OK
    # High sat + varied (high std) = complex colourful scene = bad
    if mean_sat < 40:
        sat_score = 0.95    # greyscale/grey wall
    elif mean_sat < 100:
        if sat_std < 30:
            sat_score = 0.85   # uniform colour = painted wall
        else:
            sat_score = 0.55   # varied colour = mixed scene
    elif mean_sat < 150:
        if sat_std < 25:
            sat_score = 0.70   # uniformly bright colour = painted wall
        else:
            sat_score = 0.35   # highly varied bright colours
    else:
        sat_score = 0.15    # extremely saturated throughout

    scores['saturation'] = sat_score

    # ── Check 4: Edge character ───────────────────────────────────────────────
    # Illustrations have sharp, clean, thin edges (outlines)
    # Real surfaces have noisy, thick, irregular edges (cracks, texture)

    edge_density = float(np.mean(edges > 0)) * 100

    # Check if edges are very thin and clean (outline-like)
    # Dilate edges slightly and compare ratio
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilated_edges = cv2.dilate(edges, k, iterations=1)
    dil_density = float(np.mean(dilated_edges > 0)) * 100

    if edge_density > 0:
        edge_ratio = dil_density / edge_density
        # Real photos: ratio ~2-4 (noisy edges, spread out)
        # Illustrations: ratio >5 (thin clean outlines that spread a lot)
        if edge_ratio > 6:
            edge_score = 0.2    # very thin clean lines = illustration
        elif edge_ratio > 4:
            edge_score = 0.55
        else:
            edge_score = 0.9    # thick noisy edges = real photo
    else:
        edge_score = 0.5

    if edge_density < 0.5:
        edge_score = min(edge_score, 0.45)   # nearly no edges = blank/blurry
    elif edge_density > 40:
        edge_score = min(edge_score, 0.4)    # chaotic scene

    scores['edges'] = edge_score

    # ── Check 5: Brightness distribution ─────────────────────────────────────
    mean_brightness = float(np.mean(v_ch))
    dark_pct  = float(np.mean(v_ch < 20))  * 100
    white_pct = float(np.mean(v_ch > 240)) * 100

    if dark_pct > 60:
        bright_score = 0.2
    elif white_pct > 60:
        bright_score = 0.2
    elif 35 <= mean_brightness <= 230:
        bright_score = 0.95
    elif 15 <= mean_brightness < 35:
        bright_score = 0.6
    else:
        bright_score = 0.5

    scores['brightness'] = bright_score

    # ── Weighted final score ──────────────────────────────────────────────────
    weights = {
        'realism':    0.35,   # most important — photo vs illustration
        'smoothness': 0.25,   # cel shading detection
        'saturation': 0.20,   # wall-aware colour check
        'edges':      0.12,   # outline detection
        'brightness': 0.08,
    }

    confidence = sum(scores[k] * weights[k] for k in weights)
    is_valid   = confidence >= ACCEPT_THRESHOLD

    if is_valid:
        reason = "Image accepted as a structural surface."
    else:
        worst = min(scores, key=scores.get)
        reasons = {
            'realism': (
                "Image appears to be a digital illustration, anime, or "
                "computer-generated art rather than a real photo. "
                "Please upload an actual photograph of a structural surface."
            ),
            'smoothness': (
                "Image has unnaturally smooth colour regions, suggesting "
                "it may be an illustration or digital artwork. "
                "Please upload a real photo of a wall, road, or structure."
            ),
            'saturation': (
                "Image appears too colourful or complex for structural analysis. "
                "Please upload a photo of a wall, floor, road, or building surface."
            ),
            'edges': (
                "Image edge pattern suggests a digital illustration or cartoon. "
                "Please upload a real photograph of a structural surface."
            ),
            'brightness': (
                "Image is too dark or overexposed. "
                "Please take a well-lit photo of the surface."
            ),
        }
        reason = reasons.get(worst, "Image does not appear to be a structural surface photograph.")

    return is_valid, reason, round(confidence, 3)


def get_validation_details(image_bgr: np.ndarray) -> dict:
    is_valid, reason, confidence = validate_image(image_bgr)
    return {'is_valid': is_valid, 'confidence': confidence, 'reason': reason}