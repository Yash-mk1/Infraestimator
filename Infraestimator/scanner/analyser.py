"""
scanner/analyser.py — SMART PIPELINE

Improvements:
  1. 60/40 weighted mask merge (AI=60%, OpenCV=40%)
  2. Shadow rejection before crack detection
  3. Wall region isolation — only analyse the dominant flat surface
  4. False positive filtering — remove non-crack shaped blobs
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from scipy import ndimage
from skimage import morphology, measure

from .ai_model import AIDetector


@dataclass
class DetectionResult:
    crack_score:         float = 0.0
    seep_score:          float = 0.0
    surface_score:       float = 0.0
    crack_area_pct:      float = 0.0
    seep_area_pct:       float = 0.0
    num_crack_regions:   int   = 0
    num_seep_regions:    int   = 0
    largest_crack_mm_eq: float = 0.0
    crack_spread:        float = 0.0
    ai_used:             bool  = False
    ai_confidence:       float = 0.0
    crack_ai_used:       bool  = False
    seep_ai_used:        bool  = False
    annotated_image:     Optional[np.ndarray] = field(default=None, repr=False)
    heatmap_image:       Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class HealthReport:
    health_score:         float = 10.0
    condition:            str   = 'Unknown'
    estimated_life_years: float = 0.0
    critical:             bool  = False
    warnings:             list  = field(default_factory=list)
    recommendations:      list  = field(default_factory=list)
    detection:            Optional[DetectionResult] = None
    timestamp:            str   = field(
        default_factory=lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )
    material: str = 'general'


class InfrastructureAnalyzer:
    MIN_REGION_AREA = 60
    AI_WEIGHT       = 0.60   # AI contributes 60%
    CV_WEIGHT       = 0.40   # OpenCV contributes 40%

    def __init__(self):
        self._ai = AIDetector()

    def analyze(self, image_bgr: np.ndarray) -> DetectionResult:
        result   = DetectionResult()
        h, w     = image_bgr.shape[:2]
        total_px = h * w
        gray     = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        hsv      = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        # ── Step 1: Isolate wall region ───────────────────────────────────────
        wall_mask = self._get_wall_mask(image_bgr, gray)

        # ── Step 2: Reject shadows ────────────────────────────────────────────
        shadow_mask    = self._get_shadow_mask(hsv, gray)
        no_shadow_mask = cv2.bitwise_not(shadow_mask)

        # Analysis zone = wall area minus shadows
        analysis_zone = cv2.bitwise_and(wall_mask, no_shadow_mask)

        # ── Step 3: Run AI ────────────────────────────────────────────────────
        ai_result = self._ai.predict(image_bgr)

        # ── Step 4: Get OpenCV crack mask ─────────────────────────────────────
        opencv_crack = self._opencv_cracks(gray)

        # ── Step 5: Weighted 60/40 merge for cracks ───────────────────────────
        if ai_result.get('crack_ai_available') and ai_result['crack_mask'] is not None:
            ai_crack = ai_result['crack_mask']

            # Weighted blend: AI=60%, OpenCV=40%
            ai_float  = ai_crack.astype(np.float32)   / 255.0 * self.AI_WEIGHT
            cv_float  = opencv_crack.astype(np.float32)/ 255.0 * self.CV_WEIGHT
            blended   = np.clip(ai_float + cv_float, 0, 1)

            # Threshold: pixel needs at least 40% combined confidence to count
            crack_mask = (blended >= 0.40).astype(np.uint8) * 255
            result.crack_ai_used = True
        else:
            crack_mask = opencv_crack
            result.crack_ai_used = False

        # ── Step 6: Apply analysis zone (remove shadows + non-wall) ──────────
        crack_mask = cv2.bitwise_and(crack_mask, analysis_zone)

        # ── Step 7: Filter false positives ───────────────────────────────────
        crack_mask = self._filter_false_positives(crack_mask)

        # ── Step 8: Seep masks with same 60/40 weighting ─────────────────────
        opencv_seep = self._opencv_seeps(hsv, gray)

        if ai_result.get('seep_ai_available') and ai_result['seep_mask'] is not None:
            ai_seep   = ai_result['seep_mask']
            ai_sf     = ai_seep.astype(np.float32)    / 255.0 * self.AI_WEIGHT
            cv_sf     = opencv_seep.astype(np.float32)/ 255.0 * self.CV_WEIGHT
            blended_s = np.clip(ai_sf + cv_sf, 0, 1)
            seep_mask = (blended_s >= 0.40).astype(np.uint8) * 255
            result.seep_ai_used = True
        else:
            seep_mask = opencv_seep
            result.seep_ai_used = False

        # Remove seep detections inside crack regions (avoid overlap)
        seep_mask = cv2.bitwise_and(seep_mask, cv2.bitwise_not(
            cv2.dilate(crack_mask,
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)),
                       iterations=1)
        ))
        seep_mask = cv2.bitwise_and(seep_mask, analysis_zone)

        result.ai_used       = result.crack_ai_used or result.seep_ai_used
        result.ai_confidence = ai_result.get('confidence', 0.0)

        # ── Crack metrics ─────────────────────────────────────────────────────
        crack_px = int(np.sum(crack_mask > 0))
        result.crack_area_pct = (crack_px / total_px) * 100

        c_labeled, _ = ndimage.label(crack_mask)
        c_regions = [r for r in measure.regionprops(c_labeled)
                     if r.area >= self.MIN_REGION_AREA]
        result.num_crack_regions = len(c_regions)

        if c_regions:
            largest = max(c_regions, key=lambda r: r.area)
            result.largest_crack_mm_eq = round(np.sqrt(largest.area) * 0.05, 2)

            crack_coords = np.column_stack(np.where(crack_mask > 0))
            if len(crack_coords) > 0:
                y_min, x_min = crack_coords.min(axis=0)
                y_max, x_max = crack_coords.max(axis=0)
                bbox_area    = (y_max - y_min + 1) * (x_max - x_min + 1)
                result.crack_spread = bbox_area / total_px
        else:
            result.crack_spread = 0.0

        area_score   = min(10.0, np.log1p(result.crack_area_pct * 30) * 2.2)
        spread_score = min(10.0, result.crack_spread * 12.0)
        region_score = min(10.0, np.log1p(result.num_crack_regions) * 3.5)

        result.crack_score = min(10.0,
            area_score   * 0.30 +
            spread_score * 0.45 +
            region_score * 0.25
        )

        # ── Seep metrics ──────────────────────────────────────────────────────
        seep_px = int(np.sum(seep_mask > 0))
        result.seep_area_pct = (seep_px / total_px) * 100

        s_labeled, _ = ndimage.label(seep_mask)
        s_regions = [r for r in measure.regionprops(s_labeled)
                     if r.area >= self.MIN_REGION_AREA]
        result.num_seep_regions = len(s_regions)
        result.seep_score = min(10.0, np.log1p(result.seep_area_pct * 20) * 2.0)

        # ── Surface degradation ───────────────────────────────────────────────
        result.surface_score = self._surface_degradation_score(gray, wall_mask)

        # ── Visuals ───────────────────────────────────────────────────────────
        result.annotated_image = self._annotate(
            image_bgr.copy(), crack_mask, seep_mask,
            shadow_mask, wall_mask, result
        )
        result.heatmap_image = self._build_heatmap(gray, crack_mask, seep_mask)

        return result

    # ── Wall isolation ────────────────────────────────────────────────────────
def _get_wall_mask(self, image_bgr: np.ndarray,
                   gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape

    # ── Step 1: Edge density map ──────────────────────────
    edges = cv2.Canny(
        cv2.GaussianBlur(gray, (5, 5), 1), 30, 100
    )
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    edge_regions = cv2.dilate(edges, k, iterations=2)
    low_edge = cv2.bitwise_not(edge_regions)

    # ── Step 2: Exclude window-like regions ───────────────
    # Windows are: very dark OR very bright (reflective glass)
    # with rectangular shape and high internal uniformity
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    val = hsv[:, :, 2]

    # Very dark regions (window interiors)
    very_dark = (val < 40).astype(np.uint8) * 255
    # Very bright regions (glass reflections, sky in window)
    very_bright = (val > 220).astype(np.uint8) * 255
    # High saturation non-wall objects (railings, pipes, signs)
    sat = hsv[:, :, 1]
    high_sat_objects = (sat > 120).astype(np.uint8) * 255

    # Combine exclusion zones
    exclude = cv2.bitwise_or(very_dark, very_bright)
    exclude = cv2.bitwise_or(exclude, high_sat_objects)

    # Dilate exclusion zones to cover object borders
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    exclude = cv2.dilate(exclude, k2, iterations=2)

    # ── Step 3: Find largest flat non-excluded region ─────
    candidate = cv2.bitwise_and(low_edge,
                                cv2.bitwise_not(exclude))

    num_labels, labels, stats, _ = \
        cv2.connectedComponentsWithStats(candidate,
                                         connectivity=8)

    region_sizes = [(stats[i, cv2.CC_STAT_AREA], i)
                    for i in range(1, num_labels)]
    region_sizes.sort(reverse=True)

    wall_mask = np.zeros((h, w), dtype=np.uint8)
    for _, idx in region_sizes[:3]:
        wall_mask[labels == idx] = 255

    # ── Step 4: Morphological closing to fill crack gaps ──
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE,
                                  k3, iterations=3)

    # ── Step 5: Fallback if wall too small ────────────────
    if np.sum(wall_mask > 0) < (h * w * 0.25):
        # Use full image minus obvious objects
        wall_mask = cv2.bitwise_not(exclude)

        return wall_mask   

    # ── Shadow rejection ──────────────────────────────────────────────────────
    def _get_shadow_mask(self, hsv: np.ndarray,
                         gray: np.ndarray) -> np.ndarray:
        """
        Detects cast shadows — areas that are:
          • Dark (low V in HSV)
          • Low saturation (shadows desaturate colours)
          • Have smooth gradients (not sharp crack edges)
          • Spatially large (shadows are bigger than cracks)
        """
        h_ch, s_ch, v_ch = cv2.split(hsv)

        # Shadows: very dark + low saturation
        dark_low_sat = cv2.bitwise_and(
            (v_ch < 60).astype(np.uint8) * 255,
            (s_ch < 40).astype(np.uint8) * 255
        )

        # Smooth gradient check — shadows have gradual edges
        # Cracks have sharp edges, so remove sharp-edge dark regions
        laplacian  = cv2.Laplacian(gray, cv2.CV_64F)
        sharp_mask = (np.abs(laplacian) > 25).astype(np.uint8) * 255

        # Shadow = dark+low_sat but NOT sharp edges
        shadow_candidate = cv2.bitwise_and(
            dark_low_sat, cv2.bitwise_not(sharp_mask)
        )

        # Shadows are large blobs — remove tiny ones (those are dirt/stains)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            shadow_candidate, connectivity=8
        )
        shadow_mask = np.zeros_like(shadow_candidate)
        min_shadow_area = gray.shape[0] * gray.shape[1] * 0.005  # 0.5% of image

        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_shadow_area:
                shadow_mask[labels == i] = 255

        # Slightly erode — be conservative, don't remove too much
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        shadow_mask = cv2.erode(shadow_mask, k, iterations=1)

        return shadow_mask

    # ── False positive filter ─────────────────────────────────────────────────
    def _filter_false_positives(self, crack_mask: np.ndarray) -> np.ndarray:
        """
        Removes blobs that don't look like cracks:
          • Too round (circularity > 0.6) — cracks are elongated
          • Too large and blobby — likely a shadow or object edge
          • Wrong aspect ratio — cracks are thin and long
        """
        labeled, _ = ndimage.label(crack_mask)
        regions    = measure.regionprops(labeled)
        clean      = np.zeros_like(crack_mask)

        for r in regions:
            if r.area < self.MIN_REGION_AREA:
                continue

            # Circularity: 4π·area/perimeter² — cracks are low (~0.0–0.3)
            perimeter = r.perimeter if r.perimeter > 0 else 1
            circularity = (4 * np.pi * r.area) / (perimeter ** 2)
            if circularity > 0.65:
                continue   # too round → not a crack

            # Aspect ratio from bounding box
            bbox_h = r.bbox[2] - r.bbox[0]
            bbox_w = r.bbox[3] - r.bbox[1]
            if bbox_h == 0 or bbox_w == 0:
                continue
            aspect = max(bbox_h, bbox_w) / min(bbox_h, bbox_w)
            if aspect < 1.8:
                continue   # too square → not a crack

            clean[labeled == r.label] = 255

        return clean

    # ── OpenCV crack detection ────────────────────────────────────────────────
    def _opencv_cracks(self, gray: np.ndarray) -> np.ndarray:
        blurred  = cv2.GaussianBlur(gray, (5, 5), 1.2)
        clahe    = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        edges    = cv2.Canny(enhanced, 30, 120)
        kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated  = cv2.dilate(edges, kernel, iterations=1)
        skeleton = morphology.skeletonize((dilated > 0).astype(bool))
        labeled, _ = ndimage.label(skeleton)
        sizes    = ndimage.sum(skeleton, labeled, range(labeled.max() + 1))
        mask     = sizes >= 15
        return (mask[labeled] * 255).astype(np.uint8)

    # ── OpenCV seep detection ─────────────────────────────────────────────────
    def _opencv_seeps(self, hsv: np.ndarray,
                      gray: np.ndarray) -> np.ndarray:
        h_ch, s_ch, v_ch = cv2.split(hsv)
        dark     = (v_ch < 85).astype(np.uint8) * 255
        wet      = (s_ch > 45).astype(np.uint8) * 255
        rust     = cv2.inRange(hsv, (5,  30,  40), (25,  255, 200))
        moisture = cv2.inRange(hsv, (85, 20,  30), (135, 255, 160))
        combined = cv2.bitwise_or(cv2.bitwise_and(dark, wet),
                                  cv2.bitwise_or(rust, moisture))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        return cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ── Surface degradation ───────────────────────────────────────────────────
    def _surface_degradation_score(self, gray: np.ndarray,
                                   wall_mask: np.ndarray) -> float:
        # Only measure degradation on the wall region
        wall_gray = cv2.bitwise_and(gray, wall_mask)
        blurred   = cv2.GaussianBlur(wall_gray.astype(np.float32), (15, 15), 0)
        diff      = (wall_gray.astype(np.float32) - blurred) ** 2
        local_var = cv2.GaussianBlur(diff, (31, 31), 0)
        high_var  = float(np.mean(local_var > 300))
        lap_var   = float(cv2.Laplacian(wall_gray, cv2.CV_64F).var())
        sharpness = min(1.0, lap_var / 500.0)
        return min(10.0, (high_var * 6.0) + ((1.0 - sharpness) * 4.0))

    # ── Annotate ──────────────────────────────────────────────────────────────
    def _annotate(self, img, crack_mask, seep_mask,
                  shadow_mask, wall_mask, result) -> np.ndarray:
        overlay = img.copy()

        # Show excluded shadow regions in dark blue tint
        overlay[shadow_mask > 0] = (
            overlay[shadow_mask > 0] * 0.5 +
            np.array([40, 20, 0], dtype=np.float32)
        ).clip(0, 255).astype(np.uint8)

        overlay[crack_mask > 0] = [0, 0, 220]
        overlay[seep_mask  > 0] = [200, 100, 0]

        annotated = cv2.addWeighted(img, 0.55, overlay, 0.45, 0)

        for r in measure.regionprops(ndimage.label(crack_mask)[0]):
            if r.area < self.MIN_REGION_AREA: continue
            y, x = int(r.centroid[0]), int(r.centroid[1])
            cv2.circle(annotated, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(annotated, 'CRACK', (x+8, y+4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 1, cv2.LINE_AA)

        for r in measure.regionprops(ndimage.label(seep_mask)[0]):
            if r.area < self.MIN_REGION_AREA: continue
            y, x = int(r.centroid[0]), int(r.centroid[1])
            cv2.circle(annotated, (x, y), 6, (200, 80, 0), -1)
            cv2.putText(annotated, 'SEEP', (x+8, y+4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,80,0), 1, cv2.LINE_AA)

        h, w = annotated.shape[:2]
        cv2.rectangle(annotated, (0, 0), (w, 28), (20, 20, 20), -1)
        crack_src = f'AI({int(self.AI_WEIGHT*100)}%)+CV({int(self.CV_WEIGHT*100)}%)' \
                    if result.crack_ai_used else 'CV'
        seep_src  = f'AI({int(self.AI_WEIGHT*100)}%)+CV({int(self.CV_WEIGHT*100)}%)' \
                    if result.seep_ai_used  else 'CV'
        cv2.putText(
            annotated,
            f'INFRA HEALTH MONITOR  Crack:{crack_src}  Seep:{seep_src}',
            (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1, cv2.LINE_AA
        )
        return annotated

    # ── Heatmap ───────────────────────────────────────────────────────────────
    def _build_heatmap(self, gray, crack_mask, seep_mask) -> np.ndarray:
        cf = crack_mask.astype(np.float32) / 255.0
        sf = seep_mask.astype(np.float32)  / 255.0
        sc = cv2.GaussianBlur(cf, (0,0), sigmaX=25)
        ss = cv2.GaussianBlur(sf, (0,0), sigmaX=35)
        combined = np.clip(sc*0.6 + ss*0.4, 0, 1)
        heatmap  = cv2.applyColorMap(
            (combined*255).astype(np.uint8), cv2.COLORMAP_JET
        )
        return cv2.addWeighted(
            cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 0.4, heatmap, 0.6, 0
        )


# ── Health Scorer ─────────────────────────────────────────────────────────────
class HealthScorer:
    MATERIAL_LIFETIMES = {
        'concrete':80, 'brick':100, 'steel':60,
        'wood':40, 'asphalt':25, 'general':50,
    }

    def score(self, result: DetectionResult, material='general') -> HealthReport:
        report = HealthReport(detection=result, material=material)

        damage = (
            result.crack_score   * 0.50 +
            result.seep_score    * 0.25 +
            result.surface_score * 0.25
        )

        if result.num_crack_regions > 3:     damage = min(10, damage + 0.6)
        if result.num_crack_regions > 7:     damage = min(10, damage + 0.8)
        if result.largest_crack_mm_eq > 2.0: damage = min(10, damage + 0.7)
        if result.crack_spread > 0.4:        damage = min(10, damage + 0.8)
        if result.crack_spread > 0.7:        damage = min(10, damage + 0.8)
        if result.seep_area_pct > 15:        damage = min(10, damage + 0.6)

        health = round(max(0.0, 10.0 - damage), 1)
        report.health_score = health

        if   health >= 8.5: report.condition = 'Excellent'
        elif health >= 7.0: report.condition = 'Good'
        elif health >= 5.5: report.condition = 'Fair'
        elif health >= 4.0: report.condition = 'Poor'
        elif health >= 2.5: report.condition = 'Critical'
        else:
            report.condition = 'Failure Imminent'
            report.critical  = True

        base = self.MATERIAL_LIFETIMES.get(material, 50)
        report.estimated_life_years = round(base * (health/10.0)**1.5, 1)

        if result.crack_score > 5:
            report.warnings.append(
                f'High crack severity — {result.num_crack_regions} region(s), '
                f'spread across {result.crack_spread*100:.0f}% of surface'
            )
        if result.seep_score > 4:
            report.warnings.append(
                f'Significant moisture/seepage — {result.seep_area_pct:.1f}% coverage'
            )
        if result.surface_score > 6:
            report.warnings.append('Severe surface degradation / spalling observed')
        if result.largest_crack_mm_eq > 1.5:
            report.warnings.append(
                f'Large crack ~{result.largest_crack_mm_eq} mm equivalent width'
            )
        if report.critical:
            report.warnings.insert(0, 'CRITICAL — Immediate structural inspection required!')

        if   health >= 8.5:
            report.recommendations = ['Routine inspection schedule',
                                       'Preventive coating if exposed']
        elif health >= 7.0:
            report.recommendations = ['Monitor crack progression quarterly',
                                       'Apply sealant to minor cracks']
        elif health >= 5.5:
            report.recommendations = ['Professional assessment within 3 months',
                                       'Seal cracks to prevent moisture ingress',
                                       'Investigate seep sources']
        elif health >= 4.0:
            report.recommendations = ['Immediate structural inspection required',
                                       'Restrict load/occupancy if applicable',
                                       'Epoxy injection for cracks',
                                       'Full waterproofing membrane for seepage']
        else:
            report.recommendations = ['STOP USE / EVACUATE if applicable',
                                       'Emergency structural engineering consultation',
                                       'Prepare for major repair or replacement',
                                       'Document all damage for insurance/regulatory bodies']
        return report