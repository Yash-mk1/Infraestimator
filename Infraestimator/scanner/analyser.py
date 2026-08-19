"""
scanner/analyser.py — MODIFIED
Added foreground object removal (people, cars, furniture)
before crack/seep analysis runs.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from scipy import ndimage
from skimage import morphology, measure

from .ai_model import AIDetector

# ── COCO classes to remove before analysis ────────────────────────────────────
# These are objects that should never be analysed as structural damage
REMOVE_CLASSES = {
    0:  'person',
    1:  'bicycle',
    2:  'car',
    3:  'motorcycle',
    5:  'bus',
    7:  'truck',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    60: 'dining table',
    62: 'tv',
    63: 'laptop',
    64: 'mouse',
    67: 'cell phone',
    73: 'book',
    76: 'scissors',
    77: 'teddy bear',
}

# Cached COCO model — loaded once
_coco_model = None

def _get_coco_model():
    global _coco_model
    if _coco_model is not None:
        return _coco_model
    try:
        from ultralytics import YOLO
        # yolov8n.pt is pretrained on COCO — downloads automatically (~6MB)
        _coco_model = YOLO('yolov8n.pt')
        print("[ObjRemoval] COCO model loaded.")
        return _coco_model
    except Exception as e:
        print(f"[ObjRemoval] Could not load COCO model: {e}")
        return None


@dataclass
class DetectionResult:
    crack_score:          float = 0.0
    seep_score:           float = 0.0
    surface_score:        float = 0.0
    crack_area_pct:       float = 0.0
    seep_area_pct:        float = 0.0
    num_crack_regions:    int   = 0
    num_seep_regions:     int   = 0
    largest_crack_mm_eq:  float = 0.0
    crack_spread:         float = 0.0
    ai_used:              bool  = False
    ai_confidence:        float = 0.0
    crack_ai_used:        bool  = False
    seep_ai_used:         bool  = False
    objects_removed:      list  = field(default_factory=list)  # e.g. ['person', 'car']
    annotated_image:      Optional[np.ndarray] = field(default=None, repr=False)
    heatmap_image:        Optional[np.ndarray] = field(default=None, repr=False)


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
    AI_WEIGHT       = 0.60
    CV_WEIGHT       = 0.40

    def __init__(self):
        self._ai = AIDetector()

    def analyze(self, image_bgr: np.ndarray) -> DetectionResult:
        result   = DetectionResult()
        h, w     = image_bgr.shape[:2]
        total_px = h * w

        # ── Step 0: Remove foreground objects (people, cars etc.) ─────────────
        cleaned_image, removed = self._remove_foreground_objects(image_bgr)
        result.objects_removed = removed
        if removed:
            print(f"[ObjRemoval] Removed: {', '.join(removed)}")

        # Use cleaned image for all analysis
        gray = cv2.cvtColor(cleaned_image, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(cleaned_image, cv2.COLOR_BGR2HSV)

        # ── Step 1: Wall isolation ────────────────────────────────────────────
        wall_mask = self._get_wall_mask(cleaned_image, gray)

        # ── Step 2: Shadow rejection ──────────────────────────────────────────
        shadow_mask    = self._get_shadow_mask(hsv, gray)
        no_shadow_mask = cv2.bitwise_not(shadow_mask)
        analysis_zone  = cv2.bitwise_and(wall_mask, no_shadow_mask)

        # ── Step 3: AI detection ──────────────────────────────────────────────
        ai_result = self._ai.predict(cleaned_image)

        # ── Step 4: Crack masks ───────────────────────────────────────────────
        opencv_crack = self._opencv_cracks(gray)

        if ai_result.get('crack_ai_available') and ai_result['crack_mask'] is not None:
            ai_crack     = ai_result['crack_mask']
            ai_f         = ai_crack.astype(np.float32)    / 255.0 * self.AI_WEIGHT
            cv_f         = opencv_crack.astype(np.float32)/ 255.0 * self.CV_WEIGHT
            blended      = np.clip(ai_f + cv_f, 0, 1)
            crack_mask   = (blended >= 0.40).astype(np.uint8) * 255
            result.crack_ai_used = True
        else:
            crack_mask = opencv_crack
            result.crack_ai_used = False

        crack_mask = cv2.bitwise_and(crack_mask, analysis_zone)
        crack_mask = self._filter_false_positives(crack_mask)

        # ── Step 5: Seep masks ────────────────────────────────────────────────
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
            area_score*0.30 + spread_score*0.45 + region_score*0.25)

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
            cleaned_image.copy(), crack_mask, seep_mask,
            shadow_mask, wall_mask, result
        )
        result.heatmap_image = self._build_heatmap(gray, crack_mask, seep_mask)

        return result

    # ── Foreground object removal ─────────────────────────────────────────────

    def _remove_foreground_objects(self, image_bgr: np.ndarray):
        """
        Detect people and objects using YOLOv8n (COCO pretrained).
        Fill detected regions with inpainted wall texture.
        Returns (cleaned_image, list_of_removed_labels).
        """
        model = _get_coco_model()
        if model is None:
            return image_bgr, []

        h, w = image_bgr.shape[:2]
        removed_labels = []

        try:
            results = model.predict(
                source=image_bgr,
                conf=0.65,          # only confident detections
                verbose=False
            )[0]
        except Exception as e:
            print(f"[ObjRemoval] Inference error: {e}")
            return image_bgr, []

        if results.boxes is None or len(results.boxes) == 0:
            return image_bgr, []

        # Build a combined mask of all objects to remove
        removal_mask = np.zeros((h, w), dtype=np.uint8)

        boxes   = results.boxes.xyxy.cpu().numpy().astype(int)
        classes = results.boxes.cls.cpu().numpy().astype(int)
        confs   = results.boxes.conf.cpu().numpy()

        for box, cls_id, conf in zip(boxes, classes, confs):
            if cls_id not in REMOVE_CLASSES:
                continue

            label = REMOVE_CLASSES[cls_id]
            x1, y1, x2, y2 = box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Skip if box is too large (>50% of image) — probably mislabelled wall
            box_area = (x2 - x1) * (y2 - y1)
            if box_area > (h * w * 0.50):
                print(f"[ObjRemoval] Skipping {label} — box too large ({box_area/(h*w)*100:.0f}% of image)")
                continue
            # Skip tiny detections — real people aren't tiny blobs
            if box_area < (h * w * 0.02):   # skip if < 2% of image
                continue

            removal_mask[y1:y2, x1:x2] = 255

            if label not in removed_labels:
                removed_labels.append(label)

        if np.sum(removal_mask) == 0:
            return image_bgr, []

        # ── Inpaint: fill removed regions with surrounding wall texture ───────
        # Dilate mask slightly to cover object edges cleanly
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        removal_mask_dilated = cv2.dilate(removal_mask, k, iterations=1)

        # cv2.inpaint fills the masked region by propagating surrounding pixels
        # INPAINT_TELEA gives better results for textured surfaces
        cleaned = cv2.inpaint(
            image_bgr,
            removal_mask_dilated,
            inpaintRadius=12,
            flags=cv2.INPAINT_TELEA
        )

        return cleaned, removed_labels

    # ── Wall isolation ────────────────────────────────────────────────────────

    def _get_wall_mask(self, image_bgr, gray):
        h, w = gray.shape
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5,5), 1), 30, 100)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
        edge_regions = cv2.dilate(edges, k, iterations=2)
        low_edge = cv2.bitwise_not(edge_regions)

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        val = hsv[:,:,2]
        sat = hsv[:,:,1]

        very_dark   = (val < 40).astype(np.uint8) * 255
        very_bright = (val > 220).astype(np.uint8) * 255
        high_sat    = (sat > 120).astype(np.uint8) * 255

        exclude = cv2.bitwise_or(very_dark, very_bright)
        exclude = cv2.bitwise_or(exclude, high_sat)

        k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (25,25))
        exclude = cv2.dilate(exclude, k2, iterations=2)

        candidate = cv2.bitwise_and(low_edge, cv2.bitwise_not(exclude))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            candidate, connectivity=8
        )
        region_sizes = [(stats[i, cv2.CC_STAT_AREA], i)
                        for i in range(1, num_labels)]
        region_sizes.sort(reverse=True)

        wall_mask = np.zeros((h, w), dtype=np.uint8)
        for _, idx in region_sizes[:3]:
            wall_mask[labels == idx] = 255

        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35,35))
        wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, k3, iterations=3)

        if np.sum(wall_mask > 0) < (h * w * 0.25):
            wall_mask = cv2.bitwise_not(exclude)

        return wall_mask

    # ── Shadow rejection ──────────────────────────────────────────────────────

    def _get_shadow_mask(self, hsv, gray):
        h_ch, s_ch, v_ch = cv2.split(hsv)
        dark_low_sat = cv2.bitwise_and(
            (v_ch < 60).astype(np.uint8) * 255,
            (s_ch < 40).astype(np.uint8) * 255
        )
        laplacian  = cv2.Laplacian(gray, cv2.CV_64F)
        sharp_mask = (np.abs(laplacian) > 25).astype(np.uint8) * 255
        shadow_candidate = cv2.bitwise_and(
            dark_low_sat, cv2.bitwise_not(sharp_mask)
        )
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            shadow_candidate, connectivity=8
        )
        shadow_mask   = np.zeros_like(shadow_candidate)
        min_shadow_area = gray.shape[0] * gray.shape[1] * 0.005
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_shadow_area:
                shadow_mask[labels == i] = 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        return cv2.erode(shadow_mask, k, iterations=1)

    # ── False positive filter ─────────────────────────────────────────────────

    def _filter_false_positives(self, crack_mask):
        labeled, _ = ndimage.label(crack_mask)
        regions    = measure.regionprops(labeled)
        clean      = np.zeros_like(crack_mask)
        for r in regions:
            if r.area < self.MIN_REGION_AREA:
                continue
            perimeter = r.perimeter if r.perimeter > 0 else 1
            circularity = (4 * np.pi * r.area) / (perimeter ** 2)
            if circularity > 0.65:
                continue
            bbox_h = r.bbox[2] - r.bbox[0]
            bbox_w = r.bbox[3] - r.bbox[1]
            if bbox_h == 0 or bbox_w == 0:
                continue
            if max(bbox_h, bbox_w) / min(bbox_h, bbox_w) < 1.8:
                continue
            clean[labeled == r.label] = 255
        return clean

    # ── OpenCV crack detection ────────────────────────────────────────────────

    def _opencv_cracks(self, gray):
        blurred  = cv2.GaussianBlur(gray, (5,5), 1.2)
        clahe    = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        enhanced = clahe.apply(blurred)
        edges    = cv2.Canny(enhanced, 30, 120)
        kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        dilated  = cv2.dilate(edges, kernel, iterations=1)
        skeleton = morphology.skeletonize((dilated > 0).astype(bool))
        labeled, _ = ndimage.label(skeleton)
        sizes    = ndimage.sum(skeleton, labeled, range(labeled.max() + 1))
        mask     = sizes >= 15
        return (mask[labeled] * 255).astype(np.uint8)

    # ── OpenCV seep detection ─────────────────────────────────────────────────

    def _opencv_seeps(self, hsv, gray):
        h_ch, s_ch, v_ch = cv2.split(hsv)
        dark     = (v_ch < 85).astype(np.uint8) * 255
        wet      = (s_ch > 45).astype(np.uint8) * 255
        rust     = cv2.inRange(hsv, (5,  30,  40), (25,  255, 200))
        moisture = cv2.inRange(hsv, (85, 20,  30), (135, 255, 160))
        combined = cv2.bitwise_or(cv2.bitwise_and(dark, wet),
                                  cv2.bitwise_or(rust, moisture))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
        return cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ── Surface degradation ───────────────────────────────────────────────────

    def _surface_degradation_score(self, gray, wall_mask):
        wall_gray = cv2.bitwise_and(gray, wall_mask)
        blurred   = cv2.GaussianBlur(wall_gray.astype(np.float32), (15,15), 0)
        diff      = (wall_gray.astype(np.float32) - blurred) ** 2
        local_var = cv2.GaussianBlur(diff, (31,31), 0)
        high_var  = float(np.mean(local_var > 300))
        lap_var   = float(cv2.Laplacian(wall_gray, cv2.CV_64F).var())
        sharpness = min(1.0, lap_var / 500.0)
        return min(10.0, (high_var * 6.0) + ((1.0 - sharpness) * 4.0))

    # ── Annotate ──────────────────────────────────────────────────────────────

    def _annotate(self, img, crack_mask, seep_mask,
                  shadow_mask, wall_mask, result):
        overlay = img.copy()
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
            cv2.circle(annotated, (x,y), 6, (0,0,255), -1)
            cv2.putText(annotated, 'CRACK', (x+8,y+4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 1, cv2.LINE_AA)

        for r in measure.regionprops(ndimage.label(seep_mask)[0]):
            if r.area < self.MIN_REGION_AREA: continue
            y, x = int(r.centroid[0]), int(r.centroid[1])
            cv2.circle(annotated, (x,y), 6, (200,80,0), -1)
            cv2.putText(annotated, 'SEEP', (x+8,y+4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,80,0), 1, cv2.LINE_AA)

        h, w = annotated.shape[:2]
        cv2.rectangle(annotated, (0,0), (w,28), (20,20,20), -1)
        crack_src = f'AI+CV' if result.crack_ai_used else 'CV'
        seep_src  = f'AI+CV' if result.seep_ai_used  else 'CV'
        obj_note  = f'  Removed:{",".join(result.objects_removed)}' \
                    if result.objects_removed else ''
        cv2.putText(
            annotated,
            f'INFRA HEALTH MONITOR  Crack:{crack_src}  Seep:{seep_src}{obj_note}',
            (8,18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1, cv2.LINE_AA
        )
        return annotated

    # ── Heatmap ───────────────────────────────────────────────────────────────

    def _build_heatmap(self, gray, crack_mask, seep_mask):
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

        raw_health = max(0.0, 10.0 - damage)
        health     = round(max(1.0, raw_health), 1)
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
        norm = health / 10.0
        if health >= 4.0:
            life = base * (norm ** 1.4)
        else:
            life = base * (0.02 + (norm * 0.025))
        report.estimated_life_years = round(max(1.0, life), 1)

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
        if result.objects_removed:
            report.warnings.append(
                f'Objects detected and excluded before analysis: '
                f'{", ".join(result.objects_removed)}'
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