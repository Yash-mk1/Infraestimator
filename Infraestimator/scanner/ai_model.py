"""
scanner/ai_model.py
"""
 
from pathlib import Path
import numpy as np
import cv2
 
MODELS_DIR   = Path(__file__).resolve().parent / 'models'
CRACK_MODEL  = MODELS_DIR / 'crack_model.pt'
SEEP_MODEL   = MODELS_DIR / 'seep_model.pt'
 
CONF_THRESHOLD = 0.35
 
_crack_model = None
_seep_model  = None
 
def _load(path: Path, label: str):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[AI] ultralytics not installed. Run: pip install ultralytics")
        return None
 
    if not path.exists():
        print(f"[AI] {label} model not found at {path} — using OpenCV fallback for {label}.")
        return None
 
    try:
        print(f"[AI] Loading {label} model from {path} ...")
        m = YOLO(str(path))
        print(f"[AI] {label} model loaded.")
        return m
    except Exception as e:
        print(f"[AI] Failed to load {label} model: {e}")
        return None
 
 
class AIDetector:
    def __init__(self):
        global _crack_model, _seep_model
 
        if _crack_model is None:
            _crack_model = _load(CRACK_MODEL, 'crack')
        if _seep_model is None:
            _seep_model = _load(SEEP_MODEL, 'seep')
 
        self._crack_model = _crack_model
        self._seep_model  = _seep_model
 
        self.crack_ai_available = self._crack_model is not None
        self.seep_ai_available  = self._seep_model  is not None
        self.ai_available       = self.crack_ai_available or self.seep_ai_available
 
        if not self.ai_available:
            print("[AI] No models loaded — full OpenCV fallback active.")
 
    def predict(self, image_bgr: np.ndarray) -> dict:
        h, w = image_bgr.shape[:2]
 
        crack_mask = None
        seep_mask  = None
        crack_conf = 0.0
        seep_conf  = 0.0
 
        if self.crack_ai_available:
            crack_mask, crack_conf = _run_model(
                self._crack_model, image_bgr, h, w, target='crack'
            )
 
        if self.seep_ai_available:
            seep_mask, seep_conf = _run_model(
                self._seep_model, image_bgr, h, w, target='seep'
            )
 
        if seep_mask is None:
            seep_mask = _derive_seep_mask(
                image_bgr,
                crack_mask if crack_mask is not None
                else np.zeros((h, w), dtype=np.uint8)
            )
 
        avg_conf = (crack_conf + seep_conf) / max(
            int(self.crack_ai_available) + int(self.seep_ai_available), 1
        )
 
        return {
            'crack_mask':         crack_mask,
            'seep_mask':          seep_mask,
            'confidence':         round(avg_conf, 3),
            'ai_available':       self.ai_available,
            'crack_ai_available': self.crack_ai_available,
            'seep_ai_available':  self.seep_ai_available,
        }
 
 
def _run_model(model, image_bgr, h, w, target):
    CRACK_KEYWORDS = {'crack', 'fracture', 'fissure', 'alligator',
                      'longitudinal', 'transverse', 'diagonal', 'spall'}
    SEEP_KEYWORDS  = {'seep', 'moisture', 'wet', 'stain', 'rust',
                      'leak', 'water', 'efflorescence', 'damp', 'seepage'}
 
    keywords = CRACK_KEYWORDS if target == 'crack' else SEEP_KEYWORDS
 
    try:
        results = model.predict(
            source=image_bgr, conf=CONF_THRESHOLD, verbose=False
        )[0]
    except Exception as e:
        print(f"[AI] {target} model inference error: {e}")
        return None, 0.0
 
    mask        = np.zeros((h, w), dtype=np.uint8)
    confidences = []
    names       = model.names
 
    if results.masks is not None:
        masks_data = results.masks.data.cpu().numpy()
        classes    = results.boxes.cls.cpu().numpy().astype(int)
        confs      = results.boxes.conf.cpu().numpy()
 
        for seg_mask, cls_id, conf in zip(masks_data, classes, confs):
            label = names.get(cls_id, '').lower().replace('-', ' ').replace('_', ' ')
            if _matches(label, keywords) or len(names) == 1:
                m = cv2.resize(
                    (seg_mask * 255).astype(np.uint8), (w, h),
                    interpolation=cv2.INTER_NEAREST
                )
                mask = cv2.bitwise_or(mask, m)
                confidences.append(float(conf))
 
    elif results.boxes is not None and len(results.boxes) > 0:
        boxes   = results.boxes.xyxy.cpu().numpy().astype(int)
        classes = results.boxes.cls.cpu().numpy().astype(int)
        confs   = results.boxes.conf.cpu().numpy()
 
        for box, cls_id, conf in zip(boxes, classes, confs):
            label = names.get(cls_id, '').lower().replace('-', ' ').replace('_', ' ')
            if _matches(label, keywords) or len(names) == 1:
                x1, y1, x2, y2 = box
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                mask[y1:y2, x1:x2] = 255
                confidences.append(float(conf))
 
        if target == 'crack' and np.sum(mask) > 0:
            mask = _refine_with_edges(image_bgr, mask)
 
    if np.sum(mask) == 0:
        return None, 0.0
 
    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    mask     = _clean_mask(mask, min_area=60)
    return mask, avg_conf
 
 
def _matches(label, keywords):
    return any(kw in label for kw in keywords)
 
 
def _refine_with_edges(image_bgr, box_mask):
    gray     = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(cv2.GaussianBlur(gray, (5, 5), 1.2))
    edges    = cv2.Canny(enhanced, 30, 120)
    refined  = cv2.bitwise_and(edges, box_mask)
    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.dilate(refined, kernel, iterations=1)
 
 
def _derive_seep_mask(image_bgr, crack_mask):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
 
    dark_mask     = (v_ch < 85).astype(np.uint8) * 255
    wet_mask      = (s_ch > 45).astype(np.uint8) * 255
    rust_mask     = cv2.inRange(hsv, (5,  30,  40), (25,  255, 200))
    moisture_mask = cv2.inRange(hsv, (85, 20,  30), (135, 255, 160))
 
    combined = cv2.bitwise_or(
        cv2.bitwise_and(dark_mask, wet_mask),
        cv2.bitwise_or(rust_mask, moisture_mask)
    )
 
    if np.sum(crack_mask) > 0:
        dilated  = cv2.dilate(
            crack_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
            iterations=1
        )
        combined = cv2.bitwise_and(combined, cv2.bitwise_not(dilated))
 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    return cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
 
 
def _clean_mask(mask, min_area=60):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    clean = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255
    return clean
 