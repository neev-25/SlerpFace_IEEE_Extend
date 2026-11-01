"""
modules/antispoof/face_alignment.py
=====================================
Face Detection & ROI Alignment (Pipeline Section 2).
Uses OpenCV Haar cascade if available, otherwise falls back to center crop.
NOTE: cv2 is fully optional; the module works without it.
"""
import numpy as np
from PIL import Image


def _load_cascade():
    """
    Try to load OpenCV Haar cascade. Returns cascade object or None.
    Catches all exceptions including partial/stub opencv installs that
    raise custom errors (e.g. FaceDetectionUnavailable).
    """
    try:
        import cv2
        # Verify cv2 is fully functional by calling a basic method
        _ = cv2.getBuildInformation()
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        return None if cascade.empty() else cascade
    except Exception:
        return None


_CASCADE = _load_cascade()
_CV2_AVAILABLE = _CASCADE is not None


def detect_face_roi(img_rgb: np.ndarray, target_size: int = 112):
    """
    Detect largest face and return tight crop.

    Returns:
        (cropped_face np.ndarray, bbox tuple|None)
    """
    h, w = img_rgb.shape[:2]
    bbox = None

    # Only use OpenCV if the cascade loaded successfully
    if _CV2_AVAILABLE and _CASCADE is not None:
        try:
            import cv2
            gray  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            faces = _CASCADE.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            if len(faces) > 0:
                areas        = [fw * fh for (_, _, fw, fh) in faces]
                fx, fy, fw, fh = faces[int(np.argmax(areas))]
                pad_x = int(fw * 0.20); pad_y = int(fh * 0.20)
                x1 = max(0, fx - pad_x); y1 = max(0, fy - pad_y)
                x2 = min(w, fx + fw + pad_x); y2 = min(h, fy + fh + pad_y)
                face_crop = img_rgb[y1:y2, x1:x2]
                bbox = (x1, y1, x2 - x1, y2 - y1)
                pil_crop = Image.fromarray(face_crop).resize((target_size, target_size), Image.LANCZOS)
                return np.array(pil_crop, dtype=np.uint8), bbox
        except Exception:
            pass  # Fallthrough to center crop

    # Fallback: 70% center crop (no cv2 needed)
    margin_y = int(h * 0.15)
    margin_x = int(w * 0.15)
    center_crop = img_rgb[margin_y: h - margin_y, margin_x: w - margin_x]
    if center_crop.size == 0:
        center_crop = img_rgb
    pil_crop = Image.fromarray(center_crop).resize((target_size, target_size), Image.LANCZOS)
    return np.array(pil_crop, dtype=np.uint8), None


def preprocess_for_pad(img_rgb: np.ndarray, target_size: int = 112) -> dict:
    """Full pre-processing step: align, crop, report."""
    original_h, original_w = img_rgb.shape[:2]
    face_crop, bbox = detect_face_roi(img_rgb, target_size=target_size)
    return {
        "face_crop":     face_crop,
        "face_detected": bbox is not None,
        "bbox":          bbox,
        "original_size": (original_h, original_w),
    }


def draw_face_bbox(img_rgb: np.ndarray, bbox, color=(0, 220, 80), thickness: int = 3) -> np.ndarray:
    """Draw bounding box rectangle on image (no cv2 required)."""
    if bbox is None:
        return img_rgb.copy()
    annotated = img_rgb.copy()
    x, y, bw, bh = bbox
    x2, y2 = x + bw, y + bh
    h2, w2 = annotated.shape[:2]
    t = thickness
    annotated[max(0, y): min(h2, y + t),    max(0, x): min(w2, x2)]   = color
    annotated[max(0, y2 - t): min(h2, y2),  max(0, x): min(w2, x2)]   = color
    annotated[max(0, y): min(h2, y2),        max(0, x): min(w2, x + t)] = color
    annotated[max(0, y): min(h2, y2),    max(0, x2 - t): min(w2, x2)]  = color
    return annotated
