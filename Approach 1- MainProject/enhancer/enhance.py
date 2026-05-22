import cv2
import numpy as np
from .classical_enhance import classical_enhance

MODE_FAST = "fast_classical"
MODE_ADVANCED = "advanced_hybrid"


def _extract_metrics_from_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    orb = cv2.ORB_create()
    kp = orb.detect(gray, None)
    orb_kpts = len(kp)

    brightness = np.mean(gray) / 255.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = np.mean(hsv[:, :, 1]) / 255.0
    dark = np.min(img, axis=2)
    haze_score = np.mean(dark) / 255.0

    return {
        "lap_var": float(lap_var),
        "orb_kpts": int(orb_kpts),
        "brightness": float(brightness),
        "saturation": float(saturation),
        "haze_score": float(haze_score),
    }


def _quality_score(metrics):
    """
    Heuristic score to rank enhancement candidates.
    Higher is better.
    """
    lap_norm = min(metrics["lap_var"] / 180.0, 1.0)
    orb_norm = min(metrics["orb_kpts"] / 900.0, 1.0)
    sat_norm = min(metrics["saturation"] / 0.45, 1.0)
    # Prefer brightness near ~0.52 for balanced exposure.
    bright_penalty = abs(metrics["brightness"] - 0.52)
    bright_norm = max(0.0, 1.0 - bright_penalty / 0.52)
    # Current haze proxy increases with dark-channel mean; keep moderate values.
    haze_norm = max(0.0, 1.0 - abs(metrics["haze_score"] - 0.28) / 0.35)

    return (
        0.34 * lap_norm
        + 0.30 * orb_norm
        + 0.16 * bright_norm
        + 0.10 * sat_norm
        + 0.10 * haze_norm
    )


def _gamma_variant(img, gamma):
    inv = 1.0 / max(gamma, 1e-6)
    table = np.array([(i / 255.0) ** inv * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, table)


def _enhance_fast_classical(img, metrics):
    return classical_enhance(img, metrics)


def _enhance_advanced_hybrid(img, metrics):
    candidates = [img]

    # Baseline adaptive enhance.
    candidates.append(classical_enhance(img, metrics))

    # Extra variants based on detected issue profile.
    if metrics.get("brightness", 0.5) < 0.30:
        bright = _gamma_variant(img, gamma=1.35)
        candidates.append(classical_enhance(bright, metrics))
    elif metrics.get("brightness", 0.5) > 0.80:
        dark = _gamma_variant(img, gamma=0.82)
        candidates.append(classical_enhance(dark, metrics))

    if metrics.get("lap_var", 100.0) < 70 or metrics.get("orb_kpts", 400) < 180:
        strong = classical_enhance(img, {**metrics, "lap_var": 40.0})
        strong = cv2.detailEnhance(strong, sigma_s=12, sigma_r=0.15)
        candidates.append(strong)

    # Pick best candidate by quality score.
    best_img = img
    best_score = -1.0
    for cand in candidates:
        m = _extract_metrics_from_img(cand)
        score = _quality_score(m)
        if score > best_score:
            best_score = score
            best_img = cand

    return best_img


def enhance_image(image_path, metrics, mode=MODE_ADVANCED):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    if mode == MODE_FAST:
        return _enhance_fast_classical(img, metrics)

    return _enhance_advanced_hybrid(img, metrics)
