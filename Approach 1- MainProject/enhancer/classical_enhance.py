import cv2
import numpy as np

def _clahe_luminance(img, clip_limit=2.0, tile_grid=(8, 8)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _unsharp_mask(img, sigma=1.2, amount=1.0):
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)


def _gamma_correction(img, gamma=1.0):
    gamma = max(gamma, 1e-6)
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, table)


def _mild_denoise(img, h=5):
    return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)


def classical_enhance(img, metrics=None):
    """
    Adaptive classical enhancement for drone frames.
    Uses denoise + local contrast + exposure correction + unsharp masking.
    """
    out = img.copy()

    if metrics is None:
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray) / 255.0)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    else:
        brightness = float(metrics.get("brightness", 0.5))
        lap_var = float(metrics.get("lap_var", 100.0))

    # Noise suppression first to avoid sharpening noise.
    out = _mild_denoise(out, h=5 if brightness > 0.2 else 7)

    # Local contrast enhancement. Stronger when image is flatter/darker.
    if brightness < 0.28:
        out = _clahe_luminance(out, clip_limit=3.0, tile_grid=(8, 8))
        out = _gamma_correction(out, gamma=1.30)
    elif brightness > 0.78:
        out = _clahe_luminance(out, clip_limit=2.0, tile_grid=(8, 8))
        out = _gamma_correction(out, gamma=0.85)
    else:
        out = _clahe_luminance(out, clip_limit=2.2, tile_grid=(8, 8))

    # Sharpening tuned by blur level.
    if lap_var < 50:
        out = _unsharp_mask(out, sigma=1.5, amount=1.3)
    elif lap_var < 100:
        out = _unsharp_mask(out, sigma=1.2, amount=1.0)
    else:
        out = _unsharp_mask(out, sigma=1.0, amount=0.6)

    return out
