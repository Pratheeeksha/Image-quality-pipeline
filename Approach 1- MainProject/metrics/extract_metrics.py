import cv2
import numpy as np

def extract_metrics(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    orb = cv2.ORB_create()
    kp = orb.detect(gray, None)
    orb_kpts = len(kp)

    brightness = np.mean(gray) / 255.0

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = np.mean(hsv[:,:,1]) / 255.0

    dark = np.min(img, axis=2)
    haze_score = np.mean(dark) / 255.0

    return {
        "lap_var": lap_var,
        "orb_kpts": orb_kpts,
        "brightness": brightness,
        "saturation": saturation,
        "haze_score": haze_score
    }
