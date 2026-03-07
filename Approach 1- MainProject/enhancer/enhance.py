import cv2
from .classical_enhance import classical_enhance

def enhance_image(image_path, metrics):
    img = cv2.imread(image_path)

    # Heavy blur → classical sharpen
    if metrics["lap_var"] < 60:
        img = classical_enhance(img)

    # Low features → boost contrast
    if metrics["orb_kpts"] < 200:
        img = classical_enhance(img)

    return img
