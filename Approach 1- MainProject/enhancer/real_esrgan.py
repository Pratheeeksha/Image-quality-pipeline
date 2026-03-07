import cv2
from realesrgan import RealESRGANer

def load_esrgan():
    model = RealESRGANer(
        model_path="RealESRGAN_x2plus.pth",
        scale=2,
        dni_weight=None
    )
    return model

esrgan_model = load_esrgan()

def esrgan_enhance(img):
    output, _ = esrgan_model.enhance(img)
    return output
