import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_deblurgan_model():
    from models import generator
    model = generator.Generator(in_channels=3, out_channels=3)
    model.load_state_dict(torch.hub.load_state_dict_from_url(
        "https://github.com/VITA-Group/DeblurGANv2/releases/download/v0.1/deblurganv2.pth"
    ))
    model.to(device).eval()
    return model

deblurgan_model = load_deblurgan_model()

def deblurgan_enhance(img):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    tensor = transforms.ToTensor()(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        out = deblurgan_model(tensor)[0]

    out = out.permute(1,2,0).cpu().numpy() * 255
    out = out.astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
