import torch
import cv2
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_aodnet():
    from aodnet import AODnet
    model = AODnet()
    model.load_state_dict(torch.load("aodnet.pth"))
    model.to(device).eval()
    return model

aodnet_model = load_aodnet()

def aod_dehaze(img):
    tensor = torch.from_numpy(img).float().permute(2,0,1).unsqueeze(0).to(device) / 255.
    with torch.no_grad():
        out = aodnet_model(tensor)
    out = out.squeeze().permute(1,2,0).cpu().numpy() * 255
    return out.astype(np.uint8)
