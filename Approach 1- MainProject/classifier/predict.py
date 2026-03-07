import torch
import json
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "efficientnet_b0_quality.pt")
CLASS_PATH = os.path.join(BASE_DIR, "class_name.json")

with open(CLASS_PATH, "r") as f:
    CLASS_NAMES = json.load(f)

# Load model
weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
model = models.efficientnet_b0(weights=weights)
model.classifier[1] = torch.nn.Linear(
    model.classifier[1].in_features, len(CLASS_NAMES)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE).eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    idx = probs.argmax()
    return CLASS_NAMES[idx], float(probs[idx]), probs
