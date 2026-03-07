import os
import sys
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0
from sklearn.metrics import accuracy_score
import json

# -------------------------------
# Setup Project Root
# -------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# -------------------------------
# Paths
# -------------------------------
PIPELINE_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "pipeline_results.csv")
TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "train")
VAL_DIR = os.path.join(PROJECT_ROOT, "data", "val")
INCOMING_DIR = os.path.join(PROJECT_ROOT, "data", "incoming_drone_images")
MODEL_PATH = os.path.join(PROJECT_ROOT, "classifier", "efficientnet_b0_quality.pt")
CLASS_JSON = os.path.join(PROJECT_ROOT, "classifier", "class_name.json")

CONF_THRESHOLD = 0.90
RETRAIN_EPOCHS = 3
LR = 1e-5
BATCH_SIZE = 16

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# Step 1: Add High Confidence Data
# -------------------------------
def grow_training_data():
    df = pd.read_csv(PIPELINE_CSV)
    added = 0

    for _, row in df.iterrows():
        if row["confidence"] >= CONF_THRESHOLD:
            label = row["final_label"]
            if label in ["GOOD", "BAD"]:
                src = os.path.join(INCOMING_DIR, row["image"])
                dst = os.path.join(TRAIN_DIR, label, row["image"])

                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.copy(src, dst)
                    added += 1

    print(f"Added {added} new images to training dataset.")
    return added

# -------------------------------
# Step 2: Load Model
# -------------------------------
def load_model(num_classes):
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    return model

# -------------------------------
# Step 3: Prepare Data Loaders
# -------------------------------
def get_dataloaders():
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, train_dataset.classes

# -------------------------------
# Step 4: Evaluate
# -------------------------------
def evaluate(model, loader):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)

            preds.extend(predicted.cpu().numpy())
            labels.extend(targets.numpy())

    return accuracy_score(labels, preds)

# -------------------------------
# Step 5: Fine-Tune Model
# -------------------------------
def retrain():
    train_loader, val_loader, classes = get_dataloaders()
    num_classes = len(classes)

    model = load_model(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("Evaluating before retraining...")
    old_acc = evaluate(model, val_loader)
    print(f"Validation Accuracy BEFORE retrain: {old_acc:.4f}")

    model.train()
    for epoch in range(RETRAIN_EPOCHS):
        total_loss = 0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{RETRAIN_EPOCHS}, Loss: {total_loss:.4f}")

    print("Evaluating after retraining...")
    new_acc = evaluate(model, val_loader)
    print(f"Validation Accuracy AFTER retrain: {new_acc:.4f}")

    # Only save if improved
    if new_acc >= old_acc:
        torch.save(model.state_dict(), MODEL_PATH)
        print("Model improved. Saved updated model.")
    else:
        print("Model did not improve. Keeping old model.")

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    added = grow_training_data()

    if added > 0:
        retrain()
    else:
        print("No new high-confidence samples. No retraining needed.")