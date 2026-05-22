import argparse
import json
import os
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "classifier", "efficientnet_b0_quality.pt")
DEFAULT_CLASS_JSON = os.path.join(PROJECT_ROOT, "classifier", "class_name.json")
DEFAULT_REPORT_PATH = os.path.join(DATA_ROOT, "processed", "training_report.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 for image quality classification")
    parser.add_argument("--train-dir", default=os.path.join(DATA_ROOT, "train"))
    parser.add_argument("--val-dir", default=os.path.join(DATA_ROOT, "val"))
    parser.add_argument("--test-dir", default=os.path.join(DATA_ROOT, "test"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--output-model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-class-json", default=DEFAULT_CLASS_JSON)
    parser.add_argument("--output-report", default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


def build_dataloaders(
    train_dir: str, val_dir: str, test_dir: str, batch_size: int, num_workers: int
) -> Tuple[DataLoader, DataLoader, DataLoader, list]:
    train_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_tf)
    test_ds = datasets.ImageFolder(test_dir, transform=eval_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    return train_loader, val_loader, test_loader, train_ds.classes


def build_model(num_classes: int, freeze_backbone: bool):
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
    return model.to(DEVICE)


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.set_grad_enabled(is_train):
        for imgs, targets in loader:
            imgs = imgs.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(imgs)
            loss = criterion(outputs, targets)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item()) * imgs.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.detach().cpu().tolist())
            all_targets.extend(targets.detach().cpu().tolist())

    mean_loss = total_loss / max(len(loader.dataset), 1)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    acc = (torch.tensor(all_preds) == torch.tensor(all_targets)).float().mean().item()
    return mean_loss, acc, macro_f1, all_targets, all_preds


def evaluate_for_report(model, loader, class_names) -> Dict:
    criterion = nn.CrossEntropyLoss()
    loss, acc, macro_f1, y_true, y_pred = run_epoch(model, loader, criterion, optimizer=None)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))).tolist()
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    return {
        "loss": loss,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)

    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        args.train_dir, args.val_dir, args.test_dir, args.batch_size, args.num_workers
    )
    print(f"Device: {DEVICE}")
    print(f"Classes: {class_names}")
    print(f"Train images: {len(train_loader.dataset)} | Val images: {len(val_loader.dataset)} | Test images: {len(test_loader.dataset)}")

    model = build_model(num_classes=len(class_names), freeze_backbone=args.freeze_backbone)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    best_val_f1 = -1.0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1, _, _ = run_epoch(model, train_loader, criterion, optimizer=optimizer)
        val_loss, val_acc, val_f1, _, _ = run_epoch(model, val_loader, criterion, optimizer=None)
        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_macro_f1": train_f1,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_macro_f1": val_f1,
            }
        )
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), args.output_model)

    with open(args.output_class_json, "w", encoding="utf-8") as f:
        json.dump(class_names, f)

    val_report = evaluate_for_report(model, val_loader, class_names)
    test_report = evaluate_for_report(model, test_loader, class_names)
    final_report = {
        "device": DEVICE,
        "classes": class_names,
        "train_size": len(train_loader.dataset),
        "val_size": len(val_loader.dataset),
        "test_size": len(test_loader.dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "best_val_macro_f1": best_val_f1,
        "history": history,
        "validation": val_report,
        "test": test_report,
    }

    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print(f"Saved model: {args.output_model}")
    print(f"Saved class map: {args.output_class_json}")
    print(f"Saved training report: {args.output_report}")
    print(f"Best validation macro-F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()
