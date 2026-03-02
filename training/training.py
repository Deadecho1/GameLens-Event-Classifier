import os
import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import json
import csv
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms

import timm
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


# -----------------------------
# Config
# -----------------------------
@dataclass
class TrainCfg:
    data_root: str = r"path/to/dataset"  # contains train/ valid/ test/
    model_name: str = "convnext_tiny"  # strong for UI-ish screenshots
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0

    # imbalance controls
    use_weighted_sampler: bool = True
    use_class_weighted_loss: bool = True

    # model selection metric: "macro_recall" or "macro_f1"
    selection_metric: str = "macro_recall"

    # early stopping
    patience: int = 6

    # misc
    seed: int = 1337
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def append_csv_row(csv_path: str, fieldnames: List[str], row: Dict) -> None:
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def plot_learning_curves(history: List[Dict], out_dir: str) -> None:
    epochs = [h["epoch"] for h in history]

    plt.figure()
    plt.plot(epochs, [h["train_loss"] for h in history])
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Training Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "learning_train_loss.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(epochs, [h["val_macro_precision"] for h in history], label="macro_precision")
    plt.plot(epochs, [h["val_macro_recall"] for h in history], label="macro_recall")
    plt.plot(epochs, [h["val_macro_f1"] for h in history], label="macro_f1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Macro Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "learning_val_macro_metrics.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(epochs, [h["selected_score"] for h in history], label=f"selected ({history[0]['selected_metric']})")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Selected Metric Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "learning_selected_metric.png"), dpi=200)
    plt.close()


def plot_confusion_matrix(cm: List[List[int]], class_names: List[str], out_path: str, normalize: bool = False) -> None:
    import numpy as np

    cm = np.array(cm, dtype=float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm = cm / row_sums

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    num_classes: int,
) -> Dict:
    model.eval()
    all_preds = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy()

    # macro metrics treat each class equally (good for imbalance)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # per-class
    p_c, r_c, f1_c, support_c = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    return {
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(p_weighted),
        "weighted_recall": float(r_weighted),
        "weighted_f1": float(f1_weighted),
        "per_class_precision": p_c.tolist(),
        "per_class_recall": r_c.tolist(),
        "per_class_f1": f1_c.tolist(),
        "per_class_support": support_c.tolist(),
        "confusion_matrix": cm.tolist(),
    }


def metric_value(metrics: Dict, which: str) -> float:
    if which == "macro_recall":
        return metrics["macro_recall"]
    if which == "macro_f1":
        return metrics["macro_f1"]
    raise ValueError(f"Unknown selection metric: {which}")


def make_class_weights_from_dataset(ds: ImageFolder) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(ds.targets), minlength=len(ds.classes)).float()
    weights = 1.0 / torch.clamp(counts, min=1.0)
    weights = weights * (weights.numel() / weights.sum())
    return weights


def make_weighted_sampler(ds: ImageFolder) -> WeightedRandomSampler:
    counts = torch.bincount(torch.tensor(ds.targets), minlength=len(ds.classes)).float()
    class_weight = 1.0 / torch.clamp(counts, min=1.0)
    sample_weights = class_weight[torch.tensor(ds.targets)]
    return WeightedRandomSampler(
        weights=sample_weights.double(),
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.05, hue=0.02)],
                p=0.35,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    eval_tf = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return train_tf, eval_tf


# -----------------------------
# Main Train
# -----------------------------
def train():
    cfg = TrainCfg()
    set_seed(cfg.seed)

    # Experiment output directory
    run_dir = os.path.join(os.getcwd(), "runs", time.strftime("%Y%m%d_%H%M%S"))
    ensure_dir(run_dir)
    save_json(os.path.join(run_dir, "config.json"), cfg.__dict__)

    data_root = Path(cfg.data_root)
    train_dir = data_root / "train"
    valid_dir = data_root / "valid"
    test_dir = data_root / "test"

    train_tf, eval_tf = build_transforms(cfg.img_size)

    train_ds = ImageFolder(str(train_dir), transform=train_tf)
    valid_ds = ImageFolder(str(valid_dir), transform=eval_tf)
    test_ds = ImageFolder(str(test_dir), transform=eval_tf)

    num_classes = len(train_ds.classes)
    assert num_classes > 1, "Need at least 2 classes."

    save_json(os.path.join(run_dir, "classes.json"), {"classes": train_ds.classes})

    print("Classes:", train_ds.classes)
    print("Num classes:", num_classes)
    print("Train size:", len(train_ds), "Valid size:", len(valid_ds), "Test size:", len(test_ds))
    print("Artifacts will be saved to:", run_dir)

    # Sampler for imbalance (train only)
    sampler = make_weighted_sampler(train_ds) if cfg.use_weighted_sampler else None
    shuffle = (sampler is None)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # Model
    model = timm.create_model(cfg.model_name, pretrained=True, num_classes=num_classes)
    model = model.to(cfg.device)

    # Loss (optionally class-weighted)
    if cfg.use_class_weighted_loss:
        class_weights = make_class_weights_from_dataset(train_ds).to(cfg.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.label_smoothing)
        print("Using class-weighted CrossEntropyLoss.")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_score = -1.0
    best_state = None
    epochs_no_improve = 0

    # Logging
    history: List[Dict] = []
    csv_path = os.path.join(run_dir, "history.csv")
    csv_fields = [
        "epoch",
        "train_loss",
        "val_macro_precision",
        "val_macro_recall",
        "val_macro_f1",
        "selected_metric",
        "selected_score",
        "lr",
    ]

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False)
        for images, targets in pbar:
            images = images.to(cfg.device, non_blocking=True)
            targets = targets.to(cfg.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            bs = images.size(0)
            running_loss += loss.item() * bs
            seen += bs
            pbar.set_postfix(loss=running_loss / max(1, seen))

        scheduler.step()
        train_loss = running_loss / max(1, seen)

        # Validate: use precision/recall metrics (macro)
        val_metrics = evaluate(model, valid_loader, cfg.device, num_classes)
        score = metric_value(val_metrics, cfg.selection_metric)
        current_lr = optimizer.param_groups[0]["lr"]

        # Save per-epoch metrics
        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_macro_precision": float(val_metrics["macro_precision"]),
            "val_macro_recall": float(val_metrics["macro_recall"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "selected_metric": cfg.selection_metric,
            "selected_score": float(score),
            "lr": float(current_lr),
        }
        history.append(row)
        append_csv_row(csv_path, csv_fields, row)

        # Save per-epoch full metrics + confusion matrix plots
        save_json(os.path.join(run_dir, f"val_metrics_epoch_{epoch:03d}.json"), val_metrics)
        plot_confusion_matrix(
            val_metrics["confusion_matrix"],
            train_ds.classes,
            os.path.join(run_dir, f"val_cm_epoch_{epoch:03d}.png"),
            normalize=False,
        )
        plot_confusion_matrix(
            val_metrics["confusion_matrix"],
            train_ds.classes,
            os.path.join(run_dir, f"val_cm_norm_epoch_{epoch:03d}.png"),
            normalize=True,
        )

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f}  "
            f"val_macroP={val_metrics['macro_precision']:.3f}  "
            f"val_macroR={val_metrics['macro_recall']:.3f}  "
            f"val_macroF1={val_metrics['macro_f1']:.3f}  "
            f"SELECT({cfg.selection_metric})={score:.3f}  "
            f"lr={current_lr:.2e}"
        )

        # Print per-class precision/recall
        per_r = val_metrics["per_class_recall"]
        per_p = val_metrics["per_class_precision"]
        for i, cname in enumerate(train_ds.classes):
            print(f"  {cname:15s}  P={per_p[i]:.3f}  R={per_r[i]:.3f}  support={val_metrics['per_class_support'][i]}")

        # Early stopping + best checkpoint by chosen metric
        if score > best_score + 1e-6:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0

            best_path = os.path.join(run_dir, "best_model.pt")
            torch.save(best_state, best_path)
            print(f"  Saved best model: {best_path}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve}/{cfg.patience}")

        if epochs_no_improve >= cfg.patience:
            print("Early stopping triggered.")
            break

    # Save learning curves
    save_json(os.path.join(run_dir, "history.json"), history)
    if len(history) > 0:
        plot_learning_curves(history, run_dir)

    # Load best, evaluate on test
    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, cfg.device, num_classes)
    test_score = metric_value(test_metrics, cfg.selection_metric)

    save_json(os.path.join(run_dir, "test_metrics.json"), test_metrics)
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        train_ds.classes,
        os.path.join(run_dir, "test_cm.png"),
        normalize=False,
    )
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        train_ds.classes,
        os.path.join(run_dir, "test_cm_norm.png"),
        normalize=True,
    )

    print("\n=== TEST RESULTS ===")
    print(
        f"test_macroP={test_metrics['macro_precision']:.3f}  "
        f"test_macroR={test_metrics['macro_recall']:.3f}  "
        f"test_macroF1={test_metrics['macro_f1']:.3f}  "
        f"SELECT({cfg.selection_metric})={test_score:.3f}"
    )

    per_r = test_metrics["per_class_recall"]
    per_p = test_metrics["per_class_precision"]
    for i, cname in enumerate(train_ds.classes):
        print(f"  {cname:15s}  P={per_p[i]:.3f}  R={per_r[i]:.3f}  support={test_metrics['per_class_support'][i]}")

    print("\nSaved best model as:", os.path.join(run_dir, "best_model.pt"))
    print("All artifacts saved under:", run_dir)


if __name__ == "__main__":
    train()