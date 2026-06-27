"""Training: configurable loss (CE / Dice / Focal / CE+Dice) + Adam, with
mean-IoU validation, plus a final per-class IoU report and confusion matrix."""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import CLASS_NAMES, NUM_CLASSES
from .config import load_config
from .data import S2TileDataset
from .model import build_model


def pick_device(name="auto"):
    """GPU-first with graceful CPU fallback.

    'auto' uses the GPU only if it is present AND can actually run a kernel
    (old cards may be detected but unsupported by the installed torch build).
    """
    if name == "cpu":
        return torch.device("cpu")
    if name in ("auto", "cuda"):
        if torch.cuda.is_available():
            try:
                _ = (torch.zeros(8, device="cuda") + 1).sum().item()  # smoke test
                print(f"[device] using GPU: {torch.cuda.get_device_name(0)}")
                return torch.device("cuda")
            except Exception as e:
                print(f"[device] CUDA present but unusable ({e}); using CPU")
        elif name == "cuda":
            print("[device] CUDA requested but not available; using CPU")
        else:
            print("[device] no CUDA build/GPU detected; using CPU")
    return torch.device("cpu")


def build_loss(name="ce"):
    """Loss factory. name in {ce, dice, focal, ce_dice}."""
    import segmentation_models_pytorch as smp
    name = (name or "ce").lower()
    ce = torch.nn.CrossEntropyLoss()
    if name == "ce":
        return ce
    if name == "dice":
        return smp.losses.DiceLoss(mode="multiclass")
    if name == "focal":
        return smp.losses.FocalLoss(mode="multiclass")
    if name == "ce_dice":
        dice = smp.losses.DiceLoss(mode="multiclass")
        return lambda logits, target: ce(logits, target) + dice(logits, target)
    raise ValueError(f"unknown loss '{name}' (use ce|dice|focal|ce_dice)")


@torch.no_grad()
def mean_iou(logits, target, num_classes):
    """Mean IoU over classes present in this batch (quick per-epoch metric)."""
    pred = logits.argmax(1)
    ious = []
    for c in range(num_classes):
        p, t = pred == c, target == c
        union = (p | t).sum().item()
        if union == 0:
            continue
        ious.append((p & t).sum().item() / union)
    return float(np.mean(ious)) if ious else 0.0


@torch.no_grad()
def confusion_matrix(model, loader, device, num_classes):
    """Accumulate an (N x N) confusion matrix over a data loader."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    model.eval()
    for img, mask in loader:
        img = img.to(device)
        pred = model(img).argmax(1).cpu().numpy().ravel()
        true = mask.numpy().ravel()
        cm += np.bincount(
            true * num_classes + pred, minlength=num_classes ** 2
        ).reshape(num_classes, num_classes)
    return cm


def per_class_iou(cm):
    """IoU per class from a confusion matrix (NaN where the class is absent)."""
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    denom = tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(denom > 0, tp / denom, np.nan)
    return iou


def prf_from_cm(cm):
    """Per-class precision, recall, F1 from a confusion matrix (NaN if absent)."""
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
        recall = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        f1 = np.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall),
            np.nan,
        )
    return precision, recall, f1


def write_report(cm, out_dir, tag=""):
    """Save confusion_matrix{tag}.csv and metrics{tag}.json (IoU + P/R/F1)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / f"confusion_matrix{tag}.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + CLASS_NAMES)
        for i, row in enumerate(cm):
            w.writerow([CLASS_NAMES[i]] + row.tolist())

    iou = per_class_iou(cm)
    precision, recall, f1 = prf_from_cm(cm)
    present = ~np.isnan(iou)

    def _round(x):
        return None if np.isnan(x) else round(float(x), 4)

    metrics = {
        "mean_iou": float(np.nanmean(iou)) if present.any() else 0.0,
        "macro_precision": float(np.nanmean(precision)) if present.any() else 0.0,
        "macro_recall": float(np.nanmean(recall)) if present.any() else 0.0,
        "macro_f1": float(np.nanmean(f1)) if present.any() else 0.0,
        "pixel_accuracy": float(np.diag(cm).sum() / max(1, cm.sum())),
        "per_class": {
            CLASS_NAMES[i]: {
                "iou": _round(iou[i]),
                "precision": _round(precision[i]),
                "recall": _round(recall[i]),
                "f1": _round(f1[i]),
            }
            for i in range(len(CLASS_NAMES))
        },
    }
    (out_dir / f"metrics{tag}.json").write_text(json.dumps(metrics, indent=2))

    print("\n[report] per-class metrics (classes present in validation set):")
    print(f"  {'class':<22}{'IoU':>8}{'prec':>8}{'recall':>8}{'F1':>8}")
    for i in range(len(CLASS_NAMES)):
        if present[i]:
            print(f"  {CLASS_NAMES[i]:<22}{iou[i]:>8.3f}{precision[i]:>8.3f}"
                  f"{recall[i]:>8.3f}{f1[i]:>8.3f}")
    print(f"  {'-' * 52}")
    print(f"  {'macro / mean':<22}{metrics['mean_iou']:>8.3f}"
          f"{metrics['macro_precision']:>8.3f}{metrics['macro_recall']:>8.3f}"
          f"{metrics['macro_f1']:>8.3f}")
    print(f"  pixel accuracy = {metrics['pixel_accuracy']:.4f}")
    print(f"[report] wrote metrics{tag}.json + confusion_matrix{tag}.csv to {out_dir}")
    return metrics


def train(cfg):
    device = pick_device(cfg.train.device)
    print(f"[train] device = {device}")

    tiles = Path(cfg.data.out_dir) / "tiles"
    augment = getattr(cfg.train, "augment", True)
    train_ds = S2TileDataset(tiles, split="train", augment=augment)
    val_ds = S2TileDataset(tiles, split="val", augment=False)
    train_dl = DataLoader(train_ds, batch_size=cfg.train.batch_size,
                          shuffle=True, num_workers=0, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.train.batch_size,
                        shuffle=False, num_workers=0)

    model = build_model(classes=NUM_CLASSES, in_channels=len(cfg.data.bands),
                        encoder=cfg.train.encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    criterion = build_loss(getattr(cfg.train, "loss", "ce"))
    print(f"[train] loss = {getattr(cfg.train, 'loss', 'ce')}, augment = {augment}")

    ckpt_path = Path(cfg.train.ckpt)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best_iou = -1.0

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        running = 0.0
        for img, mask in tqdm(train_dl, desc=f"epoch {epoch}/{cfg.train.epochs}"):
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            loss = criterion(model(img), mask)
            loss.backward()
            optimizer.step()
            running += loss.item()

        model.eval()
        ious = []
        with torch.no_grad():
            for img, mask in val_dl:
                img, mask = img.to(device), mask.to(device)
                ious.append(mean_iou(model(img), mask, NUM_CLASSES))
        val_iou = float(np.mean(ious)) if ious else 0.0
        print(f"  loss={running / max(1, len(train_dl)):.4f}  val_mIoU={val_iou:.4f}")

        if val_iou >= best_iou:
            best_iou = val_iou
            torch.save(
                {"model": model.state_dict(),
                 "encoder": cfg.train.encoder,
                 "in_channels": len(cfg.data.bands),
                 "classes": NUM_CLASSES},
                ckpt_path,
            )
            print(f"  saved checkpoint -> {ckpt_path} (best mIoU {best_iou:.4f})")

    # Final report using the best checkpoint.
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    cm = confusion_matrix(model, val_dl, device, NUM_CLASSES)
    write_report(cm, cfg.predict.out_dir)


if __name__ == "__main__":  # `python -m s2seg.train`
    train(load_config())
