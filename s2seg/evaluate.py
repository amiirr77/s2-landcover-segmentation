"""Evaluate the trained model on the validation split WITHOUT retraining.

Loads the best checkpoint, rebuilds the validation loader, computes a confusion
matrix and writes the full report (IoU + precision/recall/F1) to predict.out_dir.

    python -m s2seg.evaluate
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from . import NUM_CLASSES
from .config import load_config
from .data import S2TileDataset
from .model import build_model
from .train import confusion_matrix, pick_device, write_report


def evaluate(cfg):
    device = pick_device(cfg.train.device)
    val_ds = S2TileDataset(Path(cfg.data.out_dir) / "tiles", split="val", augment=False)
    val_dl = DataLoader(val_ds, batch_size=cfg.train.batch_size,
                        shuffle=False, num_workers=0)

    ckpt = torch.load(cfg.train.ckpt, map_location=device)
    model = build_model(classes=ckpt["classes"], in_channels=ckpt["in_channels"],
                        encoder=ckpt["encoder"], encoder_weights=None).to(device)
    model.load_state_dict(ckpt["model"])

    cm = confusion_matrix(model, val_dl, device, NUM_CLASSES)
    return write_report(cm, cfg.predict.out_dir)


if __name__ == "__main__":
    evaluate(load_config())
