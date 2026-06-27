"""Inference on a full Sentinel-2 scene.

Outputs (into cfg.predict.out_dir):
  mask.tif      georeferenced class map
  mask.png      coloured overlay (transparent background) for Leaflet
  image.png     true-colour RGB preview (percentile stretch) for Leaflet
  result.json   {bounds: [[s,w],[n,e]], legend: [...]}  consumed by the viewer
"""

import json
from pathlib import Path

import numpy as np
import rasterio
import torch
from PIL import Image
from rasterio.warp import transform_bounds

from . import legend, palette
from .config import load_config
from .data import S2_REFLECTANCE_SCALE
from .model import build_model
from .train import pick_device


def _stretch_rgb(rgb, low=2, high=98):
    """Percentile stretch each band of an (3,H,W) array to uint8."""
    out = np.zeros_like(rgb, dtype="uint8")
    for i in range(3):
        band = rgb[i]
        lo, hi = np.percentile(band, [low, high])
        if hi <= lo:
            hi = lo + 1.0
        out[i] = np.clip((band - lo) / (hi - lo) * 255, 0, 255).astype("uint8")
    return np.transpose(out, (1, 2, 0))  # (H, W, 3)


@torch.no_grad()
def _sliding_predict(model, img, device, tile=256, classes=11):
    """img: (C,H,W) float in 0..1 -> mask (H,W) uint8."""
    c, h, w = img.shape
    ph = (tile - h % tile) % tile
    pw = (tile - w % tile) % tile
    padded = np.pad(img, ((0, 0), (0, ph), (0, pw)), mode="reflect")
    _, H, W = padded.shape
    out = np.zeros((H, W), dtype="uint8")
    for y in range(0, H, tile):
        for x in range(0, W, tile):
            patch = padded[:, y:y + tile, x:x + tile]
            t = torch.from_numpy(patch[None]).float().to(device)
            pred = model(t).argmax(1)[0].cpu().numpy().astype("uint8")
            out[y:y + tile, x:x + tile] = pred
    return out[:h, :w]


def _default_image(cfg):
    """Pick a scene to predict on: image_00.tif, else image.tif, else first match."""
    d = Path(cfg.data.out_dir)
    for name in ("image_00.tif", "image.tif"):
        if (d / name).exists():
            return d / name
    matches = sorted(d.glob("image_*.tif"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"No scene found in {d}. Run `python -m s2seg.data` first."
    )


def predict(cfg, image_path=None):
    device = pick_device(cfg.train.device)
    image_path = Path(image_path) if image_path else _default_image(cfg)
    out_dir = Path(cfg.predict.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(cfg.train.ckpt, map_location=device)
    model = build_model(classes=ckpt["classes"], in_channels=ckpt["in_channels"],
                        encoder=ckpt["encoder"], encoder_weights=None).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    with rasterio.open(image_path) as src:
        raw = src.read().astype("float32")
        crs, transform, bounds = src.crs, src.transform, src.bounds
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

    img = np.clip(raw / S2_REFLECTANCE_SCALE, 0.0, 1.0)
    mask = _sliding_predict(model, img, device,
                            tile=cfg.data.tile_size, classes=ckpt["classes"])

    # georeferenced class map
    with rasterio.open(out_dir / "mask.tif", "w", driver="GTiff",
                       height=mask.shape[0], width=mask.shape[1], count=1,
                       dtype="uint8", crs=crs, transform=transform,
                       compress="deflate") as dst:
        dst.write(mask, 1)

    # coloured overlay PNG
    pal = palette()
    alpha = np.full(mask.shape, 200, dtype="uint8")
    Image.fromarray(np.dstack([pal[mask], alpha]), mode="RGBA").save(out_dir / "mask.png")

    # true-colour preview (bands are ordered R,G,B,NIR via config)
    Image.fromarray(_stretch_rgb(raw[:3])).save(out_dir / "image.png")

    # reference label (ground truth) for the SAME scene, if available, so the
    # web viewer can show prediction vs. reference side by side.
    has_reference = False
    label_path = image_path.parent / "label.tif"
    if label_path.exists():
        with rasterio.open(label_path) as src:
            ref = src.read(1).astype("uint8")
        if ref.shape == mask.shape:
            Image.fromarray(np.dstack([pal[ref], alpha]), mode="RGBA").save(
                out_dir / "label.png"
            )
            has_reference = True

    west, south, east, north = transform_bounds(
        crs, "EPSG:4326", bounds.left, bounds.bottom, bounds.right, bounds.top
    )
    result = {
        "bounds": [[south, west], [north, east]],
        "legend": legend(),
        "has_reference": has_reference,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    extra = " + label.png" if has_reference else ""
    print(f"[predict] wrote mask.tif / mask.png / image.png{extra} / result.json to {out_dir}")

    # If we have a reference for this (possibly unseen) scene, report how well
    # the model generalises to it — the real spatial-generalization number.
    if has_reference:
        from .train import write_report
        n = palette().shape[0]
        cm = np.bincount(
            ref.ravel().astype(np.int64) * n + mask.ravel().astype(np.int64),
            minlength=n * n,
        ).reshape(n, n)
        print("\n[predict] generalization metrics on this scene vs WorldCover:")
        write_report(cm, out_dir, tag="_predict")
    return result


if __name__ == "__main__":  # `python -m s2seg.predict`
    cfg = load_config()
    pa = getattr(cfg.predict, "aoi", None)
    if pa is not None and getattr(pa, "bbox", None):
        from .data import fetch_scenes
        inf_dir = Path(cfg.data.out_dir) / "inference"
        print(f"[predict] fetching unseen scene for bbox {pa.bbox}")
        image_paths, _ = fetch_scenes(
            pa.bbox,
            getattr(pa, "time_range", cfg.aoi.time_range),
            getattr(pa, "max_cloud", cfg.aoi.max_cloud),
            cfg.data.bands, inf_dir, num_scenes=1,
        )
        predict(cfg, image_path=image_paths[0])
    else:
        predict(cfg)
