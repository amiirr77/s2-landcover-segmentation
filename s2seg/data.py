"""Data layer.

  fetch_scene()  -> download one Sentinel-2 L2A scene + aligned ESA WorldCover
                    label from Microsoft Planetary Computer, save as GeoTIFFs.
  build_tiles()  -> cut the scene into training patches (.npy).
  S2TileDataset  -> PyTorch Dataset over those patches.

Network is only required by fetch_scene(); tiling/Dataset are offline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS as RioCRS

from . import remap_worldcover

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_REFLECTANCE_SCALE = 10000.0  # L2A surface reflectance is stored x10000


# --------------------------------------------------------------------------- #
# Fetching real Sentinel-2 + WorldCover via STAC
# --------------------------------------------------------------------------- #
def utm_epsg(lon: float, lat: float) -> int:
    """Pick the UTM zone EPSG code for a lon/lat so 10 m == 10 metres."""
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def fetch_scenes(bbox, time_range, max_cloud, bands, out_dir, num_scenes=3):
    """Download the N least-cloudy S2 scenes in the AOI onto a shared grid,
    plus one aligned ESA WorldCover label.

    All scenes are loaded onto the SAME geobox, so the single WorldCover label
    lines up with every scene. More scenes (often different dates) => a larger,
    more varied training set.

    bbox: [min_lon, min_lat, max_lon, max_lat]
    Returns (list_of_image_paths, label_path).
    """
    import odc.stac
    import planetary_computer as pc
    from pystac_client import Client

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog = Client.open(PC_STAC, modifier=pc.sign_inplace)

    items = list(
        catalog.search(
            collections=["sentinel-2-l2a"], bbox=bbox, datetime=time_range,
            query={"eo:cloud_cover": {"lt": max_cloud}},
        ).items()
    )
    if not items:
        raise RuntimeError(
            "No Sentinel-2 scenes found. Widen `time_range`/`bbox` or raise `max_cloud`."
        )
    items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100))
    items = items[:max(1, num_scenes)]

    lon, lat = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    epsg = utm_epsg(lon, lat)

    geobox = None
    image_paths = []
    for i, item in enumerate(items):
        print(f"[fetch] scene {i}: {item.id} "
              f"(cloud {item.properties.get('eo:cloud_cover', 0):.1f}%)")
        if geobox is None:
            s2 = odc.stac.load(
                [item], bands=list(bands), bbox=bbox,
                crs=f"EPSG:{epsg}", resolution=10, resampling="bilinear",
            ).isel(time=0)
            geobox = s2.odc.geobox
        else:
            # reuse the first scene's grid so all images stay pixel-aligned
            s2 = odc.stac.load(
                [item], bands=list(bands), geobox=geobox, resampling="bilinear",
            ).isel(time=0)
        img = np.stack([s2[b].values for b in bands]).astype("float32")
        path = out_dir / f"image_{i:02d}.tif"
        _write_geotiff(path, img, geobox.transform,
                       RioCRS.from_epsg(geobox.crs.epsg))
        image_paths.append(path)

    wc_items = list(catalog.search(collections=["esa-worldcover"], bbox=bbox).items())
    if not wc_items:
        raise RuntimeError("No ESA WorldCover tiles cover this AOI.")
    wc = odc.stac.load(
        wc_items, bands=["map"], geobox=geobox, resampling="nearest",
    ).isel(time=0)
    label = remap_worldcover(wc["map"].values.astype("int32"))
    label_path = out_dir / "label.tif"
    _write_geotiff(label_path, label[None], geobox.transform,
                   RioCRS.from_epsg(geobox.crs.epsg))

    print(f"[fetch] wrote {len(image_paths)} scene(s) + {label_path}")
    return image_paths, label_path


def fetch_scene(bbox, time_range, max_cloud, bands, out_dir):
    """Backward-compatible single-scene fetch. Returns (image_path, label_path)."""
    image_paths, label_path = fetch_scenes(
        bbox, time_range, max_cloud, bands, out_dir, num_scenes=1
    )
    return image_paths[0], label_path


def _write_geotiff(path, array, transform, crs):
    array = np.asarray(array)
    if array.ndim == 2:
        array = array[None]
    count, height, width = array.shape
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=count,
        dtype=array.dtype, crs=crs, transform=transform, compress="deflate",
    ) as dst:
        for i in range(count):
            dst.write(array[i], i + 1)


# --------------------------------------------------------------------------- #
# Tiling
# --------------------------------------------------------------------------- #
def build_tiles(image_paths, label_path, out_dir, size=256, stride=256):
    """Cut scene(s) into (size x size) patches; skip empty/no-data tiles.

    image_paths may be a single path or a list (multi-scene). All scenes are
    tiled against the same shared label.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(image_paths, (str, Path)):
        image_paths = [image_paths]

    # clear any stale tiles from a previous run
    for old in list(out_dir.glob("img_*.npy")) + list(out_dir.glob("lbl_*.npy")):
        old.unlink()

    with rasterio.open(label_path) as src:
        lbl = src.read(1).astype("uint8")            # (H, W)

    n = 0
    for image_path in image_paths:
        with rasterio.open(image_path) as src:
            img = src.read().astype("float32")       # (C, H, W)
        _, H, W = img.shape
        for y in range(0, H - size + 1, stride):
            for x in range(0, W - size + 1, stride):
                img_t = img[:, y:y + size, x:x + size]
                lbl_t = lbl[y:y + size, x:x + size]
                if not np.any(img_t):                # all no-data -> skip
                    continue
                if not np.isfinite(img_t).all():     # partial / NaN coverage -> skip
                    continue
                np.save(out_dir / f"img_{n:05d}.npy", img_t)
                np.save(out_dir / f"lbl_{n:05d}.npy", lbl_t)
                n += 1
    if n == 0:
        raise RuntimeError(
            "No valid tiles produced. Is the AOI larger than one tile size?"
        )
    print(f"[tiles] wrote {n} tiles of {size}x{size} to {out_dir}")
    return n


# --------------------------------------------------------------------------- #
# PyTorch Dataset
# --------------------------------------------------------------------------- #
class S2TileDataset:
    """Yields (image[C,H,W] float32 in 0..1, mask[H,W] int64) tensors.

    When augment=True, applies random horizontal/vertical flips and 90-degree
    rotations identically to image and mask (cheap, dependency-free).
    """

    def __init__(self, tile_dir, split="train", val_frac=0.2, seed=0, augment=False):
        files = sorted(Path(tile_dir).glob("img_*.npy"))
        if not files:
            raise RuntimeError(f"No tiles in {tile_dir}; run build_tiles first.")
        rng = np.random.default_rng(seed)
        rng.shuffle(files)
        n_val = max(1, int(len(files) * val_frac))
        self.files = files[n_val:] if split == "train" else files[:n_val]
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def _augment(self, img, lbl):
        if np.random.rand() < 0.5:                 # horizontal flip
            img, lbl = img[:, :, ::-1], lbl[:, ::-1]
        if np.random.rand() < 0.5:                 # vertical flip
            img, lbl = img[:, ::-1, :], lbl[::-1, :]
        k = np.random.randint(0, 4)                # 0/90/180/270 rotation
        if k:
            img = np.rot90(img, k, axes=(1, 2))
            lbl = np.rot90(lbl, k, axes=(0, 1))
        return np.ascontiguousarray(img), np.ascontiguousarray(lbl)

    def __getitem__(self, i):
        import torch  # lazy import keeps the object picklable for DataLoader workers
        img_path = self.files[i]
        lbl_path = img_path.with_name(img_path.name.replace("img_", "lbl_"))
        img = np.load(img_path).astype("float32")
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        img = np.clip(img / S2_REFLECTANCE_SCALE, 0.0, 1.0)
        lbl = np.load(lbl_path).astype("int64")
        if self.augment:
            img, lbl = self._augment(img, lbl)
        return torch.from_numpy(img), torch.from_numpy(lbl)


if __name__ == "__main__":  # `python -m s2seg.data`
    from .config import load_config
    cfg = load_config()
    out_dir = Path(cfg.data.out_dir)
    existing = sorted(out_dir.glob("image_*.tif"))
    label_path = out_dir / "label.tif"
    if existing and label_path.exists():
        print(f"[data] reusing {len(existing)} existing scene(s) "
              f"(delete data/*.tif to force re-download)")
        image_paths = existing
    else:
        image_paths, label_path = fetch_scenes(
            cfg.aoi.bbox, cfg.aoi.time_range, cfg.aoi.max_cloud,
            cfg.data.bands, cfg.data.out_dir,
            num_scenes=getattr(cfg.data, "num_scenes", 1),
        )
    build_tiles(
        image_paths, label_path, out_dir / "tiles",
        size=cfg.data.tile_size, stride=cfg.data.stride,
    )
