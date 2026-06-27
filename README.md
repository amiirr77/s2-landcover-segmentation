# Sentinel-2 Land-Cover Segmentation

An end-to-end pipeline that pulls **real Sentinel-2 L2A** imagery from a public
STAC API, trains a **U-Net** to predict multi-class land cover using **ESA
WorldCover 10 m** as reference labels, and serves the result on an interactive
**Leaflet** map — packaged with **Docker**.

It is deliberately small but production-shaped: data access, training,
inference, and serving are separate, testable modules.

```
                        Microsoft Planetary Computer (STAC)
                         ┌───────────────┬──────────────────┐
                         │ Sentinel-2 L2A│  ESA WorldCover   │
                         │ (B04 B03 B02  │  (10 m, 11-class  │
                         │  B08, 10 m)   │   reference label)│
                         └───────┬───────┴────────┬──────────┘
                                 │  fetch + align (odc-stac)
                                 ▼
                          image.tif / label.tif
                                 │  tiling (256×256)
                                 ▼
                        ┌──────────────────┐
                        │  U-Net (smp,     │  CrossEntropy + Adam
                        │  ResNet-34 enc.) │  val metric: mean IoU
                        └────────┬─────────┘
                                 │  sliding-window inference
                                 ▼
                  mask.tif (georeferenced) + PNG overlays
                                 │  FastAPI
                                 ▼
                        Leaflet map viewer
```

## What it demonstrates
- Real satellite data access over **STAC** (no manual downloads).
- A reproducible **PyTorch** segmentation training loop with a real metric.
- Correct **geospatial bookkeeping** — UTM projection, georeferenced output,
  reprojected WGS-84 bounds for web display.
- A small **FastAPI** service and a web map — i.e. turning an analysis into a
  *system*, not a notebook.

## Quickstart

> Requires Python 3.10+. Training is fine on CPU for a small AOI; a GPU is faster.

```bash
# 1. install (CPU torch shown; for GPU use the matching pytorch.org command)
make setup            # or: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
                      #     pip install -r requirements.txt

# 2. (optional) offline sanity check — no network, no data needed
make smoke

# 3. fetch real Sentinel-2 + WorldCover for the AOI in config.yaml, then tile
make data

# 4. train
make train

# 5. run inference on the fetched scene
make predict

# 6. open the map
make serve            # -> http://localhost:8000
```

With Docker:

```bash
docker compose build
docker compose run --rm api python -m s2seg.data
docker compose run --rm api python -m s2seg.train
docker compose run --rm api python -m s2seg.predict
docker compose up         # -> http://localhost:8000
```

## Configuration

Everything is driven by `config.yaml`. The most useful knobs:

| key | meaning |
|-----|---------|
| `aoi.bbox` | area of interest, `[min_lon, min_lat, max_lon, max_lat]` |
| `aoi.time_range` | STAC datetime range, e.g. `"2023-06-01/2023-09-30"` |
| `aoi.max_cloud` | reject scenes cloudier than this (%) |
| `data.bands` | bands to load — order defines the RGB preview |
| `train.epochs`, `train.lr`, `train.encoder` | training hyper-parameters |
| `train.device` | `auto` / `cpu` / `cuda` |

## How it works

**Data (`s2seg/data.py`).** Searches the `sentinel-2-l2a` collection on
Microsoft Planetary Computer, picks the least-cloudy scene in the AOI, and loads
the chosen bands into a common UTM grid with `odc-stac`. ESA WorldCover is loaded
onto the *same* grid (nearest-neighbour) so pixels line up, then remapped from
its raw values (10, 20, …) to contiguous class indices. Both are written as
GeoTIFFs and sliced into 256×256 patches.

**Model (`s2seg/model.py`).** `segmentation_models_pytorch.Unet` with an
ImageNet-pretrained ResNet-34 encoder. `smp` adapts the first convolution to 4
input channels automatically, so the NIR band is used alongside RGB.

**Training (`s2seg/train.py`).** CrossEntropy + Adam, with mean IoU on a held-out
split and best-checkpoint saving.

**Inference (`s2seg/predict.py`).** Sliding-window prediction over the full
scene, producing a georeferenced `mask.tif`, a coloured `mask.png` overlay, a
percentile-stretched true-colour `image.png`, and a `result.json` with WGS-84
bounds + legend.

**Serving (`s2seg/api.py` + `web/index.html`).** FastAPI serves the artefacts
and a Leaflet viewer that overlays the RGB and predicted mask with a toggle and a
class legend. `POST /api/predict` with a `bbox` re-runs the model on a fresh
scene.

## Notes & honest limitations
- WorldCover is itself a model product, so the network learns to *imitate* it —
  good for a demo, not a new source of truth. A real project would use
  hand-labelled tiles or a benchmark dataset for evaluation.
- A single scene is a small training set; expect modest IoU. More scenes / dates
  and augmentation help a lot and are the obvious next step.
- The Planetary Computer needs internet access; behind a restrictive network,
  point the catalog at Copernicus Data Space instead (see `data.py`).

## Possible next steps
- Multiple scenes and dates; data augmentation.
- Swap CrossEntropy for Dice/Focal loss for class imbalance.
- Export to ONNX and add COG/XYZ tile output for slippy-map streaming.
- A small GitHub Actions workflow running `make smoke`.
