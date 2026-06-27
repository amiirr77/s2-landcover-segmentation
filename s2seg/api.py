"""FastAPI service: serves the Leaflet viewer + prediction outputs.

  GET  /                serves the map viewer
  GET  /api/result      bounds + legend for the latest prediction
  POST /api/predict     fetch a fresh S2 scene for a bbox and run inference
  /outputs/*            static PNG/JSON artefacts
"""

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import load_config

cfg = load_config()
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(cfg.predict.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Sentinel-2 Land-Cover Segmentation")
app.mount("/outputs", StaticFiles(directory=str(OUT_DIR)), name="outputs")
app.mount("/static", StaticFiles(directory=str(ROOT / "web")), name="static")


@app.get("/")
def index():
    return FileResponse(str(ROOT / "web" / "index.html"))


@app.get("/api/result")
def api_result():
    path = OUT_DIR / "result.json"
    if not path.exists():
        raise HTTPException(404, "No prediction yet. Run training + predict first.")
    return json.loads(path.read_text())


class PredictRequest(BaseModel):
    bbox: list[float]          # [min_lon, min_lat, max_lon, max_lat]
    time_range: str | None = None


@app.post("/api/predict")
def api_predict(req: PredictRequest):
    """Pull a fresh scene for `bbox` and run the trained model on it."""
    from .data import fetch_scene
    from .predict import predict

    image_path, _ = fetch_scene(
        req.bbox, req.time_range or cfg.aoi.time_range, cfg.aoi.max_cloud,
        cfg.data.bands, Path(cfg.data.out_dir) / "inference",
    )
    return predict(cfg, image_path=image_path)
