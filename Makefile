.PHONY: setup smoke data train predict serve all

setup:
	pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
	pip install -r requirements.txt

smoke:                 ## offline check: model builds + one train step (no network)
	python -m tests.test_smoke

data:                  ## fetch real Sentinel-2 + WorldCover, then tile
	python -m s2seg.data

train:                 ## train the U-Net on the tiles
	python -m s2seg.train

predict:               ## run inference on the fetched scene
	python -m s2seg.predict

evaluate:              ## recompute IoU / precision / recall / F1 (no retraining)
	python -m s2seg.evaluate

serve:                 ## launch the FastAPI + Leaflet viewer
	uvicorn s2seg.api:app --host 0.0.0.0 --port 8000 --reload

all: data train predict serve
