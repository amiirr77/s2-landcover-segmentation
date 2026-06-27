"""s2seg — Sentinel-2 multi-class land-cover segmentation.

Real Sentinel-2 L2A imagery is pulled from a STAC API (Microsoft Planetary
Computer). ESA WorldCover 10m is used as the reference label. A U-Net learns to
predict land cover from the S2 bands; results are served on a Leaflet map.
"""

import os as _os

# --- PROJ data guard (must run before rasterio/pyproj load their C libs) ----
# On some systems (notably Windows with PostgreSQL/PostGIS installed) a global
# PROJ_LIB / PROJ_DATA points at a foreign proj.db and clobbers the PROJ data
# bundled with rasterio/pyproj, causing:
#   "CRSError: The EPSG code is unknown ... DATABASE.LAYOUT.VERSION ...".
# We only *clear* those vars so each library falls back to its OWN bundled,
# internally-consistent PROJ database. We deliberately do NOT pin a path here:
# rasterio and pyproj may ship different PROJ versions, and forcing one path on
# both reintroduces a layout-version mismatch.
for _v in ("PROJ_LIB", "PROJ_DATA"):
    _os.environ.pop(_v, None)
# ---------------------------------------------------------------------------

import numpy as np

__version__ = "0.1.0"

# Official ESA WorldCover v200 classes: raw value -> (name, hex colour).
WORLDCOVER_CLASSES = {
    10:  ("Tree cover",          "#006400"),
    20:  ("Shrubland",           "#ffbb22"),
    30:  ("Grassland",           "#ffff4c"),
    40:  ("Cropland",            "#f096ff"),
    50:  ("Built-up",            "#fa0000"),
    60:  ("Bare / sparse veg.",  "#b4b4b4"),
    70:  ("Snow and ice",        "#f0f0f0"),
    80:  ("Permanent water",     "#0064c8"),
    90:  ("Herbaceous wetland",  "#0096a0"),
    95:  ("Mangroves",           "#00cf75"),
    100: ("Moss and lichen",     "#fae6a0"),
}

# Contiguous 0..N-1 training indices, in ascending raw-value order.
RAW_VALUES = sorted(WORLDCOVER_CLASSES)
VALUE_TO_IDX = {raw: i for i, raw in enumerate(RAW_VALUES)}
NUM_CLASSES = len(RAW_VALUES)

CLASS_NAMES = [WORLDCOVER_CLASSES[v][0] for v in RAW_VALUES]
CLASS_COLORS = [WORLDCOVER_CLASSES[v][1] for v in RAW_VALUES]


def hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def palette() -> np.ndarray:
    """(NUM_CLASSES, 3) uint8 colour table aligned with training indices."""
    return np.array([hex_to_rgb(c) for c in CLASS_COLORS], dtype="uint8")


def remap_worldcover(raw: np.ndarray) -> np.ndarray:
    """Map raw WorldCover values (10,20,...) to contiguous indices (0..N-1)."""
    out = np.zeros(raw.shape, dtype="uint8")
    for value, idx in VALUE_TO_IDX.items():
        out[raw == value] = idx
    return out


def legend() -> list:
    """List of {idx, name, color} for the web viewer."""
    return [
        {"idx": i, "name": CLASS_NAMES[i], "color": CLASS_COLORS[i]}
        for i in range(NUM_CLASSES)
    ]
