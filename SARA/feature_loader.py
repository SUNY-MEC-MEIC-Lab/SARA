from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

from .io_utils import ensure_dir, iter_feature_paths, list_image_paths, stem_from_path


def load_features(npz_path: str) -> Dict[str, np.ndarray]:
    """
    Load local features from an .npz file.
    Expected keys: 'keypoints', 'descriptors', 'scores'
    Optional keys: 'image_shape' or 'shape'
    """
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {npz_path}")

    with np.load(path, allow_pickle=False) as data:
        kpt = data["keypoints"].astype(np.float32)
        desc = data["descriptors"].astype(np.float32)
        score = data["scores"].astype(np.float32)
        shape = tuple(data["image_shape"]) if "image_shape" in data else tuple(data.get("shape", (0, 0)))

    return {"kpt": kpt, "desc": desc, "score": score, "shape": shape}


def _normalise_stem(value: str) -> str:
    path = Path(value)
    return path.stem if path.suffix else str(value)


def ensure_features(
    img_dir: str,
    out_dir: str,
    allowed_stems: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load all feature files from the output directory.
    Matches features to images in img_dir based on filenames.
    """
    features_dir = ensure_dir(Path(out_dir) / "features")
    if allowed_stems is not None:
        image_stems = {_normalise_stem(stem) for stem in allowed_stems}
    else:
        image_stems = {stem_from_path(p) for p in list_image_paths(img_dir)}

    features: Dict[str, Dict[str, np.ndarray]] = {}
    for npz_path in iter_feature_paths(features_dir):
        stem = stem_from_path(npz_path)
        if stem not in image_stems:
            continue
        # Load generic features
        features[stem] = load_features(str(npz_path))

    missing = image_stems - set(features.keys())
    if missing:
        missing_list = ", ".join(sorted(missing))
        # Warning instead of error might be better, but strict for now
        raise FileNotFoundError(
            f"Missing features (e.g., .npz) for {len(missing)} images: {missing_list}"
        )

    return features