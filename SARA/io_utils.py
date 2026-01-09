from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple, Any

import numpy as np
import pandas as pd


def list_image_paths(img_dir: str) -> List[Path]:
    img_path = Path(img_dir)
    if not img_path.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    images: List[Path] = []
    for pattern in patterns:
        images.extend(img_path.glob(pattern))
        images.extend(img_path.glob(pattern.upper()))
    return sorted(images)


def stem_from_path(path: Path) -> str:
    return path.stem


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_pairs_csv(
    pairs: Sequence[Tuple[str, str, float, float, float]],
    output_path: Path,
) -> None:
    ensure_dir(output_path.parent)
    df = pd.DataFrame(
        pairs, columns=["i", "j", "score", "overlap", "parallax"]
    )
    df.to_csv(output_path, index=False)


def save_pairs_for_matcher(
    pairs: Iterable[Tuple[str, str, float]],
    output_path: Path,
    suffix: str = ".jpg",
) -> None:
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as fh:
        for i, j, _ in pairs:
            fh.write(json.dumps({"image_i": f"{i}{suffix}", "image_j": f"{j}{suffix}"}) + "\n")


def normalise_feature_dict(
    raw_feats: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, Dict[str, np.ndarray]]:
    normalised: Dict[str, Dict[str, np.ndarray]] = {}
    for key, payload in raw_feats.items():
        path = Path(key)
        stem = stem_from_path(path) if path.suffix else key
        required = ("keypoints", "descriptors", "scores")
        if not all(k in payload for k in required):
            missing = [k for k in required if k not in payload]
            raise KeyError(f"Feature dict for {key} missing fields: {missing}")
        normalised[stem] = {
            "kpt": np.asarray(payload["keypoints"], dtype=np.float32),
            "desc": np.asarray(payload["descriptors"], dtype=np.float32),
            "score": np.asarray(payload["scores"], dtype=np.float32),
            "shape": tuple(payload.get("image_shape", payload.get("shape", (0, 0)))),
        }
    return normalised


def iter_feature_paths(features_dir: Path) -> Iterator[Path]:
    for path in sorted(features_dir.glob("*.npz")):
        yield path


def save_matches(matches: Dict[Tuple[str, str], Any], filepath: Path):
    import h5py
    
    ensure_dir(filepath.parent)
    with h5py.File(filepath, 'w') as f:
        for i, (pair, match_data) in enumerate(matches.items()):
            grp = f.create_group(f'match_{i}')
            grp.attrs['img1'] = pair[0]
            grp.attrs['img2'] = pair[1]
            
            for key, value in match_data.items():
                if isinstance(value, np.ndarray):
                    grp.create_dataset(key, data=value)
                else:
                    grp.attrs[key] = value