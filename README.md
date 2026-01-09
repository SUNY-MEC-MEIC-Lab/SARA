# SARA: Select And Retain for Adjustment

SARA is a graph-based pair selection strategy for Structure-from-Motion (SfM). It efficiently selects optimal image pairs by evaluating geometric overlap and parallax using a lightweight "mini-RANSAC" and GPU-accelerated batch processing.

## Installation

Clone this repository.

```bash
git clone https://github.com/SUNY-MEC-MEIC-Lab/SARA.git
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

**Requirements:**

- Python 3.8+
- PyTorch (with CUDA support recommended)
- OpenCV
- Numpy, Pandas, Scipy, TIMM

## Usage

### 1. Prepare Local Features

SARA is designed to work with **Deep Learned Detectors** (e.g., ALIKED, SuperPoint, DISK, etc.).
Before running SARA, you must extract features for your images and save them as `.npz` files in the standard format described below.

**Directory Structure:**

```
/path/to/project/
├── images/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── output/
    └── features/   <-- Create this manually or via your extraction script
        ├── img1.npz
        ├── img2.npz
        └── ...
```

**NPZ Format:**
Each `.npz` file must correspond to an image stem (e.g., `img1.jpg` -> `img1.npz`) and contain the following keys:

- `keypoints`: `(N, 2)` float32 array
- `descriptors`: `(N, D)` float32 array
- `scores`: `(N,)` float32 array (detection scores)
- `image_shape`: `(H, W)` tuple (optional, but recommended)

*Note: SARA computes descriptors similarity based on the metric specified in arguments (default: cosine similarity, suitable for normalized deep descriptors).*

### 2. Run SARA

Run the `run_sara.py` script:

```bash
python run_sara.py --img_dir /path/to/project/images --out_dir /path/to/project/output
```

### Arguments

- `--img_dir`: Path to the directory containing source images.
- `--out_dir`: Output directory. SARA will look for features in `{out_dir}/features` and save results here.
- `--knn_k`: Number of initial candidates to retrieve via global retrieval (default: 100).
- `--device`: Device to use (`cuda` or `cpu`).

**Tuning Thresholds:**

- `--tau_overlap`: Minimum overlap score (default: 0.10).
- `--tau_parallax`: Minimum parallax angle score (default: 0.05).
- `--descriptor_metric`: Metric for feature comparison (`cosine` or `l2`). Use `cosine` for most deep features (ALIKED, SuperPoint).

### Output

SARA produces two main files in the `out_dir`:

1. `pairs.csv`: Detailed list of selected pairs with scores (indices, score, overlap, parallax).
2. `pairs_for_matcher.jsonl`: Simplified list of pairs formatted for feature matchers.
3. `sara_matches.h5`: (Optional) HDF5 file containing the sparse matches found during the SARA verification step.
