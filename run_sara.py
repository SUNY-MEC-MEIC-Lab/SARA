import argparse
import logging
import sys
from pathlib import Path

# Add current directory to path so we can import SARA
sys.path.insert(0, str(Path(__file__).resolve().parent))

from SARA.config import SARAConfig
from SARA.cli import run_SARA

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

def main():
    setup_logging()
    logger = logging.getLogger("run_sara")

    parser = argparse.ArgumentParser(
        description="Run SARA (Select And Retain for Adjustment) pair selection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Core paths
    parser.add_argument("--img_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for results and cache")

    # KNN and Candidates
    parser.add_argument("--knn_k", type=int, default=100, help="Number of KNN candidates per image")
    
    # Matching & RANSAC
    parser.add_argument("--top_t_mutual", type=int, default=128, help="Number of mutual nearest neighbors to keep")
    parser.add_argument("--min_nn_for_ransac", type=int, default=32, help="Minimum matches to attempt RANSAC")
    parser.add_argument("--ransac_iters", type=int, default=15, help="RANSAC iterations (mini-RANSAC)")
    parser.add_argument("--descriptor_metric", type=str, default="cosine", choices=["cosine", "l2"], help="Metric for descriptor comparison")

    # SARA Scoring Thresholds
    parser.add_argument("--tau_overlap", type=float, default=0.10, help="Overlap threshold")
    parser.add_argument("--tau_parallax", type=float, default=0.05, help="Parallax threshold")
    parser.add_argument("--alpha", type=float, default=1.0, help="Score weight alpha")
    parser.add_argument("--beta", type=float, default=1.0, help="Score weight beta")

    # Graph Construction
    parser.add_argument("--deg_cap", type=int, default=6, help="Degree cap for leaf augmentation")
    parser.add_argument("--graph_construction_mode", type=str, default="full", choices=["full", "mst_only", "mst_leaf"], help="Graph construction mode")
    
    # Advanced Augmentation
    parser.add_argument("--disable_multi_scale_loops", action="store_true", help="Disable multi-scale loop augmentation")
    parser.add_argument("--disable_long_baseline_anchors", action="store_true", help="Disable long-baseline anchors")
    parser.add_argument("--disable_weak_view_reinforcement", action="store_true", help="Disable weak-view reinforcement")
    
    # Intrinsics
    parser.add_argument("--no_intrinsics", action="store_true", help="Do not use intrinsics (assume roughly centered principal point)")
    parser.add_argument("--fx", type=float, default=None, help="Focal length x (if known/constant)")
    parser.add_argument("--fy", type=float, default=None, help="Focal length y (if known/constant)")

    # System
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--force_cpu_batch", action="store_true", help="Force CPU batch processing even if CUDA is available")
    
    args = parser.parse_args()

    # Map arguments to SARAConfig
    cfg = SARAConfig(
        img_dir=args.img_dir,
        out_dir=args.out_dir,
        knn_k=args.knn_k,
        top_t_mutual=args.top_t_mutual,
        min_nn_for_ransac=args.min_nn_for_ransac,
        ransac_iters=args.ransac_iters,
        descriptor_metric=args.descriptor_metric,
        tau_overlap=args.tau_overlap,
        tau_parallax=args.tau_parallax,
        alpha=args.alpha,
        beta=args.beta,
        deg_cap=args.deg_cap,
        graph_construction_mode=args.graph_construction_mode,
        enable_multi_scale_loops=not args.disable_multi_scale_loops,
        enable_long_baseline_anchors=not args.disable_long_baseline_anchors,
        enable_weak_view_reinforcement=not args.disable_weak_view_reinforcement,
        use_intrinsics=not args.no_intrinsics,
        fx=args.fx,
        fy=args.fy,
        device=args.device,
        use_gpu_batch=not args.force_cpu_batch,
    )

    logger.info("Starting SARA...")
    logger.info(f"Image Directory: {cfg.img_dir}")
    logger.info(f"Output Directory: {cfg.out_dir}")

    try:
        results = run_SARA(cfg)
        num_pairs = len(results["pairs"])
        logger.info(f"SARA completed successfully. Generated {num_pairs} pairs.")
        logger.info(f"Pairs saved to: {Path(cfg.out_dir) / cfg.pairs_filename}")
        
    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        logger.error("Ensure ALIKED features are extracted and placed in '{out_dir}/features/' as .npz files.")
        logger.error("Expected format per image: {stem}.npz containing 'keypoints', 'descriptors', 'scores'.")
        sys.exit(1)
    except Exception as e:
        logger.exception("An unexpected error occurred:")
        sys.exit(1)

if __name__ == "__main__":
    main()
