from __future__ import annotations

import math


def overlap_score(inliers: int, total: int) -> float:
    total = max(1, total)
    frac = max(0.0, min(1.0, inliers / total))
    return float(frac)


def edge_score(
    overlap: float,
    parallax: float,
    alpha: float,
    beta: float,
    tau_overlap: float,
    tau_parallax: float,
    scoring_mode: str = "combined",
) -> float:
    if overlap < tau_overlap or parallax < tau_parallax:
        return 0.0

    if scoring_mode == "overlap_only":
        score = overlap ** alpha
    elif scoring_mode == "parallax_only":
        score = parallax ** beta
    else:  # "combined"
        score = (overlap ** alpha) * (parallax ** beta)

    return float(score)
