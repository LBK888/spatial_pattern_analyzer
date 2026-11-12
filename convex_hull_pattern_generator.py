"""Illustrative reference patterns and deterministic bundled demo generation."""
from pathlib import Path
import csv

import numpy as np

from ripley_analysis import check_cancel, inside, uniform


def clustered(window, n, rng, stop=None):
    """Compact Gaussian clusters, conditioned on remaining within the ROI."""
    centers = uniform(window, max(2, min(8, n // 12)), rng, stop)
    scale = window.default_rmax * .12
    accepted = []
    remaining = n
    while remaining:
        check_cancel(stop)
        candidates = centers[rng.integers(len(centers), size=remaining * 2)] + rng.normal(
            scale=scale, size=(remaining * 2, 2))
        kept = candidates[inside(window, candidates)][:remaining]
        accepted.extend(kept)
        remaining -= len(kept)
    return np.asarray(accepted).reshape(-1, 2)


def regular(window, n, rng, stop=None):
    """Sequential inhibition with explicit minimum distance."""
    minimum = .65 * np.sqrt(window.area / max(1, n))
    points = []
    for _ in range(30):
        check_cancel(stop)
        for candidate in uniform(window, max(64, n * 3), rng, stop):
            if not points or np.min(np.linalg.norm(np.asarray(points) - candidate, axis=1)) >= minimum:
                points.append(candidate)
                if len(points) == n:
                    return np.asarray(points)
    raise ValueError("regular_packing_failed")


def write_demo(path):
    """Three independent samples per group, two ROIs per sample, with actual bounds."""
    from ripley_analysis import make_window

    path = Path(path)
    rng = np.random.default_rng(2406)
    window = make_window("rectangle", [], (0, 0, 200, 160))
    with path.open("w", newline="", encoding="utf-8") as stream:
        out = csv.writer(stream)
        out.writerow(["sample_id", "roi_id", "group", "cell_type", "x", "y",
                      "xmin", "ymin", "xmax", "ymax"])
        for group, generator in (("Uniform", uniform), ("Clustered", clustered), ("Regular", regular)):
            for i in range(1, 4):
                for j in range(1, 3):
                    xy = generator(window, 72, rng)
                    marks = rng.permutation(["A"] * 36 + ["B"] * 36)
                    for mark, (x, y) in zip(marks, xy):
                        out.writerow([f"{group[0]}{i}", f"F{j}", group, mark, f"{x:.5f}", f"{y:.5f}",
                                      0, 0, 200, 160])
