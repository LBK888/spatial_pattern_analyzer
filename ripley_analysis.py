from __future__ import annotations

from dataclasses import dataclass
from threading import Event

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata
import shapely
from shapely.geometry import MultiPoint, box
from shapely.ops import polylabel


@dataclass(frozen=True)
class Window:
    geometry: object
    mode: str

    @property
    def area(self):
        return self.geometry.area

    @property
    def is_rectangle(self):
        return self.geometry.equals(box(*self.geometry.bounds))

    @property
    def correction(self):
        return "translation" if self.is_rectangle else "border"

    @property
    def default_rmax(self):
        x0, y0, x1, y1 = self.geometry.bounds
        limit = min(x1 - x0, y1 - y0) / 4
        if self.is_rectangle:
            return limit
        parts = [self.geometry] if self.geometry.geom_type == "Polygon" else self.geometry.geoms
        clearance = max(polylabel(p, tolerance=limit / 20).distance(p.boundary) for p in parts)
        return min(limit, clearance * 0.65)


def make_window(mode, points, bounds=None, geometry=None, hull_ratio=0.35):
    if mode == "rectangle":
        if bounds is None:
            raise ValueError("missing_rectangle")
        x0, y0, x1, y1 = bounds
        if not np.all(np.isfinite(bounds)) or x1 <= x0 or y1 <= y0:
            raise ValueError("invalid_rectangle")
        geometry = box(x0, y0, x1, y1)
    elif mode == "custom":
        if geometry is None:
            raise ValueError("missing_roi")
    elif mode in ("infer", "locked"):
        cloud = MultiPoint(points)
        geometry = shapely.concave_hull(cloud, ratio=hull_ratio) if mode == "infer" else cloud.convex_hull
    else:
        raise ValueError("invalid_roi_mode")
    if (geometry.geom_type not in ("Polygon", "MultiPolygon") or not geometry.is_valid
            or not np.isfinite(geometry.area) or geometry.area <= 0):
        raise ValueError("invalid_roi")
    return Window(geometry, mode)


def inside(window, xy):
    return np.asarray(shapely.covers(window.geometry, shapely.points(xy)), dtype=bool)


def boundary_points(window, xy):
    return np.asarray(shapely.intersects(window.geometry.boundary, shapely.points(xy)), dtype=bool)


class Cancelled(Exception):
    pass


def check_cancel(stop: Event | None):
    if stop is not None and stop.is_set():
        raise Cancelled()


def uniform(window, count, rng, stop=None):
    """Sample by area, including disjoint components and excluding holes."""
    if count == 0:
        return np.empty((0, 2))
    parts = [window.geometry] if window.geometry.geom_type == "Polygon" else list(window.geometry.geoms)
    counts = rng.multinomial(count, np.asarray([p.area for p in parts]) / window.area)
    result = []
    for polygon, remaining in zip(parts, counts):
        x0, y0, x1, y1 = polygon.bounds
        acceptance = polygon.area / ((x1 - x0) * (y1 - y0))
        if not remaining:
            continue
        if remaining / acceptance > 20_000_000:
            # Partition a thin/rotated polygon before rejection sampling.
            rotated = polygon.minimum_rotated_rectangle
            corners = np.asarray(rotated.exterior.coords)[:4]
            u, v = corners[1] - corners[0], corners[3] - corners[0]
            acceptance = polygon.area / rotated.area
            if remaining / acceptance > 20_000_000:
                raise ValueError("roi_sampling_limit")
            draw = lambda n: corners[0] + rng.random((n, 1)) * u + rng.random((n, 1)) * v
        else:
            draw = lambda n: rng.uniform((x0, y0), (x1, y1), size=(n, 2))
        if polygon.equals(box(x0, y0, x1, y1)):
            result.append(draw(remaining))
            continue
        attempts = 0
        while remaining:
            check_cancel(stop)
            size = min(65536, max(32, int(1.15 * remaining / acceptance) + 1))
            attempts += size
            if attempts > 30_000_000:
                raise ValueError("roi_sampling_limit")
            candidates = draw(size)
            keep = candidates[shapely.contains_xy(polygon, candidates[:, 0], candidates[:, 1])][:remaining]
            result.append(keep)
            remaining -= len(keep)
    return np.concatenate(result)


@dataclass(frozen=True)
class IntensityGrid:
    """Externally supplied, piecewise-constant relative intensity on a rectangle."""
    weights: np.ndarray

    def __post_init__(self):
        w = np.asarray(self.weights, dtype=float)
        if w.ndim != 2 or not w.size or not np.all(np.isfinite(w) & (w > 0)):
            raise ValueError("invalid_intensity")
        w = w / w.max()
        if np.any(w == 0):
            raise ValueError("invalid_intensity")
        object.__setattr__(self, "weights", w / w.sum())

    def density(self, xy, window):
        x0, y0, x1, y1 = window.geometry.bounds
        ny, nx = self.weights.shape
        indexes = np.floor((xy - [x0, y0]) / [x1 - x0, y1 - y0] * [nx, ny]).astype(int)
        col, row = np.clip(indexes, 0, [nx - 1, ny - 1]).T
        return self.weights[row, col] * self.weights.size / window.area

    def sample(self, count, window, rng):
        x0, y0, x1, y1 = window.geometry.bounds
        ny, nx = self.weights.shape
        row, col = np.unravel_index(rng.choice(self.weights.size, count, p=self.weights.ravel()), (ny, nx))
        return (np.column_stack((col, row)) + rng.random((count, 2))) / [nx, ny] * [x1 - x0, y1 - y0] + [x0, y0]


def ripley(points, window, radii, targets=None, intensity=None, stop=None):
    """Ordered distinct pairs. Border correction uses only eligible focal points."""
    xy = np.asarray(points, dtype=float)
    cross = targets is not None
    other = np.asarray(targets, dtype=float) if cross else xy
    r = np.asarray(radii, dtype=float)
    if (r.ndim != 1 or not r.size or not np.all(np.isfinite(r)) or r[0] <= 0
            or np.any(np.diff(r) <= 0)):
        raise ValueError("invalid_radii")
    if len(xy) == 0 or len(other) < (1 if cross else 2):
        return np.full(len(r), np.nan)
    if intensity is not None and (not window.is_rectangle or cross):
        raise ValueError("intensity_rectangle_only")
    check_cancel(stop)
    area = window.area
    tree = cKDTree(other)
    translation = window.is_rectangle
    if translation:
        x0, y0, x1, y1 = window.geometry.bounds
        size = np.array([x1 - x0, y1 - y0])
        if r[-1] >= size.min():
            raise ValueError("invalid_radii")
        density = intensity.density(xy, window) if intensity is not None else None
    else:
        clearance = np.asarray(shapely.distance(shapely.points(xy), window.geometry.boundary))
    starts = np.zeros(len(r) + 1)
    # Cap the worst-case number of candidate pairs held in one block.
    block = max(1, min(512, 400_000 // len(other)))
    for start in range(0, len(xy), block):
        check_cancel(stop)
        query = xy[start:start + block]
        pairs = cKDTree(query).sparse_distance_matrix(tree, r[-1], output_type="ndarray")
        focal, neighbor, distance = pairs["i"] + start, pairs["j"], pairs["v"]
        keep = np.ones(len(focal), dtype=bool) if cross else focal != neighbor
        focal, neighbor, distance = focal[keep], neighbor[keep], distance[keep]
        first = np.searchsorted(r, distance, side="left")
        if translation:
            overlap = np.prod(size - np.abs(xy[focal] - other[neighbor]), axis=1)
            weights = area / overlap
            if density is not None:
                weights /= (area * density[focal]) * (area * density[neighbor])
            starts += np.bincount(first, weights=weights, minlength=len(starts))
        else:
            last = np.searchsorted(r, clearance[focal], side="right")
            valid = first < last
            starts += np.bincount(first[valid], minlength=len(starts))
            starts -= np.bincount(last[valid], minlength=len(starts))
    numerator = area * np.cumsum(starts[:-1])
    eligible = len(xy) if translation else len(xy) - np.searchsorted(np.sort(clearance), r, side="left")
    denominator = eligible * (len(other) - (0 if cross else 1))
    return np.divide(numerator, denominator, out=np.full(len(r), np.nan), where=denominator > 0)


def centered_l(k, radii):
    return np.sqrt(np.maximum(k, 0) / np.pi) - radii


@dataclass(frozen=True)
class Envelope:
    lower: np.ndarray
    upper: np.ndarray
    p: float
    status: str
    side: np.ndarray


def global_envelope(curves, alpha=0.05):
    """Two-sided extreme rank length. Keep all curves tied at the cutoff."""
    curves = np.asarray(curves, dtype=float)
    total, nr = curves.shape
    supported = np.isfinite(curves).all(axis=0)
    lo, hi = np.full(nr, np.nan), np.full(nr, np.nan)
    side = np.zeros(nr, dtype=np.int8)
    if not supported.any() or total < 2:
        return Envelope(lo, hi, np.nan, "insufficient", side)
    observed = curves[:, supported]
    rank = rankdata(observed, axis=0, method="average")
    scores = np.sort(np.minimum(rank, total + 1 - rank), axis=1)
    order = np.lexsort(scores.T[::-1])
    ordered_scores = scores[order]
    boundaries = np.r_[0, np.flatnonzero(np.any(np.diff(ordered_scores, axis=0) != 0, axis=1)) + 1, total]
    p_by_row = np.empty(total)
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        p_by_row[order[begin:end]] = end / total
    central = observed[p_by_row > alpha]
    lo[supported], hi[supported] = central.min(axis=0), central.max(axis=0)
    p = float(p_by_row[0])
    if p <= alpha:
        side[curves[0] > hi] = 1
        side[curves[0] < lo] = -1
    status = ("mixed" if np.any(side > 0) and np.any(side < 0) else
              "cluster" if np.any(side > 0) else "regular" if np.any(side < 0) else
              "different" if p <= alpha else "compatible")
    return Envelope(lo, hi, p, status, side)


def local_neighbors(points, window, radius, stop=None):
    """Observed minus expected neighbor count at one physical radius under uniformity."""
    result = np.full(len(points), np.nan)
    if len(points) < 2:
        return result
    tree = cKDTree(points)
    for start in range(0, len(points), 1024):
        check_cancel(stop)
        xy = points[start:start + 1024]
        circles = shapely.buffer(shapely.points(xy), radius, quad_segs=24)
        area = shapely.area(shapely.intersection(circles, window.geometry))
        observed = tree.query_ball_point(xy, radius, return_length=True) - 1
        result[start:start + len(xy)] = observed - (len(points) - 1) * area / window.area
    return result
