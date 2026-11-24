from __future__ import annotations

from dataclasses import dataclass, field
from threading import Event, Lock

import numpy as np
from scipy.optimize import LinearConstraint, minimize
from scipy.spatial import cKDTree
from scipy.stats import rankdata
import shapely
from shapely.geometry import MultiPoint, box
from shapely.geometry.polygon import orient
from shapely.ops import polylabel


EDGE_MODES = ("auto", "translation", "isotropic", "border")


@dataclass
class PolygonTranslationGrid:
    """Fast 2D FFT set-covariance lookup + Ohser (1983) angular factor for arbitrary polygons."""
    x0: float
    y0: float
    hx: float
    hy: float
    grid_size: int
    area: float
    min_cov: float
    cov: np.ndarray
    rho_grid: np.ndarray
    psi_grid: np.ndarray
    max_supported_r: float

    @classmethod
    def build(cls, geometry, grid_size: int = 256, min_overlap_ratio: float = 0.02) -> "PolygonTranslationGrid":
        x0, y0, x1, y1 = geometry.bounds
        lx = max(float(x1 - x0), 1e-9)
        ly = max(float(y1 - y0), 1e-9)
        hx, hy = lx / grid_size, ly / grid_size
        xs = x0 + (np.arange(grid_size, dtype=float) + 0.5) * hx
        ys = y0 + (np.arange(grid_size, dtype=float) + 0.5) * hy
        xx, yy = np.meshgrid(xs, ys)
        mask = shapely.contains_xy(geometry, xx.ravel(), yy.ravel()).reshape(grid_size, grid_size).astype(float)
        if mask.sum() < 16:
            cells = shapely.box(
                xx.ravel() - 0.5 * hx, yy.ravel() - 0.5 * hy,
                xx.ravel() + 0.5 * hx, yy.ravel() + 0.5 * hy,
            )
            mask = shapely.intersects(geometry, cells).reshape(grid_size, grid_size).astype(float)
        area = float(geometry.area)
        cell_area = area / max(1.0, float(mask.sum()))
        fft_n = 2 * grid_size
        spec = np.fft.rfft2(mask, s=(fft_n, fft_n))
        acorr = np.fft.irfft2(spec * np.conj(spec), s=(fft_n, fft_n))
        cov = np.maximum(0.0, np.fft.fftshift(acorr) * cell_area)
        min_cov = min_overlap_ratio * area

        diag = float(np.hypot(lx, ly))
        rho_grid = np.linspace(0.0, diag, 512)
        angles = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
        cos_t, sin_t = np.cos(angles)[None, :], np.sin(angles)[None, :]
        dx_mat = rho_grid[:, None] * cos_t
        dy_mat = rho_grid[:, None] * sin_t

        inst = cls(x0, y0, hx, hy, grid_size, area, min_cov, cov, rho_grid, np.ones_like(rho_grid), diag)
        cov_polar = inst.lookup_cov(dx_mat.ravel(), dy_mat.ravel()).reshape(len(rho_grid), len(angles))
        psi_grid = (cov_polar >= min_cov).mean(axis=1)
        psi_grid[0] = 1.0
        supported_idx = np.flatnonzero(psi_grid >= 0.05)
        max_r = float(rho_grid[supported_idx[-1]]) if len(supported_idx) else 0.0
        inst.psi_grid = psi_grid
        inst.max_supported_r = max_r
        return inst

    def lookup_cov(self, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
        g = self.grid_size
        ux = g + np.asarray(dx, dtype=float) / self.hx
        uy = g + np.asarray(dy, dtype=float) / self.hy
        ix = np.floor(ux).astype(int)
        iy = np.floor(uy).astype(int)
        limit = 2 * g - 2
        inside_box = (ix >= 0) & (ix <= limit) & (iy >= 0) & (iy <= limit)
        out = np.zeros(len(ux), dtype=float)
        if not inside_box.any():
            return out
        ix_v, iy_v = ix[inside_box], iy[inside_box]
        fx = ux[inside_box] - ix_v
        fy = uy[inside_box] - iy_v
        c00 = self.cov[iy_v, ix_v]
        c10 = self.cov[iy_v, ix_v + 1]
        c01 = self.cov[iy_v + 1, ix_v]
        c11 = self.cov[iy_v + 1, ix_v + 1]
        out[inside_box] = (
            (1.0 - fx) * (1.0 - fy) * c00
            + fx * (1.0 - fy) * c10
            + (1.0 - fx) * fy * c01
            + fx * fy * c11
        )
        return out

    def pair_weights(self, dx: np.ndarray, dy: np.ndarray, dist: np.ndarray) -> np.ndarray:
        c = self.lookup_cov(dx, dy)
        psi = np.interp(dist, self.rho_grid, self.psi_grid, left=1.0, right=0.0)
        valid = (c >= self.min_cov) & (psi >= 0.05)
        w = np.zeros(len(dist), dtype=float)
        w[valid] = self.area / (c[valid] * psi[valid])
        return w


@dataclass(frozen=True)
class Window:
    geometry: object
    mode: str
    edge_mode: str = "auto"
    _cache: dict = field(default_factory=dict, repr=False, compare=False, hash=False)
    _lock: Lock = field(default_factory=Lock, repr=False, compare=False, hash=False)

    @property
    def area(self):
        return self.geometry.area

    @property
    def is_rectangle(self):
        return self.geometry.equals(box(*self.geometry.bounds))

    def effective_correction(self, override: str | None = None) -> str:
        chosen = override or self.edge_mode or "auto"
        if chosen == "auto":
            return "translation" if self.is_rectangle else "border"
        return chosen

    @property
    def correction(self):
        return self.effective_correction()

    def translation_grid(self) -> PolygonTranslationGrid:
        grid = self._cache.get("trans_grid")
        if grid is None:
            with self._lock:
                grid = self._cache.get("trans_grid")
                if grid is None:
                    grid = PolygonTranslationGrid.build(self.geometry)
                    self._cache["trans_grid"] = grid
        return grid

    @property
    def default_rmax(self):
        x0, y0, x1, y1 = self.geometry.bounds
        limit = min(x1 - x0, y1 - y0) / 4
        if self.is_rectangle or self.correction in ("translation", "isotropic"):
            return limit
        parts = [self.geometry] if self.geometry.geom_type == "Polygon" else self.geometry.geoms
        clearance = max(polylabel(p, tolerance=limit / 20).distance(p.boundary) for p in parts)
        return min(limit, clearance * 0.65)


def _convex_rectangle(polygon):
    # A rectangle with center (x, y) and half-sides (w, h) is in every hull
    # half-plane n·p <= b exactly when nx*x + ny*y + |nx|*w + |ny|*h <= b.
    vertices = np.asarray(orient(polygon).exterior.coords)
    edges = np.diff(vertices, axis=0)
    normals = np.column_stack((edges[:, 1], -edges[:, 0]))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    offsets = (normals * vertices[:-1]).sum(axis=1)
    constraints = np.column_stack((normals, np.abs(normals)))
    center = np.asarray(polygon.centroid.coords[0])
    slack = offsets - normals @ center
    half_size = max(float(np.min(slack / np.abs(normals).sum(axis=1))) / 2, 1e-15)
    result = minimize(
        lambda p: -np.log(p[2:]).sum(),
        [*center, half_size, half_size],
        jac=lambda p: np.array([0., 0., -1 / p[2], -1 / p[3]]),
        method="SLSQP",
        bounds=[(0, 1), (0, 1), (1e-15, .5), (1e-15, .5)],
        constraints=LinearConstraint(constraints, -np.inf, offsets),
        options={"ftol": 1e-12},
    )
    if result.success:
        return np.r_[result.x[:2] - result.x[2:], result.x[:2] + result.x[2:]]
    return None


def _grid_rectangle(polygon, stop):
    # Search all rectangles made from fully covered cells. A histogram stack
    # visits each cell once after classification, including across narrow notches.
    size = 192
    grid = np.linspace(0, 1, size + 1)
    x, y = np.meshgrid(grid[:-1], grid[:-1])
    x1, y1 = np.meshgrid(grid[1:], grid[1:])
    cells = shapely.box(x.ravel(), y.ravel(), x1.ravel(), y1.ravel())
    covered = shapely.covers(polygon, cells).reshape(size, size)
    heights = np.zeros(size, dtype=int)
    best, best_area = None, 0
    for row, mask in enumerate(covered):
        check_cancel(stop)
        heights = (heights + 1) * mask
        stack = []
        for column, height in enumerate([*heights, 0]):
            left = column
            while stack and stack[-1][1] > height:
                left, top_height = stack.pop()
                area = top_height * (column - left)
                if area > best_area:
                    best_area = area
                    best = np.array([left, row + 1 - top_height, column, row + 1]) / size
            if not stack or stack[-1][1] < height:
                stack.append((left, height))
    if best is None:
        point = polylabel(polygon, tolerance=1 / (size * 8))
        center = np.asarray(point.coords[0])
        half = point.distance(polygon.boundary) / 2
        best = np.r_[center - half, center + half]

    # Refine grid-aligned sides against the original polygon. Reverse the
    # expansion order as well, since a sloping edge can trade width for height.
    candidates = [best]
    for sides in ((0, 2, 1, 3), (1, 3, 0, 2), (2, 0, 3, 1), (3, 1, 2, 0)):
        bounds = best.copy()
        for side in sides:
            check_cancel(stop)
            good, outside = bounds[side], float(side >= 2)
            for _ in range(32):
                bounds[side] = (good + outside) / 2
                if polygon.covers(box(*bounds)):
                    good = bounds[side]
                else:
                    outside = bounds[side]
            bounds[side] = good
        candidates.append(bounds)
    return max(candidates, key=lambda b: np.prod(b[2:] - b[:2]))


def largest_inner_rectangle(polygon, stop=None):
    """Fit an axis-aligned rectangle. Concave fits use a grid and side refinement."""
    check_cancel(stop)
    outer = box(*polygon.bounds)
    if polygon.equals(outer):
        return outer
    origin = np.array(polygon.bounds[:2])
    scale = np.array(polygon.bounds[2:]) - origin
    normalized = shapely.transform(polygon, lambda xy: (xy - origin) / scale)
    shapely.prepare(normalized)
    convex = polygon.equals(polygon.convex_hull)

    def covered_rectangle(bounds):
        if bounds is None:
            return None
        center = (bounds[:2] + bounds[2:]) / 2
        for shrink in (0, 1e-12, 1e-10, 1e-8, 1e-6):
            b = (bounds - np.tile(center, 2)) * (1 - shrink) + np.tile(center, 2)
            rect = box(*(b * np.tile(scale, 2) + np.tile(origin, 2)))
            if rect.area > 0 and polygon.covers(rect):
                return rect
        return None

    if convex:
        result = covered_rectangle(_convex_rectangle(normalized))
        if result is not None:
            return result
    result = covered_rectangle(_grid_rectangle(normalized, stop))
    if result is None:
        raise ValueError("invalid_roi")
    return result


def valid_bounds_box(bounds):
    x0, y0, x1, y1 = bounds
    if not np.all(np.isfinite(bounds)) or x1 <= x0 or y1 <= y0:
        raise ValueError("invalid_rectangle")
    return box(x0, y0, x1, y1)


def make_window(mode, points, bounds=None, geometry=None, hull_ratio=0.35, edge_mode="auto",
                *, rectangle_hull="convex", stop=None):
    if mode == "custom":
        if geometry is None and bounds is not None:
            geometry = valid_bounds_box(bounds)
        if geometry is None:
            raise ValueError("missing_roi")
    elif mode in ("infer", "locked", "rectangle"):
        cloud = MultiPoint(points)
        convex = mode == "locked" or (mode == "rectangle" and rectangle_hull == "convex")
        geometry = (
            cloud.convex_hull
            if convex or hull_ratio >= 0.999
            else shapely.concave_hull(cloud, ratio=hull_ratio)
        )
    else:
        raise ValueError("invalid_roi_mode")
    if (
        geometry.geom_type not in ("Polygon", "MultiPolygon")
        or not geometry.is_valid
        or not np.isfinite(geometry.area)
        or geometry.area <= 0
    ):
        raise ValueError("invalid_roi")
    if mode == "rectangle":
        geometry = largest_inner_rectangle(geometry, stop)
    shapely.prepare(geometry)
    return Window(geometry, mode, edge_mode=edge_mode)


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
        shapely.prepare(polygon)
        if remaining / acceptance > 20_000_000:
            rotated = polygon.minimum_rotated_rectangle
            corners = np.asarray(rotated.exterior.coords)[:4]
            u, v = corners[1] - corners[0], corners[3] - corners[0]
            acceptance = polygon.area / rotated.area
            if remaining / acceptance > 20_000_000:
                raise ValueError("roi_sampling_limit")
            draw = lambda n, c0=corners[0], uu=u, vv=v: c0 + rng.random((n, 1)) * uu + rng.random((n, 1)) * vv
        else:
            draw = lambda n, a0=x0, b0=y0, a1=x1, b1=y1: rng.uniform((a0, b0), (a1, b1), size=(n, 2))
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


_ISO_ANGLES = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
_ISO_COS = np.cos(_ISO_ANGLES)[None, :]
_ISO_SIN = np.sin(_ISO_ANGLES)[None, :]


def _isotropic_pair_weights(geometry, focal_xy: np.ndarray, dist: np.ndarray, clearance_focal: np.ndarray) -> np.ndarray:
    """Ripley circumferential edge weights 2*pi*d / |dB(x_i, d) cap W|, exact 1.0 when clearance >= d."""
    weights = np.ones(len(dist), dtype=float)
    need = (dist > clearance_focal) & (dist > 0)
    if not need.any():
        return weights
    sub_xy = focal_xy[need]
    sub_d = dist[need, None]
    # Sample 64 points around each boundary-intersecting circle in chunks
    n_need = len(sub_d)
    sub_w = np.zeros(n_need, dtype=float)
    chunk = 4096
    for s in range(0, n_need, chunk):
        e = min(n_need, s + chunk)
        px = sub_xy[s:e, 0:1] + sub_d[s:e] * _ISO_COS
        py = sub_xy[s:e, 1:2] + sub_d[s:e] * _ISO_SIN
        inside_flags = shapely.contains_xy(geometry, px.ravel(), py.ravel()).reshape(e - s, len(_ISO_ANGLES))
        frac = inside_flags.mean(axis=1)
        ok = frac >= 0.02
        sub_w[s:e][ok] = 1.0 / frac[ok]
    weights[need] = sub_w
    return weights


def ripley(points, window, radii, targets=None, intensity=None, stop=None, correction=None):
    """Ordered distinct pairs with translation (exact rectangle or 2D-FFT polygon), isotropic, or border correction."""
    xy = np.asarray(points, dtype=float)
    cross = targets is not None
    other = np.asarray(targets, dtype=float) if cross else xy
    r = np.asarray(radii, dtype=float)
    if (
        r.ndim != 1
        or not r.size
        or not np.all(np.isfinite(r))
        or r[0] <= 0
        or np.any(np.diff(r) <= 0)
    ):
        raise ValueError("invalid_radii")
    if len(xy) == 0 or len(other) < (1 if cross else 2):
        return np.full(len(r), np.nan)
    if intensity is not None and (not window.is_rectangle or cross):
        raise ValueError("intensity_rectangle_only")
    check_cancel(stop)

    eff_corr = window.effective_correction(correction)
    area = window.area
    tree = cKDTree(other)
    rect_translation = window.is_rectangle and eff_corr == "translation"

    if rect_translation:
        x0, y0, x1, y1 = window.geometry.bounds
        size = np.array([x1 - x0, y1 - y0])
        if intensity is not None or (window.edge_mode == "auto" and correction is None):
            # Exact analytical rectangle translation up to shorter side
            if r[0] >= size.min():
                return np.full(len(r), np.nan)
            if r[-1] >= size.min():
                valid_len = int(np.searchsorted(r, size.min(), side="left"))
                out = np.full(len(r), np.nan)
                out[:valid_len] = ripley(xy, window, r[:valid_len], targets, intensity, stop, correction)
                return out
            density = intensity.density(xy, window) if intensity is not None else None
            starts = np.zeros(len(r) + 1)
            block = max(1, min(512, 400_000 // len(other)))
            for start in range(0, len(xy), block):
                check_cancel(stop)
                query = xy[start:start + block]
                pairs = cKDTree(query).sparse_distance_matrix(tree, r[-1], output_type="ndarray")
                focal, neighbor, distance = pairs["i"] + start, pairs["j"], pairs["v"]
                keep = np.ones(len(focal), dtype=bool) if cross else focal != neighbor
                focal, neighbor, distance = focal[keep], neighbor[keep], distance[keep]
                first = np.searchsorted(r, distance, side="left")
                overlap = np.prod(size - np.abs(xy[focal] - other[neighbor]), axis=1)
                weights = area / overlap
                if density is not None:
                    weights /= (area * density[focal]) * (area * density[neighbor])
                starts += np.bincount(first, weights=weights, minlength=len(starts))
            numerator = area * np.cumsum(starts[:-1])
            denominator = len(xy) * (len(other) - (0 if cross else 1))
            return np.divide(numerator, float(denominator), out=np.full(len(r), np.nan), where=denominator > 0)

        # For larger radii explicit translation falls through to the angular factor.
        short_side = float(size.min())
        if r[-1] < short_side:
            starts = np.zeros(len(r) + 1)
            block = max(1, min(512, 400_000 // len(other)))
            for start in range(0, len(xy), block):
                check_cancel(stop)
                query = xy[start:start + block]
                pairs = cKDTree(query).sparse_distance_matrix(tree, r[-1], output_type="ndarray")
                focal, neighbor, distance = pairs["i"] + start, pairs["j"], pairs["v"]
                keep = np.ones(len(focal), dtype=bool) if cross else focal != neighbor
                focal, neighbor, distance = focal[keep], neighbor[keep], distance[keep]
                first = np.searchsorted(r, distance, side="left")
                overlap = np.prod(size - np.abs(xy[focal] - other[neighbor]), axis=1)
                starts += np.bincount(first, weights=area / overlap, minlength=len(starts))
            numerator = area * np.cumsum(starts[:-1])
            denominator = len(xy) * (len(other) - (0 if cross else 1))
            return numerator / float(denominator)

    if eff_corr == "translation":
        tgrid = window.translation_grid()
        starts = np.zeros(len(r) + 1)
        max_q = min(float(r[-1]), tgrid.max_supported_r)
        if max_q >= r[0]:
            block = max(1, min(512, 400_000 // len(other)))
            for start in range(0, len(xy), block):
                check_cancel(stop)
                query = xy[start:start + block]
                pairs = cKDTree(query).sparse_distance_matrix(tree, max_q, output_type="ndarray")
                focal, neighbor, distance = pairs["i"] + start, pairs["j"], pairs["v"]
                keep = np.ones(len(focal), dtype=bool) if cross else focal != neighbor
                focal, neighbor, distance = focal[keep], neighbor[keep], distance[keep]
                if not len(focal):
                    continue
                diff_xy = other[neighbor] - xy[focal]
                weights = tgrid.pair_weights(diff_xy[:, 0], diff_xy[:, 1], distance)
                first = np.searchsorted(r, distance, side="left")
                starts += np.bincount(first, weights=weights, minlength=len(starts))
        numerator = area * np.cumsum(starts[:-1])
        denom = float(len(xy) * (len(other) - (0 if cross else 1)))
        out = np.divide(numerator, denom, out=np.full(len(r), np.nan), where=denom > 0)
        out[r > tgrid.max_supported_r] = np.nan
        return out

    if eff_corr == "isotropic":
        x0, y0, x1, y1 = window.geometry.bounds
        max_iso_r = 0.5 * float(np.hypot(x1 - x0, y1 - y0))
        clearance = np.asarray(shapely.distance(shapely.points(xy), window.geometry.boundary), dtype=float)
        starts = np.zeros(len(r) + 1)
        max_q = min(float(r[-1]), max_iso_r)
        if max_q >= r[0]:
            block = max(1, min(512, 400_000 // len(other)))
            for start in range(0, len(xy), block):
                check_cancel(stop)
                query = xy[start:start + block]
                pairs = cKDTree(query).sparse_distance_matrix(tree, max_q, output_type="ndarray")
                focal, neighbor, distance = pairs["i"] + start, pairs["j"], pairs["v"]
                keep = np.ones(len(focal), dtype=bool) if cross else focal != neighbor
                focal, neighbor, distance = focal[keep], neighbor[keep], distance[keep]
                if not len(focal):
                    continue
                weights = _isotropic_pair_weights(window.geometry, xy[focal], distance, clearance[focal])
                first = np.searchsorted(r, distance, side="left")
                starts += np.bincount(first, weights=weights, minlength=len(starts))
        numerator = area * np.cumsum(starts[:-1])
        denom = float(len(xy) * (len(other) - (0 if cross else 1)))
        out = np.divide(numerator, denom, out=np.full(len(r), np.nan), where=denom > 0)
        out[r > max_iso_r] = np.nan
        return out

    # Border (reduced-sample) correction
    clearance = np.asarray(shapely.distance(shapely.points(xy), window.geometry.boundary))
    starts = np.zeros(len(r) + 1)
    block = max(1, min(512, 400_000 // len(other)))
    for start in range(0, len(xy), block):
        check_cancel(stop)
        query = xy[start:start + block]
        pairs = cKDTree(query).sparse_distance_matrix(tree, r[-1], output_type="ndarray")
        focal, neighbor, distance = pairs["i"] + start, pairs["j"], pairs["v"]
        keep = np.ones(len(focal), dtype=bool) if cross else focal != neighbor
        focal, neighbor, distance = focal[keep], neighbor[keep], distance[keep]
        first = np.searchsorted(r, distance, side="left")
        last = np.searchsorted(r, clearance[focal], side="right")
        valid = first < last
        starts += np.bincount(first[valid], minlength=len(starts))
        starts -= np.bincount(last[valid], minlength=len(starts))
    numerator = area * np.cumsum(starts[:-1])
    eligible = len(xy) - np.searchsorted(np.sort(clearance), r, side="left")
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


def global_envelope(curves, alpha=0.05, alternative="two-sided"):
    """Extreme rank length global envelope (two-sided or upper one-sided). Keep all curves tied at the cutoff."""
    curves = np.asarray(curves, dtype=float)
    total, nr = curves.shape
    supported = np.isfinite(curves).all(axis=0)
    lo, hi = np.full(nr, np.nan), np.full(nr, np.nan)
    side = np.zeros(nr, dtype=np.int8)
    if not supported.any() or total < 2:
        return Envelope(lo, hi, np.nan, "insufficient", side)
    observed = curves[:, supported]
    rank = rankdata(observed, axis=0, method="average")
    if alternative == "greater":
        scores = np.sort(total + 1 - rank, axis=1)
    else:
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
        if alternative != "greater":
            side[curves[0] < lo] = -1
    if alternative == "greater":
        status = "different" if p <= alpha else "compatible"
    else:
        status = (
            "mixed" if np.any(side > 0) and np.any(side < 0)
            else "cluster" if np.any(side > 0)
            else "regular" if np.any(side < 0)
            else "different" if p <= alpha
            else "compatible"
        )
    return Envelope(lo, hi, p, status, side)


def local_neighbors(points, window, radius, stop=None):
    """Observed minus expected neighbor count at one physical radius under uniformity."""
    result = np.full(len(points), np.nan)
    if len(points) < 2:
        return result
    tree = cKDTree(points)
    full_disk = np.pi * radius ** 2
    for start in range(0, len(points), 1024):
        check_cancel(stop)
        xy = points[start:start + 1024]
        pts = shapely.points(xy)
        clearance = np.asarray(shapely.distance(pts, window.geometry.boundary), dtype=float)
        area = np.full(len(xy), full_disk, dtype=float)
        near_boundary = clearance < radius
        if near_boundary.any():
            circles = shapely.buffer(pts[near_boundary], radius, quad_segs=32)
            area[near_boundary] = shapely.area(shapely.intersection(circles, window.geometry))
        observed = tree.query_ball_point(xy, radius, return_length=True) - 1
        result[start:start + len(xy)] = observed - (len(points) - 1) * area / window.area
    return result
