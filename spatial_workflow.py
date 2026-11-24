from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import re
from threading import Lock

import matplotlib
import numpy as np
import pandas as pd
import scipy
from scipy import stats as sp_stats
import shapely

from ripley_analysis import (
    EDGE_MODES,
    Envelope,
    boundary_points,
    centered_l,
    check_cancel,
    global_envelope,
    inside,
    local_neighbors,
    make_window,
    ripley,
    uniform,
)
from spatial_data import ISSUE_COLUMNS, issue_rows


VERSION = "2.0.0"
ALL = ""
MERGED_SEP = " + "


def task_rng(seed: int, *parts: str) -> np.random.Generator:
    """Order-independent deterministic RNG stream derived from (seed, *parts)."""
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(seed).encode("utf-8"))
    for part in parts:
        digest.update(b"\x00")
        digest.update(str(part).encode("utf-8"))
    words = np.frombuffer(digest.digest(), dtype=np.uint32)
    return np.random.default_rng(np.random.SeedSequence(words))


@dataclass(frozen=True)
class Settings:
    roi_mode: str = "infer"
    edge_correction: str = "auto"
    rectangle: tuple | None = None
    custom: dict = field(default_factory=dict)
    hull_ratio: float = .35
    r_min: float | None = None
    r_max: float | None = None
    n_radii: int = 60
    r_step: float | None = None
    simulations: int = 999
    seed: int = 2406
    workers: int = 0
    test: bool = True
    marks: str = "all"
    from_type: str = ""
    to_type: str = ""
    merged_sets: tuple[tuple[str, ...], ...] = ()
    merge_require_all: bool = True
    infer_per_label: bool = True
    intensity: object = None
    band: str = "sd"
    contrast: bool = False
    paired: bool = False
    pairing_mode: str = "auto"
    perm_stat: str = "welch"
    group_a: str = ""
    group_b: str = ""
    local_radius: float | None = None

    def validate(self):
        if (
            not 19 <= self.simulations <= 9999
            or not 0 <= self.seed < 2**64
            or not 0 <= self.hull_ratio <= 1
            or not 0 <= self.workers <= 64
            or self.edge_correction not in EDGE_MODES
            or self.perm_stat not in ("welch", "diff")
        ):
            raise ValueError("invalid_settings")
        if self.r_step is None and not 2 <= self.n_radii <= 2000:
            raise ValueError("invalid_settings")
        if self.r_min is not None and (not np.isfinite(self.r_min) or self.r_min < 0):
            raise ValueError("invalid_radii")
        positive = [d for d in (self.r_max, self.r_step, self.local_radius) if d is not None]
        if any(not np.isfinite(d) or d <= 0 for d in positive):
            raise ValueError("invalid_radii")
        if self.r_min is not None and self.r_max is not None and self.r_min >= self.r_max:
            raise ValueError("invalid_radii")
        if self.r_step is not None and self.r_max is not None:
            r0 = self.r_step if not self.r_min else self.r_min
            if r0 >= self.r_max:
                raise ValueError("invalid_radii")
            count = int(np.floor((self.r_max - r0) / self.r_step + 1e-9)) + 1
            if not 2 <= count <= 2000:
                raise ValueError("invalid_radii")
        if self.marks == "cross" and (not self.from_type or not self.to_type or self.from_type == self.to_type):
            raise ValueError("choose_two_types")
        if self.intensity is not None and (self.roi_mode != "rectangle" or self.marks in ("cross", "cross_all")):
            raise ValueError("intensity_rectangle_only")
        if self.local_radius is not None and (self.intensity is not None or self.marks in ("cross", "cross_all")):
            raise ValueError("local_uniform_only")


@dataclass
class Pattern:
    sample: str
    roi: str
    group: str
    label: str
    window: object
    xy: np.ndarray
    types: np.ndarray
    count: int
    target_count: int | None
    locked_count: int
    k: np.ndarray
    curve: np.ndarray
    envelope: object
    status: str
    null: str
    qc: list[str]
    p_adjusted: float = np.nan
    local: np.ndarray | None = None


@dataclass
class Summary:
    group: str
    label: str
    sample_ids: list[str]
    curves: np.ndarray
    mean: np.ndarray
    n: np.ndarray
    low: np.ndarray
    high: np.ndarray
    sd: np.ndarray | None = None
    se: np.ndarray | None = None
    series_id: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return (self.group, self.label)

    def format_name(self, all_text: str = "All") -> str:
        if self.series_id:
            return self.series_id
        grp = self.group
        lbl = self.label or all_text
        if grp and self.label:
            return f"{grp} · {lbl}"
        return grp or lbl


@dataclass
class Contrast:
    a: str
    b: str
    label: str
    effect: np.ndarray
    envelope: object
    n_a: int
    n_b: int
    p_adjusted: float = np.nan
    design: str = "independent"
    n_paired: int = 0
    mean_diff_scalar: float = np.nan
    parametric_test: str = ""
    parametric_stat: float = np.nan
    parametric_p: float = np.nan
    parametric_p_holm: float = np.nan
    nonparametric_test: str = ""
    nonparametric_stat: float = np.nan
    nonparametric_p: float = np.nan
    nonparametric_p_holm: float = np.nan


@dataclass
class OmnibusStat:
    series_names: list[str]
    design: str
    k_groups: int
    r_min: float
    r_max: float
    n_supported_radii: int
    statistic_curve: np.ndarray
    envelope: object
    p_global: float
    significant_spans: list
    parametric_test: str = ""
    parametric_stat: float = np.nan
    parametric_p: float = np.nan
    nonparametric_test: str = ""
    nonparametric_stat: float = np.nan
    nonparametric_p: float = np.nan


@dataclass
class GroupComparison:
    series: list[Summary]
    omnibus: OmnibusStat | None
    contrasts: list[Contrast]


@dataclass
class Results:
    data: object
    settings: Settings
    radii: np.ndarray
    patterns: list[Pattern]
    summaries: list[Summary]
    contrasts: list[Contrast]
    qc: pd.DataFrame
    source_roi: str = ""
    source_intensity: str = ""
    comparison: GroupComparison | None = None
    custom_summaries: list[Summary] = field(default_factory=list)


def sorted_types(values) -> list[str]:
    return sorted({str(v) for v in values if str(v) != ""}, key=lambda s: (s.casefold(), s))


def merged_label(types: tuple[str, ...] | list[str]) -> str:
    return MERGED_SEP.join(types)


def default_fixed_grid(data) -> tuple[float, float, float]:
    """Choose a clean default fixed (r_min, r_max, r_step) grid with 100 points for multi-sample data."""
    if data is None or data.points.empty:
        return (0.0, 250.0, 2.5)
    spans = []
    for (sample, roi), grp in data.points.groupby(["sample_id", "roi_id"], sort=False):
        if (sample, roi) in data.bounds:
            x0, y0, x1, y1 = data.bounds[(sample, roi)]
            spans.append(min(x1 - x0, y1 - y0) / 4.0)
        elif len(grp) >= 2:
            dx = float(grp.x.max() - grp.x.min())
            dy = float(grp.y.max() - grp.y.min())
            if min(dx, dy) > 0:
                spans.append(min(dx, dy) / 4.0)
    if not spans:
        return (0.0, 250.0, 2.5)
    upper = float(np.percentile(spans, 90))
    max_span = float(max(spans))
    if 120.0 <= max_span <= 400.0:
        return (0.0, 250.0, 2.5)
    exponent = np.floor(np.log10(max(upper, 1e-6)))
    base = 10.0 ** exponent
    best = base
    for mult in (1.0, 2.0, 2.5, 4.0, 5.0, 8.0, 10.0):
        cand = mult * base
        if cand <= max_span * 1.05:
            best = cand
    r_max = float(np.round(best, 6))
    r_step = float(np.round(r_max / 100.0, 8))
    return (0.0, r_max, r_step)


def resolve_radii(settings: Settings, fields) -> np.ndarray:
    min_auto = min(window.default_rmax for _, _, _, window in fields)
    max_auto = max(window.default_rmax for _, _, _, window in fields)
    if settings.r_step is not None:
        r_max = settings.r_max if settings.r_max is not None else max_auto
        step = settings.r_step
        r_start = step if not settings.r_min else settings.r_min
        if not 0 < r_start < r_max:
            raise ValueError("invalid_radii")
        count = int(np.floor((r_max - r_start) / step + 1e-9)) + 1
        if not 2 <= count <= 2000:
            raise ValueError("invalid_radii")
        radii = np.round(r_start + np.arange(count, dtype=float) * step, 12)
    else:
        r_max = settings.r_max if settings.r_max is not None else min_auto
        if settings.r_max is None:
            limit = min_auto
        else:
            limit = max(
                (
                    min(w.geometry.bounds[2] - w.geometry.bounds[0], w.geometry.bounds[3] - w.geometry.bounds[1])
                    for _, _, _, w in fields
                ),
                default=max_auto,
            )
        r_min = (r_max / settings.n_radii) if not settings.r_min else settings.r_min
        if not 0 < r_min < r_max or (settings.r_max is None and r_max > limit * (1 + 1e-12)):
            raise ValueError("radius_range", f"{min_auto:.6g}")
        radii = np.linspace(r_min, r_max, settings.n_radii)
    if settings.local_radius is not None and settings.local_radius > radii[-1]:
        raise ValueError("local_radius_range", f"{radii[-1]:.6g}")
    return radii


def finite_mean(values, axis):
    values = np.asarray(values, dtype=float)
    count = np.isfinite(values).sum(axis=axis)
    total = np.nansum(values, axis=axis)
    return np.divide(total, count, out=np.full(np.shape(total), np.nan), where=count > 0)


def adjust_holm_values(p_values) -> list[float]:
    n = len(p_values)
    out = [np.nan] * n
    indexed = sorted(((float(p), idx) for idx, p in enumerate(p_values) if np.isfinite(p)), key=lambda x: x[0])
    m = len(indexed)
    running = 0.0
    for rank, (p, idx) in enumerate(indexed):
        running = min(1.0, max(running, (m - rank) * p))
        out[idx] = running
    return out


def adjust_holm(records, get_p):
    tested = sorted(((get_p(row), row) for row in records if np.isfinite(get_p(row))), key=lambda x: x[0])
    adjusted = 0.
    for rank, (p, row) in enumerate(tested):
        adjusted = min(1., max(adjusted, (len(tested) - rank) * p))
        row.p_adjusted = adjusted


def prepare(data, settings, stop):
    fields, issues = [], [data.issues]
    grouped = dict(tuple(data.points.groupby(["sample_id", "roi_id"], sort=False)))
    for _, sample, roi in data.fields:
        if (sample, roi) not in grouped:
            issues.append(pd.DataFrame([dict(sample_id=sample, roi_id=roi, reason="no_valid_rows")]))
    for (sample, roi), frame in grouped.items():
        check_cancel(stop)
        xy = frame[["x", "y"]].to_numpy()
        try:
            window = make_window(
                settings.roi_mode,
                xy,
                data.bounds.get((sample, roi), settings.rectangle),
                settings.custom.get((sample, roi), settings.custom.get(None)),
                settings.hull_ratio,
                edge_mode=settings.edge_correction,
            )
        except ValueError as error:
            issues.append(issue_rows(frame.iloc[:1], str(error)))
            continue
        keep = inside(window, xy)
        if not keep.all():
            issues.append(issue_rows(frame.loc[~keep], "outside_roi"))
        fields.append((sample, roi, frame.loc[keep], window))
    if not fields:
        reasons = pd.concat(issues).reason.dropna().drop_duplicates()
        raise ValueError("no_valid_roi", ", ".join(reasons))
    return fields, issues


def compute_pattern(frame, window, label, radii, settings, rng, stop, subset_types=None, cross_pair=None):
    xy = frame[["x", "y"]].to_numpy()
    marks = frame.cell_type.to_numpy()
    cross = cross_pair is not None or settings.marks == "cross"
    eff_window = window
    if cross:
        a, b = cross_pair if cross_pair is not None else (settings.from_type, settings.to_type)
        keep = np.isin(marks, (a, b))
        xy, marks = xy[keep], marks[keep]
        if settings.infer_per_label and window.mode in ("infer", "locked") and len(xy) >= 3:
            try:
                eff_window = make_window(
                    window.mode, xy, hull_ratio=settings.hull_ratio, edge_mode=settings.edge_correction
                )
                keep_in = inside(eff_window, xy)
                if not keep_in.all():
                    xy, marks = xy[keep_in], marks[keep_in]
            except ValueError:
                eff_window = window
        observed, targets = xy[marks == a], xy[marks == b]
    else:
        if subset_types is not None:
            keep = np.isin(marks, subset_types)
            xy, marks = xy[keep], marks[keep]
        elif settings.marks == "each":
            keep = marks == label
            xy, marks = xy[keep], marks[keep]
        observed, targets = xy, None
        if (
            (subset_types is not None or settings.marks == "each")
            and settings.infer_per_label
            and window.mode in ("infer", "locked")
            and len(observed) >= 3
        ):
            try:
                eff_window = make_window(
                    window.mode, observed, hull_ratio=settings.hull_ratio, edge_mode=settings.edge_correction
                )
                keep_in = inside(eff_window, observed)
                if not keep_in.all():
                    observed, marks = observed[keep_in], marks[keep_in]
                    xy = observed
            except ValueError:
                eff_window = window

    k = ripley(observed, eff_window, radii, targets, settings.intensity, stop)
    curve = centered_l(k, radii)
    locked = (
        boundary_points(eff_window, observed)
        if eff_window.mode == "locked" and not cross
        else np.zeros(len(observed), dtype=bool)
    )
    null = (
        "random_label" if cross
        else "fixed_intensity" if settings.intensity is not None
        else "locked_csr" if eff_window.mode == "locked"
        else "csr"
    )
    qc = []
    if len(observed) < (1 if cross else 2) or (cross and len(targets) == 0) or not np.isfinite(curve).any():
        qc.append("insufficient")
    if not cross and len(np.unique(observed, axis=0)) < len(observed):
        qc.append("csr_coincident")
    if eff_window.mode == "locked" and not cross and len(observed) - locked.sum() < 3:
        qc.append("few_interior")
    if (
        eff_window.mode == "locked"
        and not cross
        and (subset_types is not None or settings.marks == "each")
        and not settings.infer_per_label
    ):
        qc.append("locked_all_only")
    if eff_window.mode == "infer" and not cross:
        qc.append("estimated_roi")
    if cross and (len(observed) < 2 or len(targets) < 2):
        qc.append("too_few_for_test")
    if not cross and len(observed) < 4:
        qc.append("too_few_for_test")

    envelope = None
    if settings.test and not qc:
        if eff_window.correction == "translation" and not eff_window.is_rectangle:
            eff_window.translation_grid()
        sims = np.empty((settings.simulations + 1, len(radii)))
        sims[0] = curve
        for i in range(1, len(sims)):
            check_cancel(stop)
            if cross:
                labels = rng.permutation(marks)
                simulated_k = ripley(xy[labels == a], eff_window, radii, xy[labels == b], stop=stop)
            else:
                count = len(observed) - int(locked.sum())
                simulated = (
                    settings.intensity.sample(count, eff_window, rng)
                    if settings.intensity is not None
                    else uniform(eff_window, count, rng, stop)
                )
                if locked.any():
                    simulated = np.vstack((observed[locked], simulated))
                simulated_k = ripley(simulated, eff_window, radii, intensity=settings.intensity, stop=stop)
            sims[i] = centered_l(simulated_k, radii)
        envelope = global_envelope(sims)
        if envelope.status == "insufficient":
            qc.append("no_common_support")
    status = envelope.status if envelope else ("insufficient" if "insufficient" in qc else "descriptive")
    local = None
    if settings.local_radius is not None and not cross and settings.intensity is None:
        local = local_neighbors(observed, eff_window, settings.local_radius, stop)
    return (
        eff_window,
        xy,
        marks,
        len(observed),
        None if targets is None else len(targets),
        int(locked.sum()),
        k,
        curve,
        envelope,
        status,
        null,
        qc,
        local,
    )


def compute_band(curves: np.ndarray, band: str, rng=None, stop=None):
    """Compute mean, n, sd, se, and selected variability band (sd, se, ci, off) across samples."""
    curves = np.asarray(curves, dtype=float)
    n = np.isfinite(curves).sum(axis=0)
    mean = finite_mean(curves, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sd = np.sqrt(
            np.divide(
                np.nansum((curves - mean) ** 2, axis=0),
                n - 1,
                out=np.full(len(mean), np.nan),
                where=n > 1,
            )
        )
        se = np.divide(sd, np.sqrt(n), out=np.full(len(mean), np.nan), where=n > 1)
    low, high = np.full(len(mean), np.nan), np.full(len(mean), np.nan)
    if band == "sd":
        low, high = mean - sd, mean + sd
    elif band == "se":
        low, high = mean - se, mean + se
    elif band == "ci":
        if rng is None:
            rng = np.random.default_rng(2406)
        supported = n >= 3
        if supported.any():
            sub = curves[:, supported]
            bootstrap = np.empty((2000, int(supported.sum())))
            chunk = 250
            for start in range(0, 2000, chunk):
                check_cancel(stop)
                end = min(2000, start + chunk)
                idx = rng.integers(len(curves), size=(end - start, len(curves)))
                bootstrap[start:end] = finite_mean(sub[idx], axis=1)
            valid = np.isfinite(bootstrap).sum(axis=0) >= 500
            if valid.any():
                sup_indices = np.flatnonzero(supported)[valid]
                with np.errstate(invalid="ignore"):
                    low[sup_indices], high[sup_indices] = np.nanquantile(
                        bootstrap[:, valid], [.025, .975], axis=0
                    )
    return mean, n, sd, se, low, high


def with_band(summary: Summary, band: str, seed: int = 2406) -> Summary:
    mean, n, sd, se, low, high = compute_band(summary.curves, band, task_rng(seed, "band", summary.group, summary.label))
    return replace(summary, mean=mean, n=n, sd=sd, se=se, low=low, high=high)


def summarize(patterns, settings, rng, stop):
    by_sample, by_group = {}, {}
    group_order, label_order = {}, {}
    for pattern in patterns:
        group_order.setdefault(pattern.group, len(group_order))
        label_order.setdefault(
            pattern.label, (MERGED_SEP in pattern.label, "→" in pattern.label, pattern.label.casefold(), pattern.label)
        )
        if np.isfinite(pattern.curve).any():
            key = (pattern.group, pattern.label, pattern.sample)
            by_sample.setdefault(key, []).append(pattern.curve)
    for (group, label, sample), curves in by_sample.items():
        by_group.setdefault((group, label), {})[sample] = finite_mean(curves, axis=0)
    out = []
    ordered_keys = sorted(by_group, key=lambda gl: (group_order[gl[0]], label_order[gl[1]]))
    for (group, label) in ordered_keys:
        sample_curves = by_group[(group, label)]
        curves = np.stack(list(sample_curves.values()))
        mean, n, sd, se, low, high = compute_band(curves, settings.band, rng, stop)
        out.append(Summary(group, label, list(sample_curves), curves, mean, n, low, high, sd=sd, se=se))
    return out


def build_custom_summary(
    results: Results,
    groups: tuple[str, ...] | None,
    subset_types: tuple[str, ...] | None,
    require_all: bool = True,
    title: str = "",
    stop=None,
) -> Summary | None:
    """Build a Summary on results.radii for an arbitrary treatment filter and merged point-type set."""
    data, settings, radii = results.data, results.settings, results.radii
    fields, _ = prepare(data, settings, stop)
    sample_groups = data.points.drop_duplicates("sample_id").set_index("sample_id").group
    by_sample = {}
    for sample, roi, frame, window in fields:
        grp = sample_groups[sample]
        if groups is not None and grp not in groups:
            continue
        if subset_types is not None:
            present = set(frame.cell_type.unique()) - {""}
            if require_all and not set(subset_types).issubset(present):
                continue
            if not require_all and not (set(subset_types) & present):
                continue
            sub_xy = frame.loc[frame.cell_type.isin(subset_types), ["x", "y"]].to_numpy()
        else:
            sub_xy = frame[["x", "y"]].to_numpy()
        if len(sub_xy) < 2:
            continue
        eff_window = window
        if subset_types is not None and settings.infer_per_label and window.mode in ("infer", "locked") and len(sub_xy) >= 3:
            try:
                eff_window = make_window(
                    window.mode, sub_xy, hull_ratio=settings.hull_ratio, edge_mode=settings.edge_correction
                )
                sub_xy = sub_xy[inside(eff_window, sub_xy)]
            except ValueError:
                eff_window = window
        if len(sub_xy) < 2:
            continue
        k = ripley(sub_xy, eff_window, radii, intensity=settings.intensity, stop=stop)
        curve = centered_l(k, radii)
        if np.isfinite(curve).any():
            by_sample.setdefault(sample, []).append(curve)
    if not by_sample:
        return None
    sample_ids = list(by_sample.keys())
    curves = np.stack([finite_mean(c, axis=0) for c in by_sample.values()])
    mean, n, sd, se, low, high = compute_band(
        curves, settings.band, task_rng(settings.seed, "custom", title), stop
    )
    grp_label = groups[0] if groups and len(groups) == 1 else (" + ".join(groups) if groups else "")
    pt_label = merged_label(subset_types) if subset_types else ALL
    return Summary(grp_label, pt_label, sample_ids, curves, mean, n, low, high, sd=sd, se=se, series_id=title)


def _derrick_partially_paired_t(
    sa: np.ndarray, sb: np.ndarray, sa_pair: np.ndarray, sb_pair: np.ndarray
) -> tuple[float, float]:
    """Derrick et al. (2017) two-sample t-test for partially overlapping samples."""
    na, nb, npair = len(sa), len(sb), len(sa_pair)
    if na < 2 or nb < 2:
        return (np.nan, np.nan)
    va = float(np.var(sa, ddof=1))
    vb = float(np.var(sb, ddof=1))
    r = 0.0
    if npair >= 2 and np.std(sa_pair, ddof=1) > 0 and np.std(sb_pair, ddof=1) > 0:
        corr = np.corrcoef(sa_pair, sb_pair)[0, 1]
        if np.isfinite(corr):
            r = float(np.clip(corr, -0.9999, 0.9999))
    se2 = va / na + vb / nb - 2.0 * npair * r * np.sqrt(va * vb) / (na * nb)
    diff = float(np.mean(sa) - np.mean(sb))
    if se2 <= 1e-18:
        return (0.0, 1.0) if abs(diff) <= 1e-12 else (np.copysign(np.inf, diff), 0.0)
    t_stat = diff / np.sqrt(se2)
    denom = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    welch_df = (va / na + vb / nb) ** 2 / denom if denom > 0 else float(na + nb - 2)
    total_distinct = max(1, na + nb - npair)
    unpaired = max(0, na + nb - 2 * npair)
    df = max(1.0, (npair - 1) + (unpaired / total_distinct) * max(0.0, welch_df - max(0, npair - 1)))
    p = float(2.0 * sp_stats.t.sf(abs(t_stat), df))
    return (float(t_stat), p)


def _studentized_envelope_to_physical(pivots: np.ndarray, obs_effect: np.ndarray, se0: np.ndarray, sup: np.ndarray) -> Envelope:
    """Run global ERL envelope on studentized pivotal statistics and map critical bounds back to physical units."""
    env_t = global_envelope(pivots)
    lo = np.full(len(obs_effect), np.nan)
    hi = np.full(len(obs_effect), np.nan)
    lo[sup] = env_t.lower[sup] * se0
    hi[sup] = env_t.upper[sup] * se0
    status = "different" if (np.isfinite(env_t.p) and env_t.p <= .05) else env_t.status
    return Envelope(lo, hi, env_t.p, status, env_t.side)


def _pairwise_test(
    a: Summary,
    b: Summary,
    name_a: str,
    name_b: str,
    label_tag: str,
    r_mask: np.ndarray,
    simulations: int,
    pairing_mode: str,
    perm_stat: str,
    meta: pd.DataFrame | None,
    rng,
    stop,
) -> Contrast | None:
    na_full = np.isfinite(a.curves[:, r_mask]).any(axis=1)
    nb_full = np.isfinite(b.curves[:, r_mask]).any(axis=1)
    ids_a = [s for s, ok in zip(a.sample_ids, na_full) if ok]
    ids_b = [s for s, ok in zip(b.sample_ids, nb_full) if ok]
    xa = a.curves[na_full]
    xb = b.curves[nb_full]
    if len(ids_a) < 2 or len(ids_b) < 2:
        return None

    common_ids = [s for s in ids_a if s in set(ids_b)]
    use_pair_col = False
    if not common_ids and meta is not None and "pair_id" in meta.columns:
        pa = meta.reindex(ids_a)["pair_id"].fillna("")
        pb = meta.reindex(ids_b)["pair_id"].fillna("")
        if pairing_mode == "paired" or (
            pairing_mode == "auto" and (pa != "").all() and set(pa) == set(pb) and pa.is_unique and pb.is_unique
        ):
            if "" not in set(pa) and "" not in set(pb) and pa.is_unique and pb.is_unique and set(pa) == set(pb):
                use_pair_col = True

    if pairing_mode == "independent":
        design = "independent"
    elif use_pair_col:
        design = "paired"
    elif pairing_mode == "paired":
        if len(common_ids) < 2:
            return None
        design = "paired"
    else:
        if len(common_ids) == 0:
            design = "independent"
        elif len(common_ids) == len(ids_a) == len(ids_b):
            design = "paired"
        else:
            design = "partial_overlap"

    nr = xa.shape[1]
    effects = np.full((simulations + 1, nr), np.nan)

    if design == "paired":
        if use_pair_col:
            pa = meta.loc[ids_a, "pair_id"].tolist()
            pb = meta.loc[ids_b, "pair_id"].tolist()
            b_by_pair = {p: row for p, row in zip(pb, xb)}
            xa_m = xa
            xb_m = np.stack([b_by_pair[p] for p in pa])
            n_used = len(xa_m)
        else:
            idx_a = {s: i for i, s in enumerate(ids_a)}
            idx_b = {s: i for i, s in enumerate(ids_b)}
            xa_m = np.stack([xa[idx_a[s]] for s in common_ids])
            xb_m = np.stack([xb[idx_b[s]] for s in common_ids])
            n_used = len(common_ids)
        diff = xa_m - xb_m
        sup = r_mask & (np.isfinite(diff).sum(axis=0) >= 2)
        if not sup.any():
            return None
        sub_diff = diff[:, sup]
        n_r = np.isfinite(sub_diff).sum(axis=0)
        m0 = finite_mean(sub_diff, axis=0)
        effects[0, sup] = m0

        use_welch = perm_stat == "welch" and int(n_r.min()) >= 3
        pivots = np.full((simulations + 1, nr), np.nan) if use_welch else None
        se0 = None
        if use_welch:
            sd0 = np.sqrt(np.divide(np.nansum((sub_diff - m0) ** 2, axis=0), np.maximum(1, n_r - 1)))
            se0 = np.maximum(1e-12, sd0 / np.sqrt(n_r))
            pivots[0, sup] = m0 / se0

        chunk = 250
        for start in range(1, simulations + 1, chunk):
            check_cancel(stop)
            end = min(simulations + 1, start + chunk)
            signs = rng.choice([-1.0, 1.0], size=(end - start, n_used, 1))
            d_star = sub_diff[None, :, :] * signs
            mb = finite_mean(d_star, axis=1)
            effects[start:end, sup] = mb
            if use_welch:
                sdb = np.sqrt(np.divide(np.nansum((d_star - mb[:, None, :]) ** 2, axis=1), np.maximum(1, n_r - 1)))
                seb = np.maximum(1e-12, sdb / np.sqrt(n_r))
                pivots[start:end, sup] = mb / seb

        if use_welch:
            env = _studentized_envelope_to_physical(pivots, effects[0], se0, sup)
        else:
            env = global_envelope(effects)
            if np.isfinite(env.p) and env.p <= .05:
                env = replace(env, status="different")

        sa = finite_mean(xa_m[:, sup], axis=1)
        sb = finite_mean(xb_m[:, sup], axis=1)
        valid_s = np.isfinite(sa) & np.isfinite(sb)
        sa, sb = sa[valid_s], sb[valid_s]
        p_stat, p_val, np_stat, np_val = np.nan, np.nan, np.nan, np.nan
        mean_diff = float(np.mean(sa - sb)) if len(sa) else np.nan
        if len(sa) >= 2:
            d = sa - sb
            if np.allclose(d, 0):
                p_stat, p_val, np_stat, np_val = 0.0, 1.0, 0.0, 1.0
            else:
                res_t = sp_stats.ttest_rel(sa, sb)
                p_stat, p_val = float(res_t.statistic), float(res_t.pvalue)
                try:
                    res_w = sp_stats.wilcoxon(d, zero_method="zsplit")
                    np_stat, np_val = float(res_w.statistic), float(res_w.pvalue)
                except ValueError:
                    np_stat, np_val = np.nan, np.nan
        return Contrast(
            name_a, name_b, label_tag, effects[0], env, n_used, n_used,
            design="paired", n_paired=n_used, mean_diff_scalar=mean_diff,
            parametric_test="paired_t", parametric_stat=p_stat, parametric_p=p_val,
            nonparametric_test="wilcoxon", nonparametric_stat=np_stat, nonparametric_p=np_val,
        )

    if design == "independent":
        na, nb = len(xa), len(xb)
        sup = r_mask & (np.isfinite(xa).sum(axis=0) >= 2) & (np.isfinite(xb).sum(axis=0) >= 2)
        if not sup.any():
            return None
        u, v = xa[:, sup], xb[:, sup]
        pooled = np.vstack((u, v))
        if meta is not None and "block_id" in meta.columns and set(ids_a + ids_b).issubset(meta.index):
            block = meta.loc[ids_a + ids_b, "block_id"].to_numpy()
            groups = [np.flatnonzero(block == name) for name in np.unique(block)]
            if not any(np.any(g < na) and np.any(g >= na) for g in groups):
                groups = [np.arange(na + nb)]
        else:
            groups = [np.arange(na + nb)]

        ma0, mb0 = finite_mean(u, axis=0), finite_mean(v, axis=0)
        effects[0, sup] = ma0 - mb0
        use_welch = perm_stat == "welch"
        pivots = np.full((simulations + 1, nr), np.nan) if use_welch else None
        se0 = None
        if use_welch:
            na_r0 = np.isfinite(u).sum(axis=0)
            nb_r0 = np.isfinite(v).sum(axis=0)
            va0 = np.divide(np.nansum((u - ma0) ** 2, axis=0), np.maximum(1, na_r0 - 1))
            vb0 = np.divide(np.nansum((v - mb0) ** 2, axis=0), np.maximum(1, nb_r0 - 1))
            se0 = np.sqrt(np.maximum(1e-18, va0 / np.maximum(1, na_r0) + vb0 / np.maximum(1, nb_r0)))
            pivots[0, sup] = (ma0 - mb0) / se0

        for i in range(1, simulations + 1):
            check_cancel(stop)
            assignment = np.arange(na + nb)
            for positions in groups:
                assignment[positions] = rng.permutation(positions)
            ga, gb = pooled[assignment[:na]], pooled[assignment[na:]]
            mab, mbb = finite_mean(ga, axis=0), finite_mean(gb, axis=0)
            effects[i, sup] = mab - mbb
            if use_welch:
                nar = np.isfinite(ga).sum(axis=0)
                nbr = np.isfinite(gb).sum(axis=0)
                ok = (nar >= 2) & (nbr >= 2)
                if ok.all():
                    vab = np.divide(np.nansum((ga - mab) ** 2, axis=0), nar - 1)
                    vbb = np.divide(np.nansum((gb - mbb) ** 2, axis=0), nbr - 1)
                    seb = np.sqrt(np.maximum(1e-18, vab / nar + vbb / nbr))
                    pivots[i, sup] = (mab - mbb) / seb
                elif ok.any():
                    vab = np.divide(np.nansum((ga[:, ok] - mab[ok]) ** 2, axis=0), nar[ok] - 1)
                    vbb = np.divide(np.nansum((gb[:, ok] - mbb[ok]) ** 2, axis=0), nbr[ok] - 1)
                    seb = np.sqrt(np.maximum(1e-18, vab / nar[ok] + vbb / nbr[ok]))
                    sup_idx = np.flatnonzero(sup)[ok]
                    pivots[i, sup_idx] = (mab[ok] - mbb[ok]) / seb

        if use_welch:
            env = _studentized_envelope_to_physical(pivots, effects[0], se0, sup)
        else:
            env = global_envelope(effects)
            if np.isfinite(env.p) and env.p <= .05:
                env = replace(env, status="different")

        sa = finite_mean(u, axis=1)
        sb = finite_mean(v, axis=1)
        sa, sb = sa[np.isfinite(sa)], sb[np.isfinite(sb)]
        mean_diff = float(np.mean(sa) - np.mean(sb)) if len(sa) and len(sb) else np.nan
        p_stat, p_val, np_stat, np_val = np.nan, np.nan, np.nan, np.nan
        if len(sa) >= 2 and len(sb) >= 2:
            if np.var(sa, ddof=1) == 0 and np.var(sb, ddof=1) == 0 and np.isclose(np.mean(sa), np.mean(sb)):
                p_stat, p_val = 0.0, 1.0
            else:
                res_t = sp_stats.ttest_ind(sa, sb, equal_var=False)
                p_stat, p_val = float(res_t.statistic), float(res_t.pvalue)
            res_u = sp_stats.mannwhitneyu(sa, sb, alternative="two-sided")
            np_stat, np_val = float(res_u.statistic), float(res_u.pvalue)
        return Contrast(
            name_a, name_b, label_tag, effects[0], env, na, nb,
            design="independent", n_paired=0, mean_diff_scalar=mean_diff,
            parametric_test="welch_t", parametric_stat=p_stat, parametric_p=p_val,
            nonparametric_test="mann_whitney_u", nonparametric_stat=np_stat, nonparametric_p=np_val,
        )

    # Partially overlapping samples: Wild Embryo-Cluster Score Resampling (ERL) + Derrick's t-test
    na, nb = len(xa), len(xb)
    sup = r_mask & (np.isfinite(xa).sum(axis=0) >= 2) & (np.isfinite(xb).sum(axis=0) >= 2)
    if not sup.any():
        return None
    u_curves, v_curves = xa[:, sup], xb[:, sup]
    mean_a = finite_mean(u_curves, axis=0)
    mean_b = finite_mean(v_curves, axis=0)
    na_r = np.isfinite(u_curves).sum(axis=0)
    nb_r = np.isfinite(v_curves).sum(axis=0)
    effects[0, sup] = mean_a - mean_b

    all_specimens = list(dict.fromkeys(ids_a + ids_b))
    m_clusters = len(all_specimens)
    bessel = np.sqrt(m_clusters / max(1, m_clusters - 1))
    idx_a = {s: i for i, s in enumerate(ids_a)}
    idx_b = {s: i for i, s in enumerate(ids_b)}
    nsup = int(sup.sum())
    weights = np.zeros((m_clusters, nsup), dtype=float)
    for c_idx, sid in enumerate(all_specimens):
        if sid in idx_a:
            ra = u_curves[idx_a[sid]] - mean_a
            weights[c_idx] += np.where(np.isfinite(ra) & (na_r > 0), ra / na_r, 0.0)
        if sid in idx_b:
            rb = v_curves[idx_b[sid]] - mean_b
            weights[c_idx] -= np.where(np.isfinite(rb) & (nb_r > 0), rb / nb_r, 0.0)
    weights *= bessel
    check_cancel(stop)
    xi_mat = rng.choice([-1.0, 1.0], size=(simulations, m_clusters))
    effects[1:, sup] = xi_mat @ weights
    env = global_envelope(effects)
    if np.isfinite(env.p) and env.p <= .05:
        env = replace(env, status="different")

    sa = finite_mean(u_curves, axis=1)
    sb = finite_mean(v_curves, axis=1)
    sa_pair = np.array([sa[idx_a[s]] for s in common_ids], dtype=float)
    sb_pair = np.array([sb[idx_b[s]] for s in common_ids], dtype=float)
    mean_diff = float(np.mean(sa) - np.mean(sb))
    t_stat, t_p = _derrick_partially_paired_t(sa, sb, sa_pair, sb_pair)
    sw = np.zeros(m_clusters, dtype=float)
    ma_s, mb_s = float(np.mean(sa)), float(np.mean(sb))
    for c_idx, sid in enumerate(all_specimens):
        if sid in idx_a:
            sw[c_idx] += (sa[idx_a[sid]] - ma_s) / na
        if sid in idx_b:
            sw[c_idx] -= (sb[idx_b[sid]] - mb_s) / nb
    sw *= bessel
    boot_s = np.empty(simulations + 1)
    boot_s[0] = abs(mean_diff)
    boot_s[1:] = np.abs(rng.choice([-1.0, 1.0], size=(simulations, m_clusters)) @ sw)
    np_p = float(np.mean(boot_s >= boot_s[0] - 1e-12))
    return Contrast(
        name_a, name_b, label_tag, effects[0], env, na, nb,
        design="partial_overlap", n_paired=len(common_ids), mean_diff_scalar=mean_diff,
        parametric_test="derrick_t", parametric_stat=t_stat, parametric_p=t_p,
        nonparametric_test="cluster_perm", nonparametric_stat=abs(mean_diff), nonparametric_p=np_p,
    )


def _pointwise_f_independent(groups_curves: list[np.ndarray]) -> np.ndarray:
    k = len(groups_curves)
    ns = [np.isfinite(g).sum(axis=0) for g in groups_curves]
    means = [finite_mean(g, axis=0) for g in groups_curves]
    n_tot = np.sum(ns, axis=0)
    grand = np.divide(
        np.sum([n * m for n, m in zip(ns, means)], axis=0),
        n_tot,
        out=np.full_like(means[0], np.nan),
        where=n_tot > k,
    )
    ss_between = np.sum([n * (m - grand) ** 2 for n, m in zip(ns, means)], axis=0)
    ss_within = np.sum([np.nansum((g - m) ** 2, axis=0) for g, m in zip(groups_curves, means)], axis=0)
    df1 = k - 1
    df2 = n_tot - k
    ms_between = ss_between / df1
    with np.errstate(divide="ignore", invalid="ignore"):
        ms_within = np.divide(ss_within, df2, out=np.full_like(grand, np.nan), where=df2 > 0)
        f_val = np.divide(ms_between, ms_within, out=np.zeros_like(grand), where=ms_within > 1e-18)
    f_val[df2 <= 0] = np.nan
    return f_val


def _pointwise_f_paired(aligned_stack: np.ndarray) -> np.ndarray:
    k, n_subj, _ = aligned_stack.shape
    complete = np.isfinite(aligned_stack).all(axis=(0, 1))
    out = np.full(aligned_stack.shape[2], np.nan)
    if not complete.any() or n_subj < 2 or k < 2:
        return out
    y = aligned_stack[:, :, complete]
    cond_mean = y.mean(axis=1, keepdims=True)
    subj_mean = y.mean(axis=0, keepdims=True)
    grand = y.mean(axis=(0, 1), keepdims=True)
    ss_cond = n_subj * ((cond_mean - grand) ** 2).sum(axis=(0, 1))
    residuals = y - cond_mean - subj_mean + grand
    ss_err = (residuals ** 2).sum(axis=(0, 1))
    ms_cond = ss_cond / (k - 1)
    ms_err = ss_err / ((k - 1) * (n_subj - 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        out[complete] = np.divide(ms_cond, ms_err, out=np.zeros(int(complete.sum())), where=ms_err > 1e-18)
    return out


def compare_series(
    series: list[Summary],
    radii: np.ndarray,
    data=None,
    simulations: int = 999,
    seed: int = 2406,
    pairing_mode: str = "auto",
    perm_stat: str = "welch",
    r_range: tuple[float, float] | None = None,
    all_text: str = "All",
    stop=None,
) -> GroupComparison:
    """Run omnibus + all-pairs functional & scalar biostatistical tests across selected Summary series."""
    if r_range is not None:
        r_mask = (radii >= r_range[0] - 1e-12) & (radii <= r_range[1] + 1e-12)
    else:
        r_mask = np.ones(len(radii), dtype=bool)
    meta = (
        data.points.drop_duplicates("sample_id").set_index("sample_id")
        if data is not None and not data.points.empty
        else None
    )

    only_groups = len({s.label for s in series}) == 1
    only_labels = len({s.group for s in series}) == 1
    names = []
    for s in series:
        if s.series_id:
            names.append(s.series_id)
        elif only_groups and s.group:
            names.append(s.group)
        elif only_labels:
            names.append(s.label or all_text)
        else:
            names.append(s.format_name(all_text))

    eligible = [
        (s, nm) for s, nm in zip(series, names) if np.isfinite(s.curves[:, r_mask]).any(axis=1).sum() >= 2
    ]
    if len(eligible) < 2:
        return GroupComparison(series=series, omnibus=None, contrasts=[])

    contrasts = []
    for (sa, na), (sb, nb) in combinations(eligible, 2):
        lbl = sa.label if (sa.label == sb.label and not sa.series_id and not sb.series_id) else f"{na} vs {nb}"
        pair_rng = task_rng(seed, "pair", na, nb)
        c = _pairwise_test(
            sa, sb, na, nb, lbl, r_mask, simulations, pairing_mode, perm_stat, meta, pair_rng, stop
        )
        if c is not None:
            contrasts.append(c)

    if not contrasts:
        return GroupComparison(series=series, omnibus=None, contrasts=[])

    adjust_holm(contrasts, lambda item: item.envelope.p)
    for idx, adj in enumerate(adjust_holm_values([c.parametric_p for c in contrasts])):
        contrasts[idx].parametric_p_holm = adj
    for idx, adj in enumerate(adjust_holm_values([c.nonparametric_p for c in contrasts])):
        contrasts[idx].nonparametric_p_holm = adj

    k = len(eligible)
    if k == 2:
        c0 = contrasts[0]
        sup = np.isfinite(c0.envelope.upper)
        r_sup = radii[sup] if sup.any() else radii[r_mask]
        omnibus = OmnibusStat(
            series_names=[nm for _, nm in eligible],
            design=c0.design,
            k_groups=2,
            r_min=float(r_sup[0]),
            r_max=float(r_sup[-1]),
            n_supported_radii=int(sup.sum()),
            statistic_curve=c0.effect,
            envelope=c0.envelope,
            p_global=float(c0.envelope.p),
            significant_spans=interval_strings(radii, c0.envelope.side),
            parametric_test=c0.parametric_test,
            parametric_stat=c0.parametric_stat,
            parametric_p=c0.parametric_p,
            nonparametric_test=c0.nonparametric_test,
            nonparametric_stat=c0.nonparametric_stat,
            nonparametric_p=c0.nonparametric_p,
        )
        return GroupComparison(series=series, omnibus=omnibus, contrasts=contrasts)

    om_rng = task_rng(seed, "omnibus", *(nm for _, nm in eligible))
    id_sets = [
        [sid for sid, ok in zip(s.sample_ids, np.isfinite(s.curves[:, r_mask]).any(axis=1)) if ok]
        for s, _ in eligible
    ]
    curve_mats = [s.curves[np.isfinite(s.curves[:, r_mask]).any(axis=1)] for s, _ in eligible]
    common_all = set(id_sets[0]).intersection(*id_sets[1:])
    any_overlap = any(len(set(a) & set(b)) > 0 for a, b in combinations(id_sets, 2))
    if pairing_mode == "independent" or not any_overlap:
        om_design = "independent"
    elif pairing_mode == "paired" or (
        len(common_all) == len(id_sets[0]) and all(len(x) == len(common_all) for x in id_sets)
    ):
        om_design = "paired" if len(common_all) >= 2 else "partial_overlap"
    else:
        om_design = "partial_overlap"

    sup = r_mask.copy()
    for mat in curve_mats:
        sup &= np.isfinite(mat).sum(axis=0) >= 2
    nr = len(radii)
    f_sims = np.full((simulations + 1, nr), np.nan)
    p_test, p_stat, p_val = "", np.nan, np.nan
    np_test, np_stat, np_val = "", np.nan, np.nan

    if sup.any():
        if om_design == "independent":
            sub_mats = [m[:, sup] for m in curve_mats]
            sizes = [len(m) for m in sub_mats]
            pooled = np.vstack(sub_mats)
            f_sims[0, sup] = _pointwise_f_independent(sub_mats)
            for i in range(1, simulations + 1):
                check_cancel(stop)
                perm = om_rng.permutation(len(pooled))
                splits = np.split(pooled[perm], np.cumsum(sizes)[:-1])
                f_sims[i, sup] = _pointwise_f_independent(splits)
            scalars = [finite_mean(m, axis=1) for m in sub_mats]
            scalars = [x[np.isfinite(x)] for x in scalars]
            if all(len(x) >= 2 for x in scalars):
                res_f = sp_stats.f_oneway(*scalars)
                p_test, p_stat, p_val = "oneway_anova", float(res_f.statistic), float(res_f.pvalue)
                res_kw = sp_stats.kruskal(*scalars)
                np_test, np_stat, np_val = "kruskal_wallis", float(res_kw.statistic), float(res_kw.pvalue)

        elif om_design == "paired":
            ordered_common = [sid for sid in id_sets[0] if sid in common_all]
            aligned = np.stack(
                [np.stack([mat[ids.index(sid), sup] for sid in ordered_common]) for mat, ids in zip(curve_mats, id_sets)]
            )
            f_sims[0, sup] = _pointwise_f_paired(aligned)
            n_subj = aligned.shape[1]
            for i in range(1, simulations + 1):
                check_cancel(stop)
                permuted = np.empty_like(aligned)
                for s_idx in range(n_subj):
                    order = om_rng.permutation(k)
                    permuted[:, s_idx, :] = aligned[order, s_idx, :]
                f_sims[i, sup] = _pointwise_f_paired(permuted)
            scalar_mat = finite_mean(aligned, axis=2)
            valid_subj = np.isfinite(scalar_mat).all(axis=0)
            scalar_mat = scalar_mat[:, valid_subj]
            if scalar_mat.shape[1] >= 2:
                f_rm = _pointwise_f_paired(scalar_mat[:, :, None])[0]
                df1 = k - 1
                df2 = (k - 1) * (scalar_mat.shape[1] - 1)
                p_test, p_stat = "rm_anova", float(f_rm)
                p_val = float(sp_stats.f.sf(f_rm, df1, df2)) if np.isfinite(f_rm) else np.nan
                res_fr = sp_stats.friedmanchisquare(*[scalar_mat[g] for g in range(k)])
                np_test, np_stat, np_val = "friedman", float(res_fr.statistic), float(res_fr.pvalue)

        else:
            sub_mats = [m[:, sup] for m in curve_mats]
            ns = [np.isfinite(m).sum(axis=0) for m in sub_mats]
            means = [finite_mean(m, axis=0) for m in sub_mats]
            n_tot = np.sum(ns, axis=0)
            grand = np.divide(
                np.sum([n * mu for n, mu in zip(ns, means)], axis=0),
                n_tot,
                out=np.zeros(int(sup.sum())),
                where=n_tot > 0,
            )
            ss_within = np.sum([np.nansum((m - mu) ** 2, axis=0) for m, mu in zip(sub_mats, means)], axis=0)
            sigma2 = np.maximum(1e-12, np.divide(ss_within, np.maximum(1, n_tot - k)))
            f_sims[0, sup] = np.sum([n * (mu - grand) ** 2 for n, mu in zip(ns, means)], axis=0) / ((k - 1) * sigma2)

            all_specimens = list(dict.fromkeys(sid for ids in id_sets for sid in ids))
            m_clusters = len(all_specimens)
            bessel = np.sqrt(m_clusters / max(1, m_clusters - 1))
            nsup = int(sup.sum())
            contrib = np.zeros((k, m_clusters, nsup), dtype=float)
            for g_idx, (mat, ids, mu, n_r) in enumerate(zip(sub_mats, id_sets, means, ns)):
                id_map = {sid: idx for idx, sid in enumerate(ids)}
                for c_idx, sid in enumerate(all_specimens):
                    if sid in id_map:
                        res = mat[id_map[sid]] - mu
                        contrib[g_idx, c_idx] = np.where(np.isfinite(res) & (n_r > 0), res / n_r, 0.0)
            contrib *= bessel
            for start in range(1, simulations + 1, 250):
                check_cancel(stop)
                end = min(simulations + 1, start + 250)
                xi = om_rng.choice([-1.0, 1.0], size=(end - start, m_clusters))
                star_means = np.einsum("bm,kmr->bkr", xi, contrib)
                star_grand = np.divide(
                    np.einsum("kr,bkr->br", np.asarray(ns, dtype=float), star_means),
                    n_tot[None, :],
                    out=np.zeros((end - start, nsup)),
                    where=n_tot[None, :] > 0,
                )
                ss_b = np.einsum(
                    "kr,bkr->br", np.asarray(ns, dtype=float), (star_means - star_grand[:, None, :]) ** 2
                )
                f_sims[start:end, sup] = ss_b / ((k - 1) * sigma2[None, :])

            scalars = [finite_mean(m, axis=1) for m in sub_mats]
            if all(len(x) >= 2 for x in scalars):
                res_f = sp_stats.f_oneway(*scalars)
                obs_f = float(res_f.statistic) if np.isfinite(res_f.statistic) else 0.0
                s_means = [float(np.mean(x)) for x in scalars]
                s_ns = [len(x) for x in scalars]
                s_ntot = sum(s_ns)
                s_within = sum(float(np.sum((x - mu) ** 2)) for x, mu in zip(scalars, s_means)) / max(1, s_ntot - k)
                s_contrib = np.zeros((k, m_clusters), dtype=float)
                for g_idx, (x, ids, mu, ng) in enumerate(zip(scalars, id_sets, s_means, s_ns)):
                    id_map = {sid: idx for idx, sid in enumerate(ids)}
                    for c_idx, sid in enumerate(all_specimens):
                        if sid in id_map:
                            s_contrib[g_idx, c_idx] = (x[id_map[sid]] - mu) / ng
                s_contrib *= bessel
                xi_s = om_rng.choice([-1.0, 1.0], size=(simulations, m_clusters))
                sm_mat = xi_s @ s_contrib.T  # (B, K)
                ns_arr = np.asarray(s_ns, dtype=float)
                sg_vec = (sm_mat @ ns_arr) / s_ntot
                f_boot_vals = (((sm_mat - sg_vec[:, None]) ** 2) @ ns_arr) / max(1e-12, (k - 1) * s_within)
                f_boot = np.r_[obs_f, f_boot_vals]
                p_test, p_stat, p_val = "cluster_anova", obs_f, float(np.mean(f_boot >= obs_f - 1e-12))
                res_kw = sp_stats.kruskal(*scalars)
                np_test, np_stat, np_val = "kruskal_wallis", float(res_kw.statistic), float(res_kw.pvalue)

    om_env = global_envelope(f_sims, alternative="greater")
    sup_env = np.isfinite(om_env.upper)
    r_sup = radii[sup_env] if sup_env.any() else radii[r_mask]
    omnibus = OmnibusStat(
        series_names=[nm for _, nm in eligible],
        design=om_design,
        k_groups=k,
        r_min=float(r_sup[0]) if len(r_sup) else float(radii[0]),
        r_max=float(r_sup[-1]) if len(r_sup) else float(radii[-1]),
        n_supported_radii=int(sup_env.sum()),
        statistic_curve=f_sims[0],
        envelope=om_env,
        p_global=float(om_env.p),
        significant_spans=interval_strings(radii, om_env.side),
        parametric_test=p_test,
        parametric_stat=p_stat,
        parametric_p=p_val,
        nonparametric_test=np_test,
        nonparametric_stat=np_stat,
        nonparametric_p=np_val,
    )
    return GroupComparison(series=series, omnibus=omnibus, contrasts=contrasts)


def compare_groups(data, summaries, settings, rng, stop):
    by_group = {(s.group, s.label): s for s in summaries}
    meta = data.points.drop_duplicates("sample_id").set_index("sample_id")
    contrasts = []
    for a in summaries:
        b = by_group.get((settings.group_b, a.label))
        if a.group != settings.group_a or b is None or min(len(a.curves), len(b.curves)) < 2:
            continue
        xa, xb = a.curves, b.curves
        if settings.paired:
            pa, pb = meta.loc[a.sample_ids].pair_id, meta.loc[b.sample_ids].pair_id
            if "" in set(pa) or "" in set(pb) or not pa.is_unique or not pb.is_unique or set(pa) != set(pb):
                raise ValueError("invalid_pairs")
            by_pair = {pair: curve for pair, curve in zip(pb, xb)}
            xb = np.stack([by_pair[pair] for pair in pa])
        elif any(meta.loc[a.sample_ids + b.sample_ids].pair_id != ""):
            raise ValueError("paired_required")
        stacked = np.vstack((xa, xb))
        supported = np.isfinite(stacked).all(axis=0)
        if not supported.any():
            continue
        na, nb = len(xa), len(xb)
        u, v = xa[:, supported], xb[:, supported]
        pooled = stacked[:, supported]
        block = meta.loc[a.sample_ids + b.sample_ids].block_id.to_numpy()
        groups = [np.flatnonzero(block == name) for name in np.unique(block)]
        if not settings.paired and not any(np.any(g < na) and np.any(g >= na) for g in groups):
            raise ValueError("no_exchangeable_blocks")
        if settings.paired:
            sample_by_pair = dict(zip(pb, b.sample_ids))
            blocks_a = meta.loc[a.sample_ids].block_id.to_numpy()
            blocks_b = meta.loc[[sample_by_pair[pair] for pair in pa]].block_id.to_numpy()
            if not np.array_equal(blocks_a, blocks_b):
                raise ValueError("pair_blocks_conflict")
        effects = np.full((settings.simulations + 1, len(supported)), np.nan)
        effects[0, supported] = u.mean(axis=0) - v.mean(axis=0)
        for i in range(1, len(effects)):
            check_cancel(stop)
            if settings.paired:
                effects[i, supported] = ((u - v) * rng.choice([-1, 1], size=(na, 1))).mean(axis=0)
            else:
                assignment = np.arange(na + nb)
                for positions in groups:
                    assignment[positions] = rng.permutation(positions)
                effects[i, supported] = pooled[assignment[:na]].mean(axis=0) - pooled[assignment[na:]].mean(axis=0)
        envelope = global_envelope(effects)
        if envelope.p <= .05:
            envelope = replace(envelope, status="different")
        contrasts.append(
            Contrast(
                a.group, b.group, a.label, effects[0], envelope, na, nb,
                design="paired" if settings.paired else "independent",
                n_paired=na if settings.paired else 0,
            )
        )
    if not contrasts:
        raise ValueError("contrast_needs_samples")
    adjust_holm(contrasts, lambda item: item.envelope.p)
    return contrasts


def _build_label_specs(data, settings: Settings):
    types = sorted_types(data.points.cell_type)
    if settings.marks == "cross":
        return [(f"{settings.from_type} → {settings.to_type}", None, (settings.from_type, settings.to_type))]
    if settings.marks == "cross_all":
        if len(types) < 2:
            raise ValueError("choose_two_types")
        return [(f"{a} → {b}", None, (a, b)) for a in types for b in types if a != b]
    if settings.marks == "all":
        specs = [(ALL, None, None)]
    elif settings.marks == "each":
        if not types:
            raise ValueError("missing_type")
        specs = [(t, (t,), None) for t in types]
    elif settings.marks in ("each_and_merged", "both"):
        if not types:
            specs = [(ALL, None, None)]
        else:
            specs = [(t, (t,), None) for t in types]
            if len(types) >= 2:
                specs.append((merged_label(types), tuple(types), None))
    else:
        raise ValueError("invalid_settings")
    seen = {lbl for lbl, _, _ in specs}
    for custom_set in settings.merged_sets:
        norm = tuple(sorted_types(custom_set))
        if len(norm) >= 2:
            lbl = merged_label(norm)
            if lbl not in seen:
                specs.append((lbl, norm, None))
                seen.add(lbl)
    return specs


def run_analysis(data, settings, progress=lambda done, total: None, stop=None):
    settings.validate()
    fields, issues = prepare(data, settings, stop)
    radii = resolve_radii(settings, fields)
    specs = _build_label_specs(data, settings)
    work_items = []
    for sample, roi, frame, window in fields:
        present_types = set(frame.cell_type.unique()) - {""}
        for label, subset_types, cross_pair in specs:
            if cross_pair is not None:
                if cross_pair[0] not in present_types or cross_pair[1] not in present_types:
                    continue
            elif subset_types is not None:
                if len(subset_types) == 1:
                    if subset_types[0] not in present_types:
                        continue
                elif settings.merge_require_all:
                    if not set(subset_types).issubset(present_types):
                        continue
                else:
                    if not (set(subset_types) & present_types):
                        continue
            work_items.append((sample, roi, frame, window, label, subset_types, cross_pair))
    if not work_items:
        raise ValueError("no_valid_roi", "insufficient")

    total = len(work_items)
    progress(0, total)
    sample_groups = data.points.drop_duplicates("sample_id").set_index("sample_id").group
    n_workers = min(8, os.cpu_count() or 4) if settings.workers == 0 else settings.workers

    def run_one(item_idx: int, item):
        sample, roi, frame, window, label, subset_types, cross_pair = item
        check_cancel(stop)
        item_rng = task_rng(settings.seed, sample, roi, label)
        res = compute_pattern(
            frame, window, label, radii, settings, item_rng, stop,
            subset_types=subset_types, cross_pair=cross_pair,
        )
        eff_window, *rest = res
        return item_idx, Pattern(sample, roi, sample_groups[sample], label, eff_window, *rest)

    ordered_patterns: list[Pattern | None] = [None] * total
    if n_workers <= 1 or total <= 1:
        for i, item in enumerate(work_items):
            _, pat = run_one(i, item)
            ordered_patterns[i] = pat
            progress(i + 1, total)
    else:
        done_count = 0
        prog_lock = Lock()
        with ThreadPoolExecutor(max_workers=min(n_workers, total)) as pool:
            futures = [pool.submit(run_one, i, item) for i, item in enumerate(work_items)]
            try:
                for fut in as_completed(futures):
                    idx, pat = fut.result()
                    ordered_patterns[idx] = pat
                    with prog_lock:
                        done_count += 1
                        progress(done_count, total)
            except Exception:
                if stop is not None:
                    stop.set()
                for f in futures:
                    f.cancel()
                raise

    patterns = [p for p in ordered_patterns if p is not None]
    for p in patterns:
        for reason in p.qc:
            issues.append(pd.DataFrame([dict(sample_id=p.sample, roi_id=p.roi, reason=reason, label=p.label)]))

    adjust_holm(patterns, lambda item: item.envelope.p if item.envelope else np.nan)
    sum_rng = task_rng(settings.seed, "summarize")
    summaries = summarize(patterns, settings, sum_rng, stop)
    cmp_rng = task_rng(settings.seed, "compare_groups")
    contrasts = compare_groups(data, summaries, settings, cmp_rng, stop) if settings.contrast else []
    comparison = None
    if not contrasts and len(summaries) >= 2:
        first_label = summaries[0].label
        same_label = [s for s in summaries if s.label == first_label]
        initial = same_label if len(same_label) >= 2 else summaries
        comparison = compare_series(
            initial, radii, data=data, simulations=settings.simulations,
            seed=settings.seed, pairing_mode=settings.pairing_mode, perm_stat=settings.perm_stat, stop=stop,
        )
    return Results(
        data, settings, radii, patterns, summaries, contrasts,
        pd.concat(issues, ignore_index=True), comparison=comparison,
    )


def build_natural_presets(results: Results, t) -> list[tuple[str, list[Summary]]]:
    """Return natural (display_name, list[Summary]) presets for UI selection and batch figure export."""
    if results is None or not results.summaries:
        return []
    presets = []
    labels = list(dict.fromkeys(s.label for s in results.summaries))
    groups = list(dict.fromkeys(s.group for s in results.summaries if s.group != ""))
    if len(groups) >= 2:
        for lbl in labels:
            matching = [s for s in results.summaries if s.label == lbl]
            if matching:
                presets.append((t("preset_by_type", label=lbl or t("all")), matching))
    if len(labels) >= 2:
        if groups:
            for grp in groups:
                matching = [s for s in results.summaries if s.group == grp]
                if len(matching) >= 2:
                    presets.append((t("preset_by_group", group=grp), matching))
        else:
            presets.append((t("preset_by_group", group=t("all")), list(results.summaries)))
    if not presets:
        for lbl in labels:
            matching = [s for s in results.summaries if s.label == lbl]
            presets.append((lbl or t("all"), matching))
    return presets


def _safe_slug(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", name.strip())
    cleaned = re.sub(r"\s+", "_", cleaned).strip("._")
    return cleaned[:90] or "figure"


def export_batch_bundle(
    results: Results,
    folder: Path | str,
    t,
    band: str | None = None,
    show_samples: bool = True,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> list[Path]:
    """One-click export of Excel tables plus all preset group & statistical comparison figures."""
    from spatial_plots import contrast_figure, group_figure, omnibus_figure, save_figure

    out_dir = Path(folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    contrast_dir = out_dir / "contrasts"
    written: list[Path] = []

    xlsx_path = out_dir / "spatial-results.xlsx"
    export_data(results, xlsx_path)
    written.append(xlsx_path)

    presets = build_natural_presets(results, t)
    active_band = results.settings.band if band is None else band
    for idx, (preset_title, series_list) in enumerate(presets, start=1):
        comp = (
            compare_series(
                series_list,
                results.radii,
                data=results.data,
                simulations=min(results.settings.simulations, 999),
                seed=results.settings.seed,
                pairing_mode=results.settings.pairing_mode,
                perm_stat=results.settings.perm_stat,
                all_text=t("all"),
            )
            if len(series_list) >= 2
            else None
        )
        om = comp.omnibus if comp else None
        fig = group_figure(
            results, series_list, t, band=active_band, show_samples=show_samples, omnibus=om
        )
        slug = f"{idx:02d}_{_safe_slug(preset_title)}"
        for fmt in formats:
            target = out_dir / f"{slug}.{fmt}"
            save_figure(fig, target, preset_title)
            written.append(target)
        if comp and comp.contrasts:
            contrast_dir.mkdir(parents=True, exist_ok=True)
            if om is not None and om.k_groups >= 3:
                om_fig = omnibus_figure(results, om, t)
                om_target = contrast_dir / f"{slug}_omnibus.pdf"
                save_figure(om_fig, om_target, preset_title)
                written.append(om_target)
            for c_idx, c in enumerate(comp.contrasts, start=1):
                c_fig = contrast_figure(results, c, t)
                c_slug = _safe_slug(f"{slug}_{c_idx}_{c.a}_vs_{c.b}")
                c_target = contrast_dir / f"{c_slug}.pdf"
                save_figure(c_fig, c_target, f"{c.a} vs {c.b}")
                written.append(c_target)
    return written


def interval_strings(radii, side):
    spans = []
    for i, direction in enumerate(side):
        if not direction:
            continue
        if i and side[i - 1] == direction:
            spans[-1][2] = float(radii[i])
        else:
            spans.append(["above" if direction == 1 else "below", float(radii[i]), float(radii[i])])
    return spans


def data_tables(results):
    r, settings = results.radii, results.settings
    unit = results.data.calibration.unit
    curves, summary, local = [], [], []
    for p in results.patterns:
        common = dict(
            sample_id=p.sample,
            roi_id=p.roi,
            group=p.group,
            cell_type=p.label,
            n_points=p.count,
            n_targets=p.target_count,
            n_locked=p.locked_count,
            area=p.window.area,
            correction=p.window.correction,
            unit=unit,
        )
        env = p.envelope
        with np.errstate(invalid="ignore"):
            l_val = np.sqrt(np.maximum(p.k, 0) / np.pi)
        curves.append(
            pd.DataFrame(
                dict(
                    **common,
                    radius=r,
                    K=p.k,
                    L=l_val,
                    L_minus_r=p.curve,
                    lower=env.lower if env else np.nan,
                    upper=env.upper if env else np.nan,
                    band_type="global_null_95" if env else "",
                    excursion=env.side if env else 0,
                    row_type="roi",
                )
            )
        )
        summary.append(
            dict(
                **common,
                row_type="roi",
                status=p.status,
                null=p.null if settings.test else "",
                p_global=env.p if env else np.nan,
                p_holm=p.p_adjusted,
                scales=json.dumps(interval_strings(r, env.side)) if env else "",
            )
        )
        if p.local is not None:
            local.append(
                pd.DataFrame(
                    dict(
                        **common,
                        x=p.xy[:, 0],
                        y=p.xy[:, 1],
                        radius=settings.local_radius,
                        neighbors_minus_expected=p.local,
                    )
                )
            )
    all_summaries = list(results.summaries) + list(results.custom_summaries)
    for s in all_summaries:
        grp_name = s.series_id or s.group
        for sample, curve in zip(s.sample_ids, s.curves):
            curves.append(
                pd.DataFrame(
                    dict(
                        sample_id=sample,
                        group=grp_name,
                        cell_type=s.label,
                        radius=r,
                        L_minus_r=curve,
                        unit=unit,
                        row_type="sample",
                    )
                )
            )
        curves.append(
            pd.DataFrame(
                dict(
                    group=grp_name,
                    cell_type=s.label,
                    radius=r,
                    L_minus_r=s.mean,
                    SD=s.sd if s.sd is not None else np.nan,
                    SE=s.se if s.se is not None else np.nan,
                    N=s.n,
                    lower=s.low,
                    upper=s.high,
                    band_type=settings.band,
                    unit=unit,
                    row_type="group",
                )
            )
        )
    active_contrasts = results.contrasts or (results.comparison.contrasts if results.comparison else [])
    if not results.contrasts and results.comparison and results.comparison.omnibus is not None:
        om = results.comparison.omnibus
        summary.append(
            dict(
                group=" vs ".join(om.series_names),
                cell_type="",
                row_type="omnibus",
                design=om.design,
                status=om.envelope.status,
                p_global=om.p_global,
                p_holm=om.p_global,
                k_groups=om.k_groups,
                parametric_test=om.parametric_test,
                parametric_stat=om.parametric_stat,
                parametric_p=om.parametric_p,
                nonparametric_test=om.nonparametric_test,
                nonparametric_stat=om.nonparametric_stat,
                nonparametric_p=om.nonparametric_p,
                scales=json.dumps(om.significant_spans),
            )
        )
    for c in active_contrasts:
        name = f"{c.a} − {c.b}"
        curves.append(
            pd.DataFrame(
                dict(
                    group=name,
                    cell_type=c.label,
                    radius=r,
                    L_minus_r=c.effect,
                    lower=c.envelope.lower,
                    upper=c.envelope.upper,
                    band_type=f"global_permutation_95_{settings.perm_stat}",
                    unit=unit,
                    row_type="contrast",
                    excursion=c.envelope.side,
                )
            )
        )
        summary.append(
            dict(
                group=name,
                cell_type=c.label,
                row_type="contrast",
                design=c.design,
                status=c.envelope.status,
                p_global=c.envelope.p,
                p_holm=c.p_adjusted,
                N_A=c.n_a,
                N_B=c.n_b,
                N_paired=c.n_paired,
                mean_diff=c.mean_diff_scalar,
                parametric_test=c.parametric_test,
                parametric_stat=c.parametric_stat,
                parametric_p=c.parametric_p,
                parametric_p_holm=c.parametric_p_holm,
                nonparametric_test=c.nonparametric_test,
                nonparametric_stat=c.nonparametric_stat,
                nonparametric_p=c.nonparametric_p,
                nonparametric_p_holm=c.nonparametric_p_holm,
                scales=json.dumps(interval_strings(r, c.envelope.side)),
            )
        )
    metadata = {key: value for key, value in vars(settings).items() if key not in {"custom", "intensity"}}
    metadata.update(
        version=VERSION,
        scipy=scipy.__version__,
        numpy=np.__version__,
        shapely=shapely.__version__,
        matplotlib=matplotlib.__version__,
        pandas=pd.__version__,
        alpha=.05,
        sources=results.data.sources,
        calibration=asdict(results.data.calibration),
        import_settings=results.data.import_settings,
        source_roi=results.source_roi,
        source_intensity=results.source_intensity,
        effective_r_min=float(r[0]),
        effective_r_max=float(r[-1]),
        effective_r_count=int(len(r)),
        effective_r_step=float(r[1] - r[0]) if len(r) > 1 else np.nan,
        global_test="extreme rank length global envelope, conservative ties, SeedSequence per ROI",
        inference_family="ROI tests and pairwise contrasts Holm adjusted within separate families",
        sample_weight="equal ROI means within a sample, then equal sample means within a group",
    )
    if settings.intensity is not None:
        metadata["intensity_grid"] = settings.intensity.weights.tolist()
    windows = {}
    for p in results.patterns:
        windows.setdefault((p.sample, p.roi, p.label), p.window)
    return {
        "curves": pd.concat(curves, ignore_index=True),
        "summary": pd.DataFrame(summary),
        "metadata": pd.DataFrame((key, json.dumps(value, ensure_ascii=False)) for key, value in metadata.items()).set_axis(
            ["key", "value"], axis=1
        ),
        "qc": results.qc.reindex(columns=[*ISSUE_COLUMNS, "label"]),
        "roi": pd.DataFrame(
            dict(sample_id=sample, roi_id=roi, cell_type=lbl, area=w.area, mode=w.mode, correction=w.correction, wkt=w.geometry.wkt)
            for (sample, roi, lbl), w in windows.items()
        ),
        "local": pd.concat(local, ignore_index=True) if local else pd.DataFrame(),
    }


def export_data(results, filename):
    from tempfile import TemporaryDirectory
    from zipfile import ZipFile, ZIP_DEFLATED
    from openpyxl.styles import Font

    path = Path(filename)
    tables = data_tables(results)
    if path.suffix.lower() not in {".xlsx", ".zip"}:
        raise ValueError("export_extension")
    with TemporaryDirectory(dir=path.parent) as tmp:
        target = Path(tmp) / path.name
        if path.suffix.lower() == ".xlsx":
            with pd.ExcelWriter(target, engine="openpyxl") as writer:
                for name, table in tables.items():
                    table.to_excel(writer, sheet_name=name, index=False)
                    sheet = writer.sheets[name]
                    for cell in sheet[1]:
                        cell.font = Font(bold=True)
                    sheet.freeze_panes = "A2"
                    sheet.auto_filter.ref = sheet.dimensions
                    for row in sheet.iter_rows(min_row=2):
                        for cell in row:
                            if cell.data_type == "f" or cell.data_type == "e":
                                cell.data_type = "s"
        else:
            with ZipFile(target, "w", compression=ZIP_DEFLATED) as archive:
                for name, table in tables.items():
                    archive.writestr(f"{name}.csv", table.to_csv(index=False).encode("utf-8-sig"))
        target.replace(path)
