from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
import json

import matplotlib
import numpy as np
import pandas as pd
import scipy
import shapely

from ripley_analysis import (boundary_points, centered_l, check_cancel, global_envelope, inside,
                             local_neighbors, make_window, ripley, uniform)
from spatial_data import ISSUE_COLUMNS, issue_rows


VERSION = "1.0.0"
ALL = ""


@dataclass(frozen=True)
class Settings:
    roi_mode: str = "infer"
    rectangle: tuple | None = None
    custom: dict = field(default_factory=dict)
    hull_ratio: float = .35
    r_min: float | None = None
    r_max: float | None = None
    n_radii: int = 60
    simulations: int = 999
    seed: int = 2406
    test: bool = True
    marks: str = "all"
    from_type: str = ""
    to_type: str = ""
    intensity: object = None
    band: str = "sd"
    contrast: bool = False
    paired: bool = False
    group_a: str = ""
    group_b: str = ""
    local_radius: float | None = None

    def validate(self):
        if (not 2 <= self.n_radii <= 500 or not 19 <= self.simulations <= 9999
                or not 0 <= self.seed < 2**64 or not 0 <= self.hull_ratio <= 1):
            raise ValueError("invalid_settings")
        distances = [d for d in (self.r_min, self.r_max, self.local_radius) if d is not None]
        if any(not np.isfinite(d) or d <= 0 for d in distances):
            raise ValueError("invalid_radii")
        if self.marks == "cross" and (not self.from_type or not self.to_type or self.from_type == self.to_type):
            raise ValueError("choose_two_types")
        if self.intensity is not None and (self.roi_mode != "rectangle" or self.marks == "cross"):
            raise ValueError("intensity_rectangle_only")
        if self.local_radius is not None and (self.intensity is not None or self.marks == "cross"):
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


def finite_mean(values, axis):
    values = np.asarray(values, dtype=float)
    count = np.isfinite(values).sum(axis=axis)
    total = np.nansum(values, axis=axis)
    return np.divide(total, count, out=np.full(np.shape(total), np.nan), where=count > 0)


def adjust_holm(records, get_p):
    tested = sorted(((get_p(row), row) for row in records if np.isfinite(get_p(row))), key=lambda x: x[0])
    adjusted = 0.
    for rank, (p, row) in enumerate(tested):
        adjusted = min(1., max(adjusted, (len(tested) - rank) * p))
        row.p_adjusted = adjusted


def prepare(data, settings, stop):
    fields, issues = [], [data.issues]
    grouped = dict(tuple(data.points.groupby(["sample_id", "roi_id"], sort=False)))
    # Preserve source fields in the quality-control log even if every row failed validation.
    for _, sample, roi in data.fields:
        if (sample, roi) not in grouped:
            issues.append(pd.DataFrame([dict(sample_id=sample, roi_id=roi, reason="no_valid_rows")]))
    for (sample, roi), frame in grouped.items():
        check_cancel(stop)
        xy = frame[["x", "y"]].to_numpy()
        try:
            window = make_window(settings.roi_mode, xy, data.bounds.get((sample, roi), settings.rectangle),
                                 settings.custom.get((sample, roi), settings.custom.get(None)), settings.hull_ratio)
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


def compute_pattern(frame, window, label, radii, settings, rng, stop):
    xy = frame[["x", "y"]].to_numpy()
    marks = frame.cell_type.to_numpy()
    cross = settings.marks == "cross"
    if cross:
        a, b = settings.from_type, settings.to_type
        keep = np.isin(marks, (a, b))
        xy, marks = xy[keep], marks[keep]
        observed, targets = xy[marks == a], xy[marks == b]
    else:
        if settings.marks == "each":
            keep = marks == label
            xy, marks = xy[keep], marks[keep]
        observed, targets = xy, None
    k = ripley(observed, window, radii, targets, settings.intensity, stop)
    curve = centered_l(k, radii)
    locked = (boundary_points(window, observed) if window.mode == "locked" and not cross
              else np.zeros(len(observed), dtype=bool))
    null = ("random_label" if cross else "fixed_intensity" if settings.intensity is not None
            else "locked_csr" if window.mode == "locked" else "csr")
    qc = []
    if len(observed) < (1 if cross else 2) or (cross and len(targets) == 0) or not np.isfinite(curve).any():
        qc.append("insufficient")
    if not cross and len(np.unique(observed, axis=0)) < len(observed):
        qc.append("csr_coincident")
    if window.mode == "locked" and not cross and len(observed) - locked.sum() < 3:
        qc.append("few_interior")
    if window.mode == "locked" and not cross and settings.marks == "each":
        qc.append("locked_all_only")
    if window.mode == "infer" and not cross:
        qc.append("estimated_roi")
    if cross and (len(observed) < 2 or len(targets) < 2):
        qc.append("too_few_for_test")
    if not cross and len(observed) < 4:
        qc.append("too_few_for_test")
    envelope = None
    if settings.test and not qc:
        sims = np.empty((settings.simulations + 1, len(radii)))
        sims[0] = curve
        for i in range(1, len(sims)):
            check_cancel(stop)
            if cross:
                labels = rng.permutation(marks)
                simulated_k = ripley(xy[labels == a], window, radii, xy[labels == b], stop=stop)
            else:
                count = len(observed) - int(locked.sum())
                simulated = (settings.intensity.sample(count, window, rng) if settings.intensity is not None
                             else uniform(window, count, rng, stop))
                if locked.any():
                    simulated = np.vstack((observed[locked], simulated))
                simulated_k = ripley(simulated, window, radii, intensity=settings.intensity, stop=stop)
            sims[i] = centered_l(simulated_k, radii)
        envelope = global_envelope(sims)
        if envelope.status == "insufficient":
            qc.append("no_common_support")
    status = envelope.status if envelope else ("insufficient" if "insufficient" in qc else "descriptive")
    local = None
    if settings.local_radius is not None and not cross and settings.intensity is None:
        local = local_neighbors(observed, window, settings.local_radius, stop)
    return xy, marks, len(observed), None if targets is None else len(targets), int(locked.sum()), k, curve, envelope, status, null, qc, local


def summarize(patterns, settings, rng, stop):
    by_sample, by_group = {}, {}
    for pattern in patterns:
        if np.isfinite(pattern.curve).any():
            key = (pattern.group, pattern.label, pattern.sample)
            by_sample.setdefault(key, []).append(pattern.curve)
    for (group, label, sample), curves in by_sample.items():
        by_group.setdefault((group, label), {})[sample] = finite_mean(curves, axis=0)
    out = []
    for (group, label), sample_curves in by_group.items():
        curves = np.stack(list(sample_curves.values()))
        n = np.isfinite(curves).sum(axis=0)
        mean = finite_mean(curves, axis=0)
        low, high = np.full(len(mean), np.nan), np.full(len(mean), np.nan)
        if settings.band == "sd":
            sd = np.sqrt(np.divide(np.nansum((curves - mean)**2, axis=0), n - 1,
                                   out=np.full(len(mean), np.nan), where=n > 1))
            low, high = mean - sd, mean + sd
        elif settings.band == "ci":
            # Draw whole sample curves, retaining dependence across distance.
            supported = (n >= 3) & (n == len(curves))
            if supported.any():
                bootstrap = np.empty((2000, supported.sum()))
                complete = curves[:, supported]
                for i in range(len(bootstrap)):
                    check_cancel(stop)
                    bootstrap[i] = complete[rng.integers(len(curves), size=len(curves))].mean(axis=0)
                low[supported], high[supported] = np.quantile(bootstrap, [.025, .975], axis=0)
        out.append(Summary(group, label, list(sample_curves), curves, mean, n, low, high))
    return out


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
            if ("" in set(pa) or "" in set(pb) or not pa.is_unique or not pb.is_unique or set(pa) != set(pb)):
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
        contrasts.append(Contrast(a.group, b.group, a.label, effects[0], envelope, na, nb))
    if not contrasts:
        raise ValueError("contrast_needs_samples")
    adjust_holm(contrasts, lambda item: item.envelope.p)
    return contrasts


def run_analysis(data, settings, progress=lambda done, total: None, stop=None):
    settings.validate()
    fields, issues = prepare(data, settings, stop)
    rng = np.random.default_rng(settings.seed)
    auto_max = min(window.default_rmax for _, _, _, window in fields)
    r_max = settings.r_max or auto_max
    r_min = settings.r_min or r_max / settings.n_radii
    if not 0 < r_min < r_max or r_max > auto_max * (1 + 1e-12):
        raise ValueError("radius_range", f"{auto_max:.6g}")
    if settings.local_radius is not None and settings.local_radius > r_max:
        raise ValueError("local_radius_range", f"{r_max:.6g}")
    radii = np.linspace(r_min, r_max, settings.n_radii)
    labels = (sorted(set(data.points.cell_type) - {""}) if settings.marks == "each"
              else [f"{settings.from_type} → {settings.to_type}"] if settings.marks == "cross" else [ALL])
    if not labels:
        raise ValueError("missing_type")
    patterns = []
    total = len(fields) * len(labels)
    progress(0, total)
    sample_groups = data.points.drop_duplicates("sample_id").set_index("sample_id").group
    for sample, roi, frame, window in fields:
        for label in labels:
            check_cancel(stop)
            p = Pattern(sample, roi, sample_groups[sample], label, window,
                        *compute_pattern(frame, window, label, radii, settings, rng, stop))
            for reason in p.qc:
                issues.append(pd.DataFrame([dict(sample_id=sample, roi_id=roi, reason=reason, label=label)]))
            patterns.append(p)
            progress(len(patterns), total)
    adjust_holm(patterns, lambda item: item.envelope.p if item.envelope else np.nan)
    summaries = summarize(patterns, settings, rng, stop)
    contrasts = compare_groups(data, summaries, settings, rng, stop) if settings.contrast else []
    return Results(data, settings, radii, patterns, summaries, contrasts, pd.concat(issues, ignore_index=True))


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
        common = dict(sample_id=p.sample, roi_id=p.roi, group=p.group, cell_type=p.label,
                      n_points=p.count, n_targets=p.target_count, n_locked=p.locked_count,
                      area=p.window.area, correction=p.window.correction, unit=unit)
        env = p.envelope
        curves.append(pd.DataFrame(dict(**common, radius=r, K=p.k, L=np.sqrt(p.k / np.pi), L_minus_r=p.curve,
                                        lower=env.lower if env else np.nan, upper=env.upper if env else np.nan,
                                        band_type="global_null_95" if env else "",
                                        excursion=env.side if env else 0, row_type="roi")))
        summary.append(dict(**common, status=p.status, null=p.null if settings.test else "",
                            p_global=env.p if env else np.nan, p_holm=p.p_adjusted,
                            scales=json.dumps(interval_strings(r, env.side)) if env else ""))
        if p.local is not None:
            local.append(pd.DataFrame(dict(**common, x=p.xy[:, 0], y=p.xy[:, 1], radius=settings.local_radius,
                                           neighbors_minus_expected=p.local)))
    for s in results.summaries:
        for sample, curve in zip(s.sample_ids, s.curves):
            curves.append(pd.DataFrame(dict(sample_id=sample, group=s.group, cell_type=s.label, radius=r,
                                            L_minus_r=curve, unit=unit, row_type="sample")))
        curves.append(pd.DataFrame(dict(group=s.group, cell_type=s.label, radius=r, L_minus_r=s.mean,
                                        N=s.n, lower=s.low, upper=s.high, band_type=settings.band,
                                        unit=unit, row_type="group")))
    for c in results.contrasts:
        name = f"{c.a} − {c.b}"
        curves.append(pd.DataFrame(dict(group=name, cell_type=c.label, radius=r, L_minus_r=c.effect,
                                        lower=c.envelope.lower, upper=c.envelope.upper,
                                        band_type="global_permutation_95", unit=unit, row_type="contrast",
                                        excursion=c.envelope.side)))
        summary.append(dict(group=name, cell_type=c.label, status=c.envelope.status, p_global=c.envelope.p,
                            p_holm=c.p_adjusted, N_A=c.n_a, N_B=c.n_b,
                            scales=json.dumps(interval_strings(r, c.envelope.side))))
    metadata = {key: value for key, value in vars(settings).items() if key not in {"custom", "intensity"}}
    metadata.update(version=VERSION, scipy=scipy.__version__, numpy=np.__version__, shapely=shapely.__version__,
                    matplotlib=matplotlib.__version__, pandas=pd.__version__, alpha=.05,
                    sources=results.data.sources, calibration=asdict(results.data.calibration),
                    import_settings=results.data.import_settings, source_roi=results.source_roi,
                    source_intensity=results.source_intensity, effective_r_min=float(r[0]),
                    effective_r_max=float(r[-1]), global_test="two-sided extreme rank length, conservative ties",
                    inference_family="ROI tests and contrasts Holm adjusted within separate families",
                    sample_weight="equal ROI means within a sample, then equal sample means within a group")
    if settings.intensity is not None:
        metadata["intensity_grid"] = settings.intensity.weights.tolist()
    windows = {(p.sample, p.roi): p.window for p in results.patterns}
    return {
        "curves": pd.concat(curves, ignore_index=True),
        "summary": pd.DataFrame(summary),
        "metadata": pd.DataFrame((key, json.dumps(value, ensure_ascii=False)) for key, value in metadata.items())
                      .set_axis(["key", "value"], axis=1),
        "qc": results.qc.reindex(columns=[*ISSUE_COLUMNS, "label"]),
        "roi": pd.DataFrame(dict(sample_id=sample, roi_id=roi, area=w.area, mode=w.mode,
                                  wkt=w.geometry.wkt) for (sample, roi), w in windows.items()),
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
