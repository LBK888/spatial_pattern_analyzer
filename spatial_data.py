from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from PIL import Image
import shapely
from shapely.affinity import scale
from shapely.geometry import shape


COLUMNS = ("x", "y", "sample_id", "roi_id", "cell_type", "group", "pair_id", "block_id",
           "xmin", "ymin", "xmax", "ymax")
ALIASES = {
    "x": ("x", "position_x", "x_coordinate", "x_coord", "centroid_x", "x座標"),
    "y": ("y", "position_y", "y_coordinate", "y_coord", "centroid_y", "y座標"),
    "sample_id": ("sample_id", "sample", "subject_id", "animal_id", "樣本"),
    "roi_id": ("roi_id", "roi", "field", "field_id", "視野"),
    "cell_type": ("cell_type", "type", "label", "class", "category", "細胞類型"),
    "group": ("group", "condition", "treatment", "組別"),
    "pair_id": ("pair_id", "pair", "配對"),
    "block_id": ("block_id", "batch", "block", "批次"),
}
RECT_COLUMNS = ("xmin", "ymin", "xmax", "ymax")
ISSUE_COLUMNS = ("source", "row", "sample_id", "roi_id", "reason")


@dataclass(frozen=True)
class ReadOptions:
    header: bool = True
    delimiter: str = "auto"
    encoding: str = "auto"
    decimal: str = "."
    sheet: str | int = 0


@dataclass(frozen=True)
class Calibration:
    x: float = 1.0
    y: float = 1.0
    unit: str = "units"
    y_down: bool = False

    @property
    def factors(self):
        return np.array([self.x, self.y])


@dataclass(frozen=True)
class Dataset:
    points: pd.DataFrame
    issues: pd.DataFrame
    sources: tuple[str, ...]
    calibration: Calibration
    fields: tuple[tuple[str, str, str], ...]
    bounds: dict = field(default_factory=dict)
    import_settings: dict = field(default_factory=dict)


def excel_sheets(path):
    with pd.ExcelFile(path) as book:
        return book.sheet_names


def read_table(path, options, limit=None):
    path = Path(path)
    args = dict(header=None, dtype=str, keep_default_na=False, nrows=limit)
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path, sheet_name=options.sheet, **args)
    if path.suffix.lower() not in (".csv", ".tsv", ".txt"):
        raise ValueError("unsupported_file")
    encodings = ("utf-8-sig", "cp950", "utf-16") if options.encoding == "auto" else (options.encoding,)
    with path.open("rb") as stream:
        if stream.read(2) in (b"\xff\xfe", b"\xfe\xff") and options.encoding == "auto":
            encodings = ("utf-16", "utf-8-sig", "cp950")
    for encoding in encodings:
        try:
            delimiter = options.delimiter
            if delimiter == "auto":
                with path.open(encoding=encoding, newline="") as stream:
                    sample = stream.read(8192)
                if not sample.strip():
                    raise ValueError("empty_file")
                if path.suffix.lower() == ".tsv":
                    delimiter = "\t"
                else:
                    try:
                        delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
                    except csv.Error:
                        delimiter = ","
            return pd.read_csv(path, sep=delimiter, encoding=encoding, skip_blank_lines=False, **args)
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError as error:
            raise ValueError("empty_file") from error
    raise ValueError("invalid_encoding")


def column_labels(raw, header):
    return ([str(c).strip() for c in raw.iloc[0]] if header and len(raw)
            else [f"#{i + 1}" for i in range(raw.shape[1])])


def infer_mapping(labels, raw=None, header=True, decimal="."):
    keys = [re.sub(r"[^\w]+", "_", c.strip().casefold()).strip("_") for c in labels]
    mapping = {}
    for field in COLUMNS:
        exact = [i for i, key in enumerate(keys) if key in ALIASES.get(field, (field,))]
        if len(exact) == 1:
            mapping[field] = exact[0]
    if not header and raw is not None and "x" not in mapping and "y" not in mapping:
        numeric = [i for i in range(raw.shape[1]) if np.isfinite(numbers(raw.iloc[:, i], decimal)).mean() >= .9]
        if len(numeric) == 2:
            mapping.update(x=numeric[0], y=numeric[1])
    return mapping


def numbers(values, decimal):
    text = values.astype(str).str.strip()
    if decimal == ",":
        text = text.where(~text.str.contains(".", regex=False), "").str.replace(",", ".", regex=False)
    return pd.to_numeric(text, errors="coerce").to_numpy(dtype=float)


def load_dataset(paths, options, mapping, calibration, duplicates="keep"):
    if mapping.get("x") is None or mapping.get("y") is None or mapping["x"] == mapping["y"]:
        raise ValueError("select_xy")
    if not np.all(np.isfinite(calibration.factors) & (calibration.factors > 0)):
        raise ValueError("invalid_scale")
    paths = tuple(Path(p) for p in paths)
    if "sample_id" not in mapping and len({p.stem for p in paths}) < len(paths):
        raise ValueError("same_file_names")
    frames, issues, fields = [], [], set()
    expected_labels = None
    for path in paths:
        raw = read_table(path, options)
        if raw.empty:
            raise ValueError("empty_file")
        labels = column_labels(raw, options.header)
        if expected_labels is None:
            expected_labels = {key: labels[i] for key, i in mapping.items() if 0 <= i < len(labels)}
        if any(i < 0 or i >= len(labels) or labels[i] != expected_labels.get(key)
               for key, i in mapping.items()):
            raise ValueError("different_schema")
        if options.header:
            raw = raw.iloc[1:]
        frame = pd.DataFrame({"source": str(path), "row": raw.index.to_numpy() + 1})
        for key, default in {"sample_id": path.stem, "roi_id": "1", "cell_type": "",
                             "group": "", "pair_id": "", "block_id": ""}.items():
            frame[key] = (raw.iloc[:, mapping[key]].astype(str).str.strip().to_numpy()
                          if key in mapping else default)
        for key, factor in zip(("x", "y", *RECT_COLUMNS), np.tile(calibration.factors, 3)):
            if key in mapping:
                with np.errstate(over="ignore", invalid="ignore"):
                    frame[key] = numbers(raw.iloc[:, mapping[key]], options.decimal) * factor
        valid_id = (frame.sample_id != "") & (frame.roi_id != "")
        finite = np.isfinite(frame.x) & np.isfinite(frame.y)
        for mask, reason in ((~valid_id, "missing_id"), (valid_id & ~finite, "invalid_coordinates")):
            issues.append(issue_rows(frame.loc[mask], reason))
        fields.update(frame.loc[valid_id, ["source", "sample_id", "roi_id"]].itertuples(index=False, name=None))
        frame = frame.loc[valid_id & finite].copy()
        if "cell_type" in mapping:
            issues.append(issue_rows(frame.loc[frame.cell_type == ""], "missing_type"))
        frames.append(frame)
    if not frames:
        raise ValueError("empty_file")
    data = pd.concat(frames, ignore_index=True)
    if data.empty:
        raise ValueError("no_valid_rows")
    for _, group in data.groupby("sample_id", sort=False):
        if any(group[key].nunique() != 1 for key in ("group", "pair_id", "block_id")):
            raise ValueError("sample_metadata_conflict")
    coordinates = ["sample_id", "roi_id", "x", "y"]
    same_record = data.duplicated(coordinates + ["cell_type"])
    coincident = data.duplicated(coordinates, keep=False)
    if duplicates == "drop":
        issues.append(issue_rows(data.loc[same_record], "duplicate_removed"))
        data = data.loc[~same_record].copy()
    issues.append(issue_rows(data.loc[coincident], "coincident_points"))
    bounds = {}
    if all(k in mapping for k in RECT_COLUMNS):
        for key, group in data.groupby(["sample_id", "roi_id"], sort=False):
            box_rows = group[list(RECT_COLUMNS)].drop_duplicates().to_numpy()
            if (len(box_rows) == 1 and np.isfinite(box_rows).all()
                    and (box_rows[0, 2:] > box_rows[0, :2]).all()):
                bounds[key] = tuple(box_rows[0])
            else:
                issues.append(issue_rows(group.iloc[:1], "invalid_bounds_columns"))
    return Dataset(data.reset_index(drop=True), pd.concat(issues, ignore_index=True),
                   tuple(str(path) for path in paths), calibration, tuple(sorted(fields)), bounds,
                   {"read_options": vars(options), "columns": mapping, "duplicates": duplicates})


def issue_rows(frame, reason):
    return frame.reindex(columns=ISSUE_COLUMNS[:-1]).assign(reason=reason)


def load_custom_roi(path, calibration, origin=(0.0, 0.0)):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".wkt":
        records = [(None, shapely.from_wkt(path.read_text(encoding="utf-8-sig")))]
    elif suffix in (".json", ".geojson"):
        source = json.loads(path.read_text(encoding="utf-8-sig"))
        features = source.get("features") if source.get("type") == "FeatureCollection" else [source]
        records = []
        for feature in features:
            if feature.get("type") != "Feature":
                records.append((None, shape(feature)))
                continue
            properties = feature.get("properties") or {}
            sample, roi = properties.get("sample_id"), properties.get("roi_id")
            if (sample is None) != (roi is None):
                raise ValueError("missing_id")
            key = None if sample is None else (str(sample), str(roi))
            records.append((key, shape(feature["geometry"])))
    elif suffix in (".png", ".tif", ".tiff"):
        with Image.open(path) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("mask_single_frame")
            mask = np.asarray(image.convert("L")) != 0
        runs = []
        for y, row in enumerate(mask):
            changes = np.flatnonzero(np.diff(np.r_[False, row, False]))
            if len(changes):
                x0, x1 = changes[::2], changes[1::2]
                runs.extend(shapely.box(x0 + origin[0], y + origin[1], x1 + origin[0], y + origin[1] + 1))
        records = [(None, shapely.union_all(runs))]
    else:
        raise ValueError("unsupported_roi")
    result = {}
    for key, geometry in records:
        if key in result:
            raise ValueError("duplicate_roi")
        result[key] = scale(geometry, xfact=calibration.x, yfact=calibration.y, origin=(0, 0))
    return result
