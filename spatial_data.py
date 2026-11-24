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
    "x": ("x", "position_x", "x_coordinate", "x_coord", "centroid_x", "xm", "x座標"),
    "y": ("y", "position_y", "y_coordinate", "y_coord", "centroid_y", "ym", "y座標"),
    "sample_id": ("sample_id", "sample", "sample_no", "sample_number", "specimen",
                  "subject_id", "animal_id", "樣本", "樣品", "樣品編號", "樣本編號", "樣本id", "樣品id"),
    "roi_id": ("roi_id", "roi", "field", "field_id", "視野", "視野編號"),
    "cell_type": ("cell_type", "point_type", "point_name", "type", "label", "class", "category", "name",
                  "細胞類型", "點類型", "點的名稱", "點名稱"),
    "group": ("group", "condition", "treatment", "experiment", "experimental_treatment",
              "組別", "處理", "實驗處理", "實驗組別"),
    "pair_id": ("pair_id", "pair", "配對", "配對編號"),
    "block_id": ("block_id", "batch", "block", "批次", "區塊"),
}
MEASUREMENT_KEYS = {"area", "mean", "stddev", "std", "min", "max", "intden", "rawintden",
                    "perim", "circ", "ar", "round", "solidity", "feret", "median", "mode", "kurt", "skew"}
RECT_COLUMNS = ("xmin", "ymin", "xmax", "ymax")
ISSUE_COLUMNS = ("source", "row", "sample_id", "roi_id", "reason")


@dataclass(frozen=True)
class ReadOptions:
    header: bool = True
    delimiter: str = "auto"
    encoding: str = "auto"
    decimal: str = "."
    sheet: str | int = 0
    sheets: tuple[str | int, ...] | None = None
    sheet_as_group: bool = False
    unstack_blocks: bool = False


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


def _norm_key(text):
    return re.sub(r"[^\w]+", "_", str(text).strip().casefold()).strip("_")


def detect_block_width(raw, header=True):
    """Return horizontal repeating block width if row 0 repeats identical X/Y sub-tables."""
    if not header or raw is None or raw.empty or raw.shape[1] < 6:
        return None
    keys = [_norm_key(c) for c in raw.iloc[0]]
    x_pos = [i for i, k in enumerate(keys) if k in ALIASES["x"]]
    y_pos = [i for i, k in enumerate(keys) if k in ALIASES["y"]]
    if len(x_pos) < 2 or len(x_pos) != len(y_pos):
        return None
    width = x_pos[1] - x_pos[0]
    if width < 3 or any(d != width for d in np.diff(x_pos)) or any(d != width for d in np.diff(y_pos)):
        return None
    if x_pos[0] >= width or y_pos[0] >= width:
        return None
    total_used = len(x_pos) * width
    if raw.shape[1] < total_used:
        return None
    if raw.shape[1] > total_used:
        tail_headers = [str(c).strip() for c in raw.iloc[0, total_used:]]
        if any(tail_headers):
            return None
    return int(width)


def unstack_table(raw, header=True, width=None):
    """Stack horizontally repeated equal-width column blocks into one vertical table."""
    if raw is None or raw.empty:
        return raw
    width = width or detect_block_width(raw, header)
    if not width:
        return raw
    if header:
        keys0 = [_norm_key(c) for c in raw.iloc[0]]
        n_blocks = sum(1 for i in range(0, raw.shape[1], width)
                       if i < len(keys0) and any(_norm_key(c) in ALIASES["x"] for c in raw.iloc[0, i:i + width]))
    else:
        n_blocks = raw.shape[1] // width
    if n_blocks < 2:
        return raw
    keys = [_norm_key(c) for c in raw.iloc[0, :width]] if header else []
    x_idx = next((i for i, k in enumerate(keys) if k in ALIASES["x"]), None)
    y_idx = next((i for i, k in enumerate(keys) if k in ALIASES["y"]), None)
    parts = []
    if header:
        top = raw.iloc[:1, :width].copy()
        top.columns = range(width)
        parts.append(top)
    start = 1 if header else 0
    for b in range(n_blocks):
        block = raw.iloc[start:, b * width:(b + 1) * width].copy()
        block.columns = range(width)
        stripped = block.astype(str).apply(lambda s: s.str.strip())
        non_empty = (stripped != "").any(axis=1)
        if x_idx is not None and y_idx is not None:
            has_coord = (stripped.iloc[:, x_idx] != "") | (stripped.iloc[:, y_idx] != "")
            non_empty = non_empty & has_coord
        parts.append(block.loc[non_empty])
    return pd.concat(parts, axis=0) if parts else raw.iloc[:, :width]


def excel_sheets(path):
    with pd.ExcelFile(path) as book:
        return book.sheet_names


def default_excel_sheets(path, header=True):
    """Inspect workbook sheets and return compatible point-table sheet names."""
    sheets = excel_sheets(path)
    if len(sheets) <= 1:
        return tuple(sheets)
    valid = []
    with pd.ExcelFile(path) as book:
        for name in sheets:
            raw = pd.read_excel(book, sheet_name=name, header=None, dtype=str, keep_default_na=False, nrows=8)
            if raw.empty or raw.shape[1] < 2:
                continue
            bw = detect_block_width(raw, header)
            sample_raw = unstack_table(raw, header, bw) if bw else raw
            labels = column_labels(sample_raw, header)
            inferred = infer_mapping(labels, sample_raw, header)
            if "x" in inferred and "y" in inferred:
                valid.append((name, sample_raw.shape[1]))
    if not valid:
        return (sheets[0],)
    target_cols = valid[0][1]
    matched = tuple(name for name, cols in valid if cols == target_cols)
    return matched or (valid[0][0],)


def read_table(path, options, limit=None, sheet=None):
    path = Path(path)
    chosen_sheet = options.sheet if sheet is None else sheet
    if chosen_sheet is None and options.sheets:
        chosen_sheet = options.sheets[0]
    row_limit = None if (options.unstack_blocks and limit is not None) else limit
    args = dict(header=None, dtype=str, keep_default_na=False, nrows=row_limit)
    if path.suffix.lower() in (".xlsx", ".xls"):
        if options.unstack_blocks and limit is not None:
            args["nrows"] = max(limit, 80)
        try:
            raw = pd.read_excel(path, sheet_name=chosen_sheet, **args)
        except ImportError as error:
            raise ValueError("missing_xlrd") from error
        except ValueError as error:
            raise ValueError("missing_sheet", str(chosen_sheet)) from error
        if options.unstack_blocks:
            raw = unstack_table(raw, options.header)
            if limit is not None:
                raw = raw.iloc[:limit]
        return raw
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
            raw = pd.read_csv(path, sep=delimiter, encoding=encoding, skip_blank_lines=False, **args)
            if options.unstack_blocks:
                raw = unstack_table(raw, options.header)
                if limit is not None:
                    raw = raw.iloc[:limit]
            return raw
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError as error:
            raise ValueError("empty_file") from error
    raise ValueError("invalid_encoding")


def column_labels(raw, header):
    return ([str(c).strip() for c in raw.iloc[0]] if header and len(raw)
            else [f"#{i + 1}" for i in range(raw.shape[1])])


def infer_mapping(labels, raw=None, header=True, decimal="."):
    keys = [_norm_key(c) for c in labels]
    mapping = {}
    for field in COLUMNS:
        exact = [i for i, key in enumerate(keys) if key in ALIASES.get(field, (field,))]
        if len(exact) == 1:
            mapping[field] = exact[0]
    if not header and raw is not None and "x" not in mapping and "y" not in mapping:
        numeric = [i for i in range(raw.shape[1]) if np.isfinite(numbers(raw.iloc[:, i], decimal)).mean() >= .9]
        if len(numeric) == 2:
            mapping.update(x=numeric[0], y=numeric[1])
    # Support ImageJ-style tables where columns A-B hold repeated point-type / sample-ID values on row 1.
    if (header and raw is not None and len(raw) >= 2 and "x" in mapping and "y" in mapping
            and ("cell_type" not in mapping or "sample_id" not in mapping)):
        used = set(mapping.values())
        candidates = []
        for idx, (label, key) in enumerate(zip(labels, keys)):
            if idx in used or key in MEASUREMENT_KEYS or not label:
                continue
            col_vals = raw.iloc[1:, idx].astype(str).str.strip()
            non_empty = col_vals[col_vals != ""]
            if non_empty.empty:
                continue
            if np.isfinite(numbers(non_empty, decimal)).mean() >= .8:
                continue
            if label.strip() in set(non_empty.iloc[:20]):
                candidates.append((idx, non_empty))
        if len(candidates) >= 2 and "cell_type" not in mapping and "sample_id" not in mapping:
            (i0, v0), (i1, v1) = candidates[0], candidates[1]
            s0 = v0.str.contains(r"(?:^|[\s_-])\d+$", regex=True).mean() >= .7
            s1 = v1.str.contains(r"(?:^|[\s_-])\d+$", regex=True).mean() >= .7
            if s1 and not s0:
                mapping["cell_type"], mapping["sample_id"] = i0, i1
            elif s0 and not s1:
                mapping["sample_id"], mapping["cell_type"] = i0, i1
            else:
                mapping["cell_type"], mapping["sample_id"] = i0, i1
        elif len(candidates) == 1:
            idx, vals = candidates[0]
            if vals.str.contains(r"(?:^|[\s_-])\d+$", regex=True).mean() >= .7 and "sample_id" not in mapping:
                mapping["sample_id"] = idx
            elif "cell_type" not in mapping:
                mapping["cell_type"] = idx
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
    expected_coord_labels = None
    sources_used = []
    for path in paths:
        is_excel = path.suffix.lower() in (".xlsx", ".xls")
        if is_excel and options.sheets:
            sheet_list = list(options.sheets)
        elif is_excel:
            sheet_list = [options.sheet]
        else:
            sheet_list = [None]
        multi_sheet = len(sheet_list) > 1
        for sh in sheet_list:
            raw = read_table(path, options, sheet=sh)
            if raw.empty:
                if multi_sheet:
                    continue
                raise ValueError("empty_file")
            labels = column_labels(raw, options.header)
            if any(i < 0 or i >= len(labels) for i in mapping.values()):
                raise ValueError("different_schema")
            coord_keys = [k for k in ("x", "y", *RECT_COLUMNS) if k in mapping]
            if options.header:
                current_coords = {k: _norm_key(labels[mapping[k]]) for k in coord_keys}
                if expected_coord_labels is None:
                    expected_coord_labels = current_coords
                elif current_coords != expected_coord_labels:
                    raise ValueError("different_schema")
            xy_headers = (labels[mapping["x"]].strip().casefold(), labels[mapping["y"]].strip().casefold())
            if options.header:
                raw = raw.iloc[1:]
            if raw.empty:
                continue
            x_str = raw.iloc[:, mapping["x"]].astype(str).str.strip()
            y_str = raw.iloc[:, mapping["y"]].astype(str).str.strip()
            blank_xy = (x_str == "") & (y_str == "")
            repeated_hdr = (options.header
                            & (x_str.str.casefold() == xy_headers[0])
                            & (y_str.str.casefold() == xy_headers[1]))
            keep_rows = ~(blank_xy | repeated_hdr)
            raw = raw.loc[keep_rows]
            if raw.empty:
                continue
            source_tag = f"{path} [{sh}]" if multi_sheet else str(path)
            sources_used.append(source_tag)
            default_sid = (str(sh).strip() if (multi_sheet and len(paths) == 1)
                           else f"{path.stem}_{sh}" if multi_sheet else path.stem)
            default_grp = str(sh).strip() if (options.sheet_as_group and sh is not None) else ""
            frame = pd.DataFrame({"source": source_tag, "row": raw.index.to_numpy() + 1})
            for key, default in {"sample_id": default_sid, "roi_id": "1", "cell_type": "",
                                 "group": default_grp, "pair_id": "", "block_id": ""}.items():
                if key in mapping:
                    vals = raw.iloc[:, mapping[key]].astype(str).str.strip().to_numpy()
                    if key == "group" and default_grp:
                        vals = np.where(vals == "", default_grp, vals)
                    frame[key] = vals
                else:
                    frame[key] = default
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
    # Check within-source metadata consistency first; disambiguate sample_id across sheets if groups differ.
    for _, src_grp in data.groupby(["source", "sample_id"], sort=False):
        if any(src_grp[key].nunique() != 1 for key in ("group", "pair_id", "block_id")):
            raise ValueError("sample_metadata_conflict")
    colliding = {sid for sid, grp in data.groupby("sample_id", sort=False) if grp["group"].nunique() > 1}
    source_group = {}
    if colliding:
        remap_mask = data.sample_id.isin(colliding) & (data.group != "")
        source_group = dict(data.drop_duplicates(["source", "sample_id"])
                            .set_index(["source", "sample_id"])["group"])
        data.loc[remap_mask, "sample_id"] = data.loc[remap_mask, "group"] + " " + data.loc[remap_mask, "sample_id"]
        fields = {(src, f"{source_group.get((src, sid), '')} {sid}".strip() if sid in colliding else sid, roi)
                  for src, sid, roi in fields}
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
    issue_df = pd.concat(issues, ignore_index=True) if issues else pd.DataFrame(columns=ISSUE_COLUMNS)
    if colliding and not issue_df.empty:
        mask_iss = issue_df["sample_id"].isin(colliding)
        if mask_iss.any():
            issue_df.loc[mask_iss, "sample_id"] = [
                f"{source_group.get((src, sid), '')} {sid}".strip()
                for src, sid in zip(issue_df.loc[mask_iss, "source"], issue_df.loc[mask_iss, "sample_id"])
            ]
    return Dataset(data.reset_index(drop=True), issue_df,
                   tuple(sources_used or (str(p) for p in paths)), calibration, tuple(sorted(fields)), bounds,
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
