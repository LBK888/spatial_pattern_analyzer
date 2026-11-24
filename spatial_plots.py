from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib import font_manager
from matplotlib.text import Text
import numpy as np

from ripley_analysis import boundary_points
from spatial_workflow import with_band


PALETTE = ["#1A6FB5", "#0D8F6B", "#9B3072", "#B45309", "#6D3BBF", "#C4211C", "#0E7490", "#4D7C0F", "#BE185D"]
INK, MUTED, GRID = "#172c3c", "#526779", "#e2e9ee"


def use_cjk_font(figure, t):
    if t.lang == "zh-TW" or any(any("\u3400" <= c <= "\u9fff" for c in node.get_text())
                               for node in figure.findobj(Text)):
        for font in ("Microsoft JhengHei", "Noto Sans CJK TC", "Noto Sans TC", "Noto Sans CJK SC"):
            try:
                font_manager.findfont(font, fallback_to_default=False)
            except ValueError:
                continue
            for node in figure.findobj(Text):
                prop = node.get_fontproperties().copy()
                prop.set_family([font, "DejaVu Sans"])
                node.set_fontproperties(prop)
            break


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(color=GRID, labelcolor=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=.7)
    ax.set_axisbelow(True)
    for label in (ax.xaxis.label, ax.yaxis.label):
        label.set_color(INK)
    ax.title.set_color(INK)


def spatial_axes(ax, xy, geometry, calibration, t):
    ax.set_xlabel(t("xlabel", unit=calibration.unit))
    ax.set_ylabel(t("ylabel", unit=calibration.unit))
    if geometry is not None:
        parts = (geometry,) if geometry.geom_type == "Polygon" else geometry.geoms
        for part in parts:
            for ring in (part.exterior, *part.interiors):
                x, y = ring.xy
                ax.plot(x, y, color=MUTED, linewidth=.9, alpha=.8)
    ax.set_aspect("equal", adjustable="datalim")
    if calibration.y_down:
        ax.invert_yaxis()
    style(ax)


def select_indices(n, max_points=12000):
    return (np.arange(n) if n <= max_points else
            np.sort(np.random.default_rng(2026).choice(n, max_points, replace=False)))


def _series_labels_and_title(summaries, t, fallback_title=""):
    if not summaries:
        return [], fallback_title or t("all")
    distinct_labels = {s.label for s in summaries}
    distinct_groups = {s.group for s in summaries}
    has_custom_id = any(bool(s.series_id) for s in summaries)
    if not has_custom_id and len(distinct_labels) == 1:
        lbl = next(iter(distinct_labels))
        names = [s.group or t("all") for s in summaries]
        title = lbl or t("all")
    elif not has_custom_id and len(distinct_groups) == 1:
        grp = next(iter(distinct_groups))
        names = [s.label or t("all") for s in summaries]
        title = grp or t("all")
    else:
        names = [s.format_name(t("all")) for s in summaries]
        title = fallback_title or t("selected_groups")
    return names, title


def group_figure(results, label_or_summaries, t, band=None, show_samples=True, omnibus=None, title_override=""):
    fig = Figure(figsize=(8.8, 5.15), layout="constrained")
    ax, n_axis = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [4, 1]})
    r = results.radii
    unit = results.data.calibration.unit
    active_band = results.settings.band if band is None else band
    if isinstance(label_or_summaries, str):
        raw_summaries = [s for s in results.summaries if s.label == label_or_summaries]
        fallback = label_or_summaries or t("all")
    else:
        raw_summaries = list(label_or_summaries)
        fallback = title_override or t("selected_groups")
    summaries = [
        with_band(s, active_band, results.settings.seed) if active_band != results.settings.band else s
        for s in raw_summaries
    ]
    series_names, group_title = _series_labels_and_title(summaries, t, fallback)
    if title_override:
        group_title = title_override
    ax.axhline(0, color=MUTED, linestyle="--", linewidth=.9)
    n_max = 1
    for i, (summary, base_name) in enumerate(zip(summaries, series_names)):
        color = PALETTE[i % len(PALETTE)]
        linestyle = ("-", "--", ":", "-.")[i // len(PALETTE) % 4]
        n_max = max(n_max, int(summary.n.max()) if len(summary.n) else 1)
        n_label = (str(int(summary.n[0])) if len(summary.n) and np.all(summary.n == summary.n[0])
                   else f"{int(summary.n.min())}–{int(summary.n.max())}")
        name = f"{base_name} (N={n_label})"
        if show_samples and len(summary.curves):
            ax.plot(r, summary.curves.T, color=color, linewidth=.65, alpha=.18)
        if active_band != "off" and np.isfinite(summary.low).any():
            ax.fill_between(r, summary.low, summary.high, color=color, alpha=.16, linewidth=0)
        ax.plot(r, summary.mean, label=name, color=color, linestyle=linestyle, linewidth=2.15)
        n_axis.step(r, summary.n, where="mid", color=color, linestyle=linestyle, linewidth=1.5)
    if omnibus is not None and omnibus.envelope is not None and np.any(omnibus.envelope.side != 0):
        sig_mask = omnibus.envelope.side != 0
        for direction, start, end in omnibus.significant_spans:
            pad = (r[1] - r[0]) * 0.45 if len(r) > 1 else 0.5
            ax.axvspan(max(0.0, start - pad), end + pad, color="#FEF3C7", alpha=0.35, zorder=0, linewidth=0)
        if sig_mask.any():
            ax.plot([], [], color="#D97706", linewidth=5, alpha=0.45, label=t("sig_r_bar"))
    band_present = active_band != "off" and any(np.isfinite(s.low).any() for s in summaries)
    title_parts = [t("view_groups"), group_title]
    if band_present:
        title_parts.append(t(active_band))
    if omnibus is not None and np.isfinite(omnibus.p_global):
        title_parts.append(f"Global p = {omnibus.p_global:.4g}")
    ax.set_title("  ·  ".join(title_parts), loc="left", fontsize=10.5)
    ax.set_ylabel(f"L(r) − r ({unit})")
    if summaries:
        ax.legend(loc="best", fontsize=8, ncol=2 if len(summaries) > 3 else 1, frameon=False)
    n_axis.set_ylim(-.05, n_max + .5)
    n_axis.set_yticks(sorted({0, n_max // 2, n_max}))
    n_axis.set_ylabel("N")
    n_axis.set_xlabel(t("radius", unit=unit))
    for axes in (ax, n_axis):
        style(axes)
    use_cjk_font(fig, t)
    return fig


def roi_figure(results, pattern, t):
    fig = Figure(figsize=(9.6, 4.65), layout="constrained")
    map_ax, curve_ax = fig.subplots(1, 2)
    xy, window = pattern.xy, pattern.window
    indices = select_indices(len(xy))
    drawn = xy[indices]
    if pattern.local is not None and np.isfinite(pattern.local).any():
        local = pattern.local[indices]
        limit = max(float(np.nanmax(np.abs(pattern.local))), .001)
        colors = matplotlib.colors.LinearSegmentedColormap.from_list(
            "neighbor_delta", ["#1A6FB5", "#edf2f6", "#C05621"])
        dots = map_ax.scatter(*drawn.T, s=13, c=local, cmap=colors,
                              norm=TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit), alpha=.85)
        fig.colorbar(dots, ax=map_ax, shrink=.72, label=t("local_legend"))
    elif pattern.target_count is not None:
        for i, mark in enumerate((results.settings.from_type, results.settings.to_type)):
            keep = pattern.types[indices] == mark
            map_ax.scatter(*drawn[keep].T, s=15, color=PALETTE[i], marker=("o", "^")[i], label=mark)
        map_ax.legend(frameon=False, loc="best")
    else:
        map_ax.scatter(*drawn.T, s=13, c=PALETTE[0], alpha=.78)
    if pattern.locked_count:
        fixed = xy[boundary_points(window, xy)]
        map_ax.scatter(*fixed.T, s=26, edgecolors=PALETTE[4], facecolors="none", label=t("locked_points"))
        map_ax.legend(frameon=False, fontsize=8)
    spatial_axes(map_ax, xy, window.geometry, results.data.calibration, t)
    map_ax.set_title(f"{pattern.sample} / {pattern.roi}", loc="left")
    unit = results.data.calibration.unit
    env = pattern.envelope
    curve_ax.axhline(0, color=MUTED, linestyle="--", linewidth=.9, label=t("reference"))
    if env is not None:
        curve_ax.fill_between(results.radii, env.lower, env.upper, color="#A3C8E8",
                              alpha=.45, label=t("global_band"), linewidth=0)
        for sign, color in ((1, "#C05621"), (-1, "#6D3BBF")):
            out = env.side == sign
            curve_ax.scatter(results.radii[out], pattern.curve[out], s=14, color=color, zorder=5)
    curve_ax.plot(results.radii, pattern.curve, color=PALETTE[0], linewidth=2, label=t("estimate"))
    curve_ax.legend(loc="best", fontsize=8, frameon=False)
    curve_ax.set_xlabel(t("radius", unit=unit))
    curve_ax.set_ylabel(f"L(r) − r ({unit})")
    curve_ax.set_title(pattern.label or t("all"), loc="left")
    caption = t("inference", status=t(pattern.status), null=t(pattern.null), n=pattern.count)
    if env is not None and np.isfinite(env.p):
        caption += "\n" + t("p_values", p=env.p, adjusted=pattern.p_adjusted)
    fig.suptitle(caption, color=INK, fontsize=10)
    style(curve_ax)
    use_cjk_font(fig, t)
    return fig


def contrast_figure(results, contrast, t):
    fig = Figure(figsize=(8.7, 4.4), layout="constrained")
    ax = fig.subplots()
    env = contrast.envelope
    ax.axhline(0, color=MUTED, linestyle="--", linewidth=.9)
    ax.fill_between(results.radii, env.lower, env.upper, color="#A3C8E8", alpha=.45,
                    label=t("compare_caption"), linewidth=0)
    ax.plot(results.radii, contrast.effect, color=PALETTE[0], linewidth=2.1, label=t("estimate"))
    for sign, color in ((1, "#C05621"), (-1, "#6D3BBF")):
        out = env.side == sign
        ax.scatter(results.radii[out], contrast.effect[out], s=15, color=color, zorder=5)
    suffix = f"  ·  {contrast.label}" if contrast.label and contrast.label != f"{contrast.a} vs {contrast.b}" else ""
    ax.set_title(f"{contrast.a} − {contrast.b}{suffix}", loc="left")
    ax.set_xlabel(t("radius", unit=results.data.calibration.unit))
    ax.set_ylabel(t("effect", unit=results.data.calibration.unit))
    ax.legend(loc="best", fontsize=8, frameon=False)
    if np.isfinite(env.p):
        fig.suptitle(t("p_values", p=env.p, adjusted=contrast.p_adjusted), color=INK, fontsize=10)
    style(ax)
    use_cjk_font(fig, t)
    return fig


def omnibus_figure(results, omnibus, t):
    fig = Figure(figsize=(8.7, 4.4), layout="constrained")
    ax = fig.subplots()
    r = results.radii
    env = omnibus.envelope
    ax.fill_between(r, 0, env.upper, color="#A3C8E8", alpha=.4, label=t("omnibus_envelope_label"), linewidth=0)
    ax.plot(r, omnibus.statistic_curve, color=PALETTE[0], linewidth=2.1, label=t("estimate"))
    out = env.side > 0
    if out.any():
        ax.scatter(r[out], omnibus.statistic_curve[out], s=16, color="#C05621", zorder=5)
    ax.set_title(t("omnibus_title", k=omnibus.k_groups), loc="left")
    ax.set_xlabel(t("radius", unit=results.data.calibration.unit))
    ax.set_ylabel(t("omnibus_ylabel"))
    ax.legend(loc="best", fontsize=8, frameon=False)
    if np.isfinite(omnibus.p_global):
        fig.suptitle(f"Global p = {omnibus.p_global:.4g}  ·  {t('design_tag_' + omnibus.design)}", color=INK, fontsize=10)
    style(ax)
    use_cjk_font(fig, t)
    return fig


def save_pdf(figure, filename, title):
    save_figure(figure, filename, title)


def save_figure(figure, filename, title):
    path = Path(filename)
    ext = path.suffix.lower()
    fmt = "svg" if ext == ".svg" else "png" if ext == ".png" else "pdf"
    rc = {"pdf.fonttype": 42, "svg.fonttype": "none"}
    kwargs = {"format": fmt}
    if fmt == "pdf":
        kwargs["metadata"] = {"Title": title, "Creator": "Spatial Pattern Analyzer"}
    elif fmt == "png":
        kwargs["dpi"] = 300
        kwargs["facecolor"] = "white"
    with TemporaryDirectory(dir=path.parent) as tmp, matplotlib.rc_context(rc):
        target = Path(tmp) / path.name
        figure.savefig(target, **kwargs)
        target.replace(path)
