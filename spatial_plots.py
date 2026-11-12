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


PALETTE = ["#1A6FB5", "#0D8F6B", "#9B3072", "#B45309", "#6D3BBF", "#C4211C", "#0E7490"]
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


def group_figure(results, label, t):
    fig = Figure(figsize=(8.7, 5.1), layout="constrained")
    ax, n_axis = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [4, 1]})
    r = results.radii
    unit = results.data.calibration.unit
    summaries = [s for s in results.summaries if s.label == label]
    ax.axhline(0, color=MUTED, linestyle="--", linewidth=.9)
    n_max = 1
    for i, summary in enumerate(summaries):
        color = PALETTE[i % len(PALETTE)]
        linestyle = ("-", "--", ":", "-.")[i // len(PALETTE) % 4]
        n_max = max(n_max, int(summary.n.max()))
        n_label = str(summary.n[0]) if np.all(summary.n == summary.n[0]) else f"{summary.n.min()}–{summary.n.max()}"
        name = f"{summary.group or t('all')} (N={n_label})"
        ax.plot(r, summary.curves.T, color=color, linewidth=.7, alpha=.2)
        ax.fill_between(r, summary.low, summary.high, color=color, alpha=.15, linewidth=0)
        ax.plot(r, summary.mean, label=name, color=color, linestyle=linestyle, linewidth=2.1)
        n_axis.step(r, summary.n, where="mid", color=color, linestyle=linestyle, linewidth=1.5)
    band = results.settings.band
    band_present = any(np.isfinite(s.low).any() for s in summaries)
    ax.set_title(f"{t('view_groups')}  ·  {label or t('all')}" + (f"  ·  {t(band)}" if band_present else ""), loc="left")
    ax.set_ylabel(f"L(r) − r ({unit})")
    if summaries:
        ax.legend(loc="best", fontsize=8, ncol=2, frameon=False)
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
    fig = Figure(figsize=(8.7, 5), layout="constrained")
    ax = fig.subplots()
    env = contrast.envelope
    ax.axhline(0, color=MUTED, linestyle="--", linewidth=.9)
    ax.fill_between(results.radii, env.lower, env.upper, color="#A3C8E8", alpha=.45,
                    label=t("compare_caption"), linewidth=0)
    ax.plot(results.radii, contrast.effect, color=PALETTE[0], linewidth=2.1, label=t("estimate"))
    for sign, color in ((1, "#C05621"), (-1, "#6D3BBF")):
        out = env.side == sign
        ax.scatter(results.radii[out], contrast.effect[out], s=14, color=color)
    ax.set_title(f"{contrast.a} − {contrast.b}  ·  {contrast.label or t('all')}", loc="left")
    ax.set_xlabel(t("radius", unit=results.data.calibration.unit))
    ax.set_ylabel(t("effect", unit=results.data.calibration.unit))
    ax.legend(loc="best", fontsize=8, frameon=False)
    if np.isfinite(env.p):
        fig.suptitle(t("p_values", p=env.p, adjusted=contrast.p_adjusted), color=INK, fontsize=10)
    style(ax)
    use_cjk_font(fig, t)
    return fig


def save_pdf(figure, filename, title):
    path = Path(filename)
    with TemporaryDirectory(dir=path.parent) as tmp, matplotlib.rc_context({"pdf.fonttype": 42}):
        target = Path(tmp) / path.name
        figure.savefig(target, format="pdf", metadata={"Title": title, "Creator": "Spatial Pattern Analyzer"})
        target.replace(path)
