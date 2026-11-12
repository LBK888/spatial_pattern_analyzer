from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from shapely.geometry import Polygon

from ripley_analysis import IntensityGrid, make_window
from spatial_data import (COLUMNS, RECT_COLUMNS, Calibration, ReadOptions, column_labels,
                          excel_sheets, infer_mapping, read_table)
from spatial_plots import INK, MUTED, PALETTE, select_indices, spatial_axes, use_cjk_font


def modal(parent, title):
    dialog = tk.Toplevel(parent)
    dialog.title(title)
    dialog.transient(parent)
    dialog.configure(bg="#f3f6f8")
    previous_grab = parent.grab_current()
    if previous_grab is not None:
        def restore_grab(event):
            if event.widget == dialog and previous_grab.winfo_exists():
                previous_grab.grab_set()
        dialog.bind("<Destroy>", restore_grab)
    dialog.bind("<Escape>", lambda _: dialog.destroy())
    dialog.grab_set()
    return dialog


def help_button(parent, text):
    tip = None

    def toggle():
        nonlocal tip
        if tip is not None and tip.winfo_exists():
            tip.destroy()
            return
        tip = tk.Toplevel(button)
        tip.wm_overrideredirect(True)
        tip.transient(button.winfo_toplevel())
        label = tk.Label(tip, text=text, fg="white", bg="#203a50", justify="left",
                         wraplength=320, padx=13, pady=10)
        label.pack()
        tip.update_idletasks()
        x = min(button.winfo_rootx(), button.winfo_screenwidth() - tip.winfo_reqwidth() - 14)
        y = button.winfo_rooty() + button.winfo_height() + 4
        if y + tip.winfo_reqheight() > button.winfo_screenheight() - 20:
            y = button.winfo_rooty() - tip.winfo_reqheight() - 4
        tip.geometry(f"+{max(0, x)}+{max(0, y)}")
        tip.lift()
        tip.bind("<Button-1>", lambda _: tip.destroy())

    button = ttk.Button(parent, text="ⓘ", width=2, style="Help.TButton", command=toggle)
    return button


def frame(parent, padding=14):
    content = ttk.Frame(parent, padding=padding)
    content.pack(fill="both", expand=True)
    return content


def form_row(parent, t, label, key=None):
    line = ttk.Frame(parent)
    line.pack(fill="x", pady=3)
    heading = ttk.Frame(line)
    heading.pack(side="left", anchor="w")
    ttk.Label(heading, text=t(label)).pack(side="left")
    help_button(heading, t(key or label + "_help")).pack(side="left", padx=(3, 8))
    return line


def entry(parent, t, label, value="", help_key=None, width=19):
    row = form_row(parent, t, label, help_key)
    var = tk.StringVar(row, value=str(value))
    ttk.Entry(row, textvariable=var, width=width).pack(side="right")
    return var


def choice(parent, t, label, values, value, help_key=None, translated=True, width=27):
    row = form_row(parent, t, label, help_key)
    labels = [t(x) if translated else x for x in values]
    combo = ttk.Combobox(row, values=labels, state="readonly", width=width)
    combo.pack(side="right", fill="x", expand=True)
    if value in values:
        combo.current(values.index(value))
    return combo


def flag(parent, t, label, value, help_key=None):
    row = ttk.Frame(parent)
    row.pack(fill="x", pady=4)
    var = tk.BooleanVar(row, value=value)
    ttk.Checkbutton(row, text=t(label), variable=var).pack(side="left")
    help_button(row, t(help_key or label + "_help")).pack(side="left", padx=4)
    return var


def buttons(parent, t, callback, accept="accept"):
    bar = ttk.Frame(parent)
    bar.pack(fill="x", pady=(14, 0))
    ttk.Button(bar, text=t(accept), style="Primary.TButton", command=callback).pack(side="right")
    ttk.Button(bar, text=t("cancel"), command=parent.winfo_toplevel().destroy).pack(side="right", padx=(0, 8))


def number(var, kind, t, field, optional=False):
    value = var.get().strip()
    if optional and not value:
        return None
    try:
        return kind(value)
    except ValueError as error:
        raise ValueError(t("numeric_value", field=t(field))) from error


def show_error(parent, t, error):
    messagebox.showerror(t("failed"), t.error(error), parent=parent)


class ImportDialog:
    def __init__(self, parent, t, paths, callback):
        self.t, self.paths, self.callback = t, paths, callback
        self.options = ReadOptions()
        self.calibration = Calibration()
        self.duplicates = "keep"
        self.window = modal(parent, t("import_title"))
        self.window.geometry("940x625")
        self.window.minsize(740, 510)
        base = frame(self.window)
        header = ttk.Frame(base)
        header.pack(fill="x", pady=(0, 9))
        ttk.Label(header, text=t("batch_files", count=len(paths)), style="Section.TLabel").pack(side="left")
        ttk.Button(header, text=t("parse_options"), command=self.format_dialog).pack(side="right")
        ttk.Button(header, text=t("coordinates"), command=self.calibration_dialog).pack(side="right", padx=8)
        split = ttk.Panedwindow(base, orient="horizontal")
        split.pack(fill="both", expand=True)
        self.preview = ttk.Frame(split, padding=(0, 0, 9, 0))
        self.controls = ttk.Frame(split, padding=(9, 0, 0, 0))
        split.add(self.preview, weight=3)
        split.add(self.controls, weight=2)
        self.header = flag(self.controls, t, "has_header", True)
        self.header.trace_add("write", lambda *_: self.refresh())
        self.mapping_area = ttk.Frame(self.controls)
        self.mapping_area.pack(fill="x")
        buttons(base, t, self.accept, "load")
        self.mapping, self.combos, self.labels = {}, {}, []
        self.refresh()

    def collect(self):
        for key, combo in self.combos.items():
            self.mapping.pop(key, None)
            if combo.current() > 0:
                self.mapping[key] = combo.current() - 1

    def refresh(self):
        t = self.t
        for pane in (self.preview, self.mapping_area):
            for child in pane.winfo_children():
                child.destroy()
        self.mapping, self.combos, self.labels = {}, {}, []
        self.options = replace(self.options, header=self.header.get())
        path = self.paths[0]
        try:
            raw = read_table(path, self.options, limit=31)
            if raw.empty:
                raise ValueError("empty_file")
        except Exception as error:
            ttk.Label(self.preview, text=t.error(error), wraplength=430, style="Muted.TLabel").pack(anchor="nw")
            return
        labels = self.labels = column_labels(raw, self.options.header)
        self.mapping = infer_mapping(labels, raw, self.options.header, self.options.decimal)
        ttk.Label(self.preview, text=t("max_preview", file=Path(path).name), style="Muted.TLabel").pack(anchor="w", pady=(0, 6))
        container = ttk.Frame(self.preview)
        container.pack(fill="both", expand=True)
        columns = [f"c{i}" for i in range(len(labels))]
        tree = ttk.Treeview(container, columns=columns, show="headings", selectmode="none")
        vs = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
        hs = ttk.Scrollbar(container, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vs.grid(row=0, column=1, sticky="ns")
        hs.grid(row=1, column=0, sticky="ew")
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)
        for col, label in zip(columns, labels):
            tree.heading(col, text=label)
            tree.column(col, width=105, minwidth=60, stretch=False)
        for row in raw.iloc[int(self.options.header):][:30].itertuples(index=False, name=None):
            tree.insert("", "end", values=row)
        for field in COLUMNS[:6]:
            self.combos[field] = self.mapping_choice(self.mapping_area, field)
        ttk.Button(self.mapping_area, text=t("more_columns"), command=self.more_columns).pack(anchor="w", pady=8)

    def mapping_choice(self, parent, key):
        choices = [self.t("none")] + [f"{i + 1} · {label}" for i, label in enumerate(self.labels)]
        return choice(parent, self.t, key, choices, choices[self.mapping.get(key, -1) + 1],
                      "bounds_columns_help" if key in RECT_COLUMNS else None, translated=False, width=17)

    def more_columns(self):
        self.collect()
        t = self.t
        dialog = modal(self.window, t("more_columns"))
        base = frame(dialog)
        controls = {key: self.mapping_choice(base, key) for key in COLUMNS[6:]}

        def apply():
            for key, combo in controls.items():
                self.mapping.pop(key, None)
                if combo.current() > 0:
                    self.mapping[key] = combo.current() - 1
            dialog.destroy()

        buttons(base, t, apply)

    def format_dialog(self):
        t = self.t
        dialog = modal(self.window, t("file_format"))
        base = frame(dialog)
        path = self.paths[0]
        excel = Path(path).suffix.lower() in (".xls", ".xlsx")
        delimiters, names = ("auto", ",", ";", "\t"), ("auto", "comma", "semicolon", "tab")
        encodings = ("auto", "utf-8-sig", "cp950", "utf-16")
        decimal = choice(base, t, "decimal", [".", ","], self.options.decimal, translated=False)
        sheet, sep, encoding = None, None, None
        try:
            if excel:
                sheets = excel_sheets(path)
                current = self.options.sheet
                sheet = choice(base, t, "sheet", sheets,
                               sheets[current] if isinstance(current, int) else current, translated=False)
            else:
                sep = choice(base, t, "delimiter", list(names), names[delimiters.index(self.options.delimiter)])
                display = [t("auto"), "UTF-8", "CP950", "UTF-16"]
                encoding = choice(base, t, "encoding", display, display[encodings.index(self.options.encoding)], translated=False)
        except Exception as error:
            dialog.destroy()
            show_error(self.window, t, error)
            return

        def apply():
            self.options = replace(self.options, decimal=decimal.get(), sheet=sheet.get() if excel else 0,
                                   delimiter=delimiters[sep.current()] if sep else "auto",
                                   encoding=encodings[encoding.current()] if encoding else "auto")
            dialog.destroy()
            self.refresh()

        buttons(base, t, apply, "refresh")

    def calibration_dialog(self):
        t = self.t
        dialog = modal(self.window, t("coordinates"))
        base = frame(dialog)
        x = entry(base, t, "x_scale", self.calibration.x, "scale_help")
        y = entry(base, t, "y_scale", self.calibration.y, "scale_help")
        unit = entry(base, t, "unit", self.calibration.unit)
        y_down = flag(base, t, "y_down", self.calibration.y_down)
        dup = choice(base, t, "duplicates", ["keep", "drop"], self.duplicates)

        def apply():
            try:
                scales = [number(x, float, t, "x_scale"), number(y, float, t, "y_scale")]
                if not np.all(np.isfinite(scales)) or min(scales) <= 0:
                    raise ValueError("invalid_scale")
                self.calibration = Calibration(*scales, unit.get().strip() or "units", y_down.get())
                self.duplicates = ["keep", "drop"][dup.current()]
                dialog.destroy()
            except ValueError as error:
                show_error(dialog, t, error)

        buttons(base, t, apply)

    def accept(self):
        self.collect()
        if self.mapping.get("x") is None or self.mapping.get("y") is None or self.mapping["x"] == self.mapping["y"]:
            show_error(self.window, self.t, ValueError("select_xy"))
            return
        self.window.destroy()
        self.callback(self.paths, self.options, self.mapping.copy(), self.calibration, self.duplicates)


def advanced_dialog(parent, t, data, settings, intensity_path, callback):
    dialog = modal(parent, t("advanced"))
    dialog.minsize(570, 415)
    base = frame(dialog)
    tabs = ttk.Notebook(base)
    tabs.pack(fill="both", expand=True)
    pages = {}
    for section in ("distance", "tests", "groups"):
        pages[section] = ttk.Frame(tabs, padding=14)
        tabs.add(pages[section], text=t(section))
    scales = pages["distance"]
    keys = ("r_min", "r_max", "n_radii", "hull_ratio", "local_radius")
    scale_vars = {key: entry(scales, t, key, getattr(settings, key) if getattr(settings, key) is not None else "")
                  for key in keys}
    test = pages["tests"]
    run_test = flag(test, t, "run_test", settings.test)
    simulations = entry(test, t, "simulations", settings.simulations)
    seed = entry(test, t, "seed", settings.seed)
    modes = ["all", "each", "cross"]
    marks = choice(test, t, "marks", modes, settings.marks)
    categories = sorted(set(data.points.cell_type) - {""}) if data is not None else []
    focal = choice(test, t, "from_type", categories, settings.from_type, "cross_help", translated=False)
    neighbor = choice(test, t, "to_type", categories, settings.to_type, "cross_help", translated=False)
    if categories and not focal.get():
        focal.current(0)
    if len(categories) > 1 and not neighbor.get():
        neighbor.current(1)

    def update_marks(_=None):
        state = "readonly" if marks.current() == 2 else "disabled"
        focal.configure(state=state)
        neighbor.configure(state=state)
    marks.bind("<<ComboboxSelected>>", update_marks)
    update_marks()
    intensity = settings.intensity
    grid_name = tk.StringVar(dialog, value=Path(intensity_path).name if intensity_path else t("none"))

    def open_grid():
        nonlocal intensity, intensity_path
        path = filedialog.askopenfilename(parent=dialog, filetypes=[("CSV", "*.csv")])
        if path:
            try:
                intensity = IntensityGrid(np.loadtxt(path, delimiter=",", ndmin=2, encoding="utf-8-sig"))
                intensity_path = path
                grid_name.set(Path(path).name)
            except (OSError, ValueError):
                show_error(dialog, t, ValueError("invalid_intensity"))

    def clear_grid():
        nonlocal intensity, intensity_path
        intensity, intensity_path = None, ""
        grid_name.set(t("none"))

    grid = ttk.Frame(test)
    grid.pack(fill="x", pady=(10, 0))
    ttk.Button(grid, text=t("intensity"), command=open_grid).pack(side="left")
    help_button(grid, t("intensity_help")).pack(side="left", padx=4)
    ttk.Button(grid, text=t("clear_file"), command=clear_grid).pack(side="right")
    ttk.Label(test, textvariable=grid_name, style="Muted.TLabel").pack(anchor="w")
    groups = pages["groups"]
    bands = ["sd", "ci", "off"]
    band = choice(groups, t, "band", bands, settings.band)
    contrast = flag(groups, t, "contrast", settings.contrast)
    paired = flag(groups, t, "paired", settings.paired)
    names = sorted(set(data.points.group) - {""}) if data is not None else []
    a = choice(groups, t, "group_a", names, settings.group_a, "group_choice_help", translated=False)
    b = choice(groups, t, "group_b", names, settings.group_b, "group_choice_help", translated=False)
    if names and not a.get():
        a.current(0)
    if len(names) > 1 and not b.get():
        b.current(1)

    def apply():
        try:
            values = {key: number(var, int if key == "n_radii" else float, t, key,
                                  optional=key in ("r_min", "r_max", "local_radius"))
                      for key, var in scale_vars.items()}
            new = replace(settings, **values, test=run_test.get(), marks=modes[marks.current()],
                          simulations=number(simulations, int, t, "simulations"),
                          seed=number(seed, int, t, "seed"), from_type=focal.get(), to_type=neighbor.get(),
                          intensity=intensity, band=bands[band.current()], contrast=contrast.get(),
                          paired=paired.get(), group_a=a.get(), group_b=b.get())
            new.validate()
            if new.contrast and (not new.group_a or not new.group_b or new.group_a == new.group_b):
                raise ValueError("choose_two_groups")
            callback(new, intensity_path)
            dialog.destroy()
        except ValueError as error:
            show_error(dialog, t, error)

    buttons(base, t, apply)


def geometry_dialog(parent, t, data, callback):
    dialog = modal(parent, t("draw_roi"))
    dialog.geometry("730x590")
    base = frame(dialog)
    ttk.Label(base, text=t("draw_help"), wraplength=625, style="Muted.TLabel").pack(anchor="w", pady=(0, 8))
    fields = list(data.points.groupby(["sample_id", "roi_id"], sort=False))
    names = [f"{i + 1} · {key[0]} / {key[1]}" for i, (key, _) in enumerate(fields)]
    selector = choice(base, t, "field", names, names[0], translated=False)
    fig = Figure(figsize=(7, 4), layout="constrained")
    ax = fig.add_subplot()
    canvas = FigureCanvasTkAgg(fig, master=base)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    vertices = []

    def redraw(_=None):
        vertices.clear()
        ax.clear()
        xy = fields[selector.current()][1][["x", "y"]].to_numpy()
        ax.scatter(*xy[select_indices(len(xy))].T, s=9, color=PALETTE[0], alpha=.6)
        ax.margins(.12)
        spatial_axes(ax, xy, None, data.calibration, t)
        ax.plot([], [], "-o", markersize=4, color=PALETTE[3])
        use_cjk_font(fig, t)
        canvas.draw_idle()

    def click(event):
        if event.inaxes is ax and event.button == 1 and event.xdata is not None:
            vertices.append((event.xdata, event.ydata))
            shown = vertices + ([vertices[0]] if len(vertices) >= 3 else [])
            ax.lines[-1].set_data(*zip(*shown))
            canvas.draw_idle()

    def accept():
        try:
            if len(vertices) < 3:
                raise ValueError("invalid_roi")
            geometry = make_window("custom", [], geometry=Polygon(vertices)).geometry
            callback(fields[selector.current()][0], geometry)
            dialog.destroy()
        except ValueError as error:
            show_error(dialog, t, error)

    selector.bind("<<ComboboxSelected>>", redraw)
    canvas.mpl_connect("button_press_event", click)
    ttk.Button(base, text=t("reset"), command=redraw).pack(anchor="w")
    buttons(base, t, accept)
    redraw()
