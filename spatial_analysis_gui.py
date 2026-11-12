"""Tk desktop application for two-dimensional point-pattern analysis."""
from __future__ import annotations

from dataclasses import replace
import locale
from pathlib import Path
from queue import Empty, Queue
import sys
from threading import Event, Thread
import tkinter as tk
from tkinter import filedialog, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from i18n import Translator
from ripley_analysis import Cancelled, make_window
from spatial_data import (RECT_COLUMNS, Calibration, ReadOptions, column_labels, infer_mapping,
                          load_custom_roi, load_dataset, read_table)
from spatial_dialogs import (ImportDialog, advanced_dialog, buttons, entry, frame, geometry_dialog,
                             help_button, modal, number, show_error)
from spatial_plots import contrast_figure, group_figure, roi_figure, save_pdf, select_indices
from spatial_workflow import Settings, VERSION, export_data, interval_strings, run_analysis


ASSETS = Path(__file__).resolve().parent
MODES = ("infer", "locked", "rectangle", "custom")
VIEWS = ("view_groups", "view_roi", "view_contrast", "view_qc")
BG, WHITE, INK, MUTED, ACCENT = "#f3f6f8", "#ffffff", "#172c3c", "#526779", "#1a6fb5"


class PlotToolbar(NavigationToolbar2Tk):
    def __init__(self, canvas, parent, t):
        names = {"Home": "plot_home", "Back": "plot_back", "Forward": "plot_forward",
                 "Pan": "plot_pan", "Zoom": "plot_zoom"}
        self.toolitems = [(name, t(names[name]), icon, method)
                          for name, _, icon, method in self.toolitems if name in names]
        super().__init__(canvas, parent, pack_toolbar=False)


def demo_data():
    path = ASSETS / "demo" / "points.csv"
    options = ReadOptions()
    raw = read_table(path, options, limit=4)
    mapping = infer_mapping(column_labels(raw, options.header))
    return load_dataset([path], options, mapping, Calibration(unit="µm"))


class SpatialAnalysisGUI:
    def __init__(self, root, language="en"):
        self.root, self.t = root, Translator(language)
        self.data, self.results, self.figure = None, None, None
        self.settings = Settings()
        self.dataset_name = ""
        self.roi_path, self.intensity_path = "", ""
        self.view, self.selected = "view_groups", {}
        self.queue, self.job, self.next_job = Queue(), None, 0
        self.stop = None
        self.status = ("ready", {})
        self.draw_shell()
        root.after(70, self.poll)
        root.protocol("WM_DELETE_WINDOW", self.close)

    def setup_styles(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        family = "Microsoft JhengHei UI" if self.t.lang == "zh-TW" else "Segoe UI"
        self.root.option_add("*Font", (family, 10))
        style.configure(".", font=(family, 10), background=BG, foreground=INK)
        style.configure("TFrame", background=BG)
        style.configure("TButton", padding=(11, 6), background=WHITE, bordercolor="#c5d3dd", relief="flat")
        style.map("TButton", background=[("active", "#e7f1fa"), ("disabled", "#e9eff3")])
        style.configure("Primary.TButton", background=ACCENT, bordercolor=ACCENT, foreground=WHITE)
        style.map("Primary.TButton", background=[("disabled", "#a1b8ca"), ("active", "#145b96")],
                  foreground=[("disabled", WHITE), ("active", WHITE)])
        style.configure("Help.TButton", padding=(1, 0), borderwidth=0, background=BG, foreground=ACCENT)
        style.configure("Title.TLabel", font=(family, 20, "bold"), foreground=INK)
        style.configure("Hero.TLabel", font=(family, 24, "bold"), background=WHITE, foreground=INK)
        style.configure("HeroBody.TLabel", font=(family, 11), background=WHITE, foreground=MUTED)
        style.configure("Card.TFrame", background=WHITE)
        style.configure("Section.TLabel", font=(family, 11, "bold"))
        style.configure("Muted.TLabel", foreground=MUTED)
        style.configure("TNotebook", background=BG, borderwidth=0, padding=0)
        style.configure("TNotebook.Tab", padding=(13, 8))
        style.map("TNotebook.Tab", background=[("selected", WHITE), ("!selected", "#e5edf3")])
        style.configure("Treeview", background=WHITE, fieldbackground=WHITE, rowheight=27)
        style.configure("Treeview.Heading", background="#e7eef4", padding=(6, 5))
        self.root.configure(bg=BG)

    def draw_shell(self):
        for child in self.root.winfo_children():
            child.destroy()
        self.figure = None
        self.setup_styles()
        root, t = self.root, self.t
        root.title(f"{t('title')}  ·  {VERSION}")
        header = ttk.Frame(root, padding=(21, 13, 21, 12))
        header.pack(fill="x")
        names = ttk.Frame(header)
        names.pack(side="left")
        ttk.Label(names, text=t("title"), style="Title.TLabel").pack(anchor="w")
        ttk.Label(names, text=t("subtitle"), style="Muted.TLabel").pack(anchor="w")
        languages = ttk.Combobox(header, values=("English", "繁體中文"), width=11, state="readonly")
        languages.current(t.lang == "zh-TW")
        languages.pack(side="right", padx=(8, 0))
        languages.bind("<<ComboboxSelected>>", lambda _: self.switch_language(languages.current()))
        ttk.Button(header, text=t("methods"), command=self.methods).pack(side="right")
        self.open_btn = ttk.Button(header, text=t("open"), command=self.open_data)
        self.open_btn.pack(side="right", padx=8)
        body = ttk.Frame(root, padding=(20, 0, 20, 0))
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        side = ttk.Frame(body, padding=(0, 13, 17, 0), width=275)
        side.grid(row=0, column=0, sticky="nsw")
        side.grid_propagate(False)
        side.pack_propagate(False)
        ttk.Label(side, text=t("workflow"), style="Section.TLabel").pack(anchor="w")
        ttk.Separator(side).pack(fill="x", pady=(8, 13))
        ttk.Label(side, text=t("data"), style="Section.TLabel").pack(anchor="w")
        self.dataset_label = ttk.Label(side, wraplength=240, justify="left", style="Muted.TLabel")
        self.dataset_label.pack(fill="x", pady=(4, 8))
        self.demo_btn = ttk.Button(side, text=t("demo"), command=self.open_demo)
        self.demo_btn.pack(anchor="w")
        ttk.Separator(side).pack(fill="x", pady=14)
        roi_line = ttk.Frame(side)
        roi_line.pack(fill="x")
        ttk.Label(roi_line, text=t("roi"), style="Section.TLabel").pack(side="left")
        help_button(roi_line, t("roi_help")).pack(side="left", padx=4)
        self.mode = ttk.Combobox(side, values=[t(mode) for mode in MODES], state="readonly", width=28)
        self.mode.current(MODES.index(self.settings.roi_mode))
        self.mode.pack(fill="x", pady=(4, 0))
        self.mode.bind("<<ComboboxSelected>>", self.change_roi)
        self.roi_note = ttk.Label(side, wraplength=243, justify="left", style="Muted.TLabel")
        self.roi_note.pack(fill="x", pady=(8, 6))
        self.roi_buttons = ttk.Frame(side)
        self.roi_buttons.pack(fill="x")
        self.advanced_btn = ttk.Button(side, text=t("advanced"), command=self.open_advanced)
        self.advanced_btn.pack(fill="x", pady=(15, 5))
        self.run_btn = ttk.Button(side, text=t("run"), style="Primary.TButton", command=self.start)
        self.run_btn.pack(fill="x", pady=(0, 6))
        self.cancel_btn = ttk.Button(side, text=t("cancel_job"), command=self.cancel)
        self.cancel_btn.pack(fill="x")
        main = ttk.Frame(body)
        main.grid(row=0, column=1, sticky="nsew")
        self.main = main
        footer = ttk.Frame(root, padding=(21, 9))
        footer.pack(fill="x")
        self.status_label = ttk.Label(footer, text=t(self.status[0], **self.status[1]), style="Muted.TLabel")
        self.status_label.pack(side="left")
        self.progress = ttk.Progressbar(footer, length=185, mode="determinate")
        self.progress.pack(side="right")
        self.refresh_sidebar()
        self.render_main()
        root.bind("<Control-o>", lambda _: self.open_data())

    def set_status(self, key, **values):
        self.status = (key, values)
        self.status_label.configure(text=self.t(key, **values))

    def switch_language(self, selected):
        self.t = Translator("zh-TW" if selected else "en")
        self.draw_shell()

    def refresh_sidebar(self):
        t = self.t
        if self.data is None:
            text = t("no_data")
        else:
            data = self.data
            text = self.dataset_name + "\n" + t("data_count", n=len(data.points), N=data.points.sample_id.nunique(),
                                               fields=len({(s, r) for _, s, r in data.fields}))
            if len(data.issues):
                text += "\n" + t("qc_count", count=len(data.issues))
        self.dataset_label.configure(text=text)
        mode = self.settings.roi_mode
        note = t(mode + "_note")
        if mode == "rectangle" and self.settings.rectangle is not None:
            note += "\n" + " · ".join(f"{x:g}" for x in self.settings.rectangle)
        if mode == "custom" and self.settings.custom:
            note += "\n" + (Path(self.roi_path).name if self.roi_path else str(len(self.settings.custom)))
        self.roi_note.configure(text=note)
        for child in self.roi_buttons.winfo_children():
            child.destroy()
        actions = {"rectangle": (("configure_roi", self.rectangle_dialog),),
                   "custom": (("load_roi", self.import_roi), ("draw_roi", self.draw_roi))}
        for label, command in actions.get(mode, ()):
            ttk.Button(self.roi_buttons, text=t(label), command=command,
                       state="normal" if self.data is not None and not self.job else "disabled").pack(anchor="w", pady=2)
        busy = self.job is not None
        for widget in (self.open_btn, self.demo_btn):
            widget.configure(state="disabled" if busy else "normal")
        state = "normal" if self.data is not None and not busy else "disabled"
        self.mode.configure(state="readonly" if state == "normal" else "disabled")
        self.advanced_btn.configure(state=state)
        self.run_btn.configure(state=state)
        self.cancel_btn.configure(state="normal" if busy and self.job[1] == "analysis" else "disabled")
        if self.results is not None and hasattr(self, "clear_btn") and self.clear_btn.winfo_exists():
            self.clear_btn.configure(state="disabled" if busy else "normal")

    def change_roi(self, _):
        self.settings = replace(self.settings, roi_mode=MODES[self.mode.current()])
        self.refresh_sidebar()

    def open_data(self):
        if self.job is not None:
            return
        files = filedialog.askopenfilenames(parent=self.root, title=self.t("open"), filetypes=[
            ("CSV / TSV / Excel", ("*.csv", "*.tsv", "*.txt", "*.xlsx", "*.xls")), ("*", "*")])
        if files:
            ImportDialog(self.root, self.t, files, self.import_points)

    def import_points(self, paths, options, mapping, calibration, duplicates):
        name = Path(paths[0]).name if len(paths) == 1 else f"{Path(paths[0]).name} +{len(paths) - 1}"
        self.launch("import", lambda _job, _stop: (load_dataset(paths, options, mapping, calibration, duplicates), name))

    def open_demo(self):
        if self.job is None:
            self.launch("import", lambda _job, _stop: (demo_data(), "Demo · points.csv"))

    def rectangle_dialog(self):
        dialog = modal(self.root, self.t("configure_roi"))
        base = frame(dialog)
        factors = self.data.calibration.factors
        old = self.settings.rectangle
        values = old if old is not None else ("", "", "", "")
        if old is not None:
            values = [v / factors[i % 2] for i, v in enumerate(old)]
        controls = [entry(base, self.t, label, value, "bounds_help") for label, value in zip(RECT_COLUMNS, values)]

        def accept():
            try:
                bounds = tuple(number(control, float, self.t, key) * factors[i % 2]
                               for i, (key, control) in enumerate(zip(RECT_COLUMNS, controls)))
                make_window("rectangle", [], bounds)
                self.settings = replace(self.settings, rectangle=bounds)
                self.refresh_sidebar()
                dialog.destroy()
            except ValueError as error:
                show_error(dialog, self.t, error)

        buttons(base, self.t, accept)

    def import_roi(self):
        t = self.t
        path = filedialog.askopenfilename(parent=self.root, title=t("load_roi"), filetypes=[
            ("ROI · GeoJSON / WKT / PNG / TIFF", ("*.geojson", "*.json", "*.wkt", "*.png", "*.tif", "*.tiff"))])
        if not path:
            return

        def load(origin, parent):
            try:
                geometry = load_custom_roi(path, self.data.calibration, origin)
                for window in geometry.values():
                    make_window("custom", [], geometry=window)
                self.settings = replace(self.settings, custom=geometry)
                self.roi_path = path
                self.refresh_sidebar()
                return True
            except Exception as error:
                show_error(parent, t, error)
                return False

        if Path(path).suffix.lower() not in (".png", ".tif", ".tiff"):
            load((0, 0), self.root)
            return
        dialog = modal(self.root, t("mask_origin"))
        base = frame(dialog)
        x = entry(base, t, "x", "0", "origin_help")
        y = entry(base, t, "y", "0", "origin_help")

        def accept():
            try:
                if load((number(x, float, t, "x"), number(y, float, t, "y")), dialog):
                    dialog.destroy()
            except ValueError as error:
                show_error(dialog, t, error)

        buttons(base, t, accept)

    def draw_roi(self):
        def save(key, geometry):
            self.settings = replace(self.settings, custom={**self.settings.custom, key: geometry})
            self.refresh_sidebar()
        geometry_dialog(self.root, self.t, self.data, save)

    def open_advanced(self):
        def save(settings, path):
            self.settings, self.intensity_path = settings, path
        advanced_dialog(self.root, self.t, self.data, self.settings, self.intensity_path, save)

    def launch(self, kind, work):
        if self.job is not None:
            return
        self.next_job += 1
        job = self.next_job
        self.job = (job, kind)
        self.stop = Event()
        stop = self.stop
        self.set_status("loading" if kind == "import" else "preparing")
        self.progress.configure(value=0, maximum=100)
        self.refresh_sidebar()

        def worker():
            try:
                result = work(job, stop)
                self.queue.put((job, "success", result))
            except Cancelled:
                self.queue.put((job, "canceled", None))
            except Exception as error:
                self.queue.put((job, "error", error))

        Thread(target=worker, daemon=True).start()

    def start(self):
        if self.data is None or self.job is not None:
            return
        data, settings = self.data, self.settings
        roi_path, intensity_path = self.roi_path, self.intensity_path

        def analyze(job, stop):
            results = run_analysis(data, settings, lambda done, total: self.queue.put((job, "progress", (done, total))), stop)
            results.source_roi, results.source_intensity = roi_path, intensity_path
            return results

        self.launch("analysis", analyze)

    def poll(self):
        try:
            while True:
                job, message, payload = self.queue.get_nowait()
                if self.job is None or self.job[0] != job:
                    continue
                kind = self.job[1]
                if message == "progress":
                    done, total = payload
                    if not self.stop.is_set():
                        self.set_status("progress", done=done, total=total) if done < total else self.set_status("finalizing")
                    self.progress.configure(value=done, maximum=max(1, total))
                    continue
                canceled = self.stop.is_set() or message == "canceled"
                self.job, self.stop = None, None
                try:
                    if canceled:
                        self.set_status("canceled")
                    elif message == "error":
                        self.set_status("failed")
                        show_error(self.root, self.t, payload)
                    elif kind == "import":
                        self.data, self.dataset_name = payload
                        self.results = None
                        self.settings = Settings(roi_mode="rectangle" if self.data.bounds else "infer")
                        self.mode.current(MODES.index(self.settings.roi_mode))
                        self.roi_path, self.intensity_path = "", ""
                        self.selected.clear()
                        self.set_status("ready")
                        self.render_main()
                    else:
                        self.results = payload
                        self.view = "view_groups"
                        self.selected.clear()
                        self.set_status("done")
                        self.render_main()
                except Exception as error:
                    self.set_status("failed")
                    show_error(self.root, self.t, error)
                finally:
                    self.refresh_sidebar()
        except Empty:
            pass
        finally:
            self.root.after(70, self.poll)

    def cancel(self):
        if self.stop is not None:
            self.stop.set()
            self.set_status("canceled")

    def close(self):
        if self.stop is not None:
            self.stop.set()
        self.root.destroy()

    def render_main(self):
        t = self.t
        self.figure = None
        for child in self.main.winfo_children():
            child.destroy()
        if self.results is None:
            self.welcome()
            return
        bar = ttk.Frame(self.main)
        bar.pack(fill="x", pady=(0, 6))
        ttk.Label(bar, text=t("results"), style="Section.TLabel").pack(side="left")
        self.pdf_btn = ttk.Button(bar, text=t("pdf"), command=self.export_pdf)
        self.pdf_btn.pack(side="right")
        ttk.Button(bar, text=t("data_export"), command=self.export_tables).pack(side="right", padx=6)
        self.clear_btn = ttk.Button(bar, text=t("clear_results"), command=self.clear_results,
                                    state="disabled" if self.job else "normal")
        self.clear_btn.pack(side="right")
        tabs = ttk.Notebook(self.main)
        tabs.pack(fill="both", expand=True)
        self.pages = {}
        for key in VIEWS:
            page = ttk.Frame(tabs, padding=(9, 8))
            self.pages[key] = page
            tabs.add(page, text=t(key))
        tabs.select(self.pages[self.view])

        def change(_):
            selected = tabs.select()
            key = next((k for k, page in self.pages.items() if str(page) == selected), None)
            if key is not None and (key != self.view or not self.pages[key].winfo_children()):
                self.view = key
                self.render_view()

        tabs.bind("<<NotebookTabChanged>>", change)
        self.render_view()

    def welcome(self):
        t = self.t
        hero = ttk.Frame(self.main, style="Card.TFrame", padding=(32, 37))
        hero.pack(fill="both", expand=True)
        ttk.Label(hero, text=t("welcome_title"), style="Hero.TLabel", wraplength=530).pack(anchor="w")
        ttk.Label(hero, text=t("welcome_body"), style="HeroBody.TLabel", wraplength=520, justify="left").pack(anchor="w", pady=(12, 18))
        if self.data is not None and len(self.data.issues):
            ttk.Label(hero, text=t("qc_before_run"), style="Hero.TLabel").pack(anchor="w", pady=(8, 3))
            self.qc_table(hero, self.data.issues)

    def selected_item(self, page, labels):
        key = self.view
        index = min(self.selected.get(key, 0), len(labels) - 1)
        if len(labels) < 2:
            return index
        combo = ttk.Combobox(page, values=labels, state="readonly", width=45)
        combo.current(index)
        combo.pack(anchor="w", pady=(0, 6))

        def select(_):
            self.selected[key] = combo.current()
            self.render_view()
        combo.bind("<<ComboboxSelected>>", select)
        return index

    def render_view(self):
        t = self.t
        page = self.pages[self.view]
        for child in page.winfo_children():
            child.destroy()
        self.figure = None
        result = self.results
        if self.view == "view_groups":
            labels = list(dict.fromkeys(s.label for s in result.summaries))
            if labels:
                i = self.selected_item(page, [label or t("all") for label in labels])
                notes = [t("group_caption"), t("group_caption_n")]
                if result.settings.band != "off" and any(s.n.min() < (3 if result.settings.band == "ci" else 2)
                                                         for s in result.summaries if s.label == labels[i]):
                    notes.append(t("band_na"))
                if result.settings.band == "ci":
                    notes.append(t("no_band_common"))
                self.show_figure(page, group_figure(result, labels[i], t), notes)
            else:
                ttk.Label(page, text=t("no_summary"), wraplength=550).pack(anchor="w")
        elif self.view == "view_roi":
            patterns = result.patterns
            i = self.selected_item(page, [f"{p.sample} / {p.roi}  ·  {p.label or t('all')}" for p in patterns])
            p = patterns[i]
            note = t("inference", status=t(p.status), null=t(p.null), n=p.count)
            if p.target_count is not None:
                note += t("n_targets", n=p.target_count)
            notes = [note + "  ·  " + t(p.window.correction)]
            notes.extend(self.envelope_notes(p.envelope, p.p_adjusted))
            shown = len(select_indices(len(p.xy)))
            if shown < len(p.xy):
                notes.append(t("point_count_drawn", n=shown, total=len(p.xy)))
            notes.extend(t(qc) for qc in p.qc)
            self.show_figure(page, roi_figure(result, p, t), notes)
        elif self.view == "view_contrast":
            contrasts = result.contrasts
            if contrasts:
                i = self.selected_item(page, [c.label or t("all") for c in contrasts])
                c = contrasts[i]
                notes = [f"{c.a} (N={c.n_a})  ·  {c.b} (N={c.n_b})"]
                notes.extend(self.envelope_notes(c.envelope, c.p_adjusted))
                self.show_figure(page, contrast_figure(result, c, t), notes)
            else:
                ttk.Label(page, text=t("no_contrast"), wraplength=550).pack(anchor="w")
        else:
            self.qc_table(page, result.qc)
        self.pdf_btn.configure(state="normal" if self.figure is not None else "disabled")

    def envelope_notes(self, env, adjusted):
        if env is None:
            return []
        t = self.t
        notes = []
        if env.p == env.p:
            notes.append(t("p_values", p=env.p, adjusted=adjusted))
        if not (env.lower == env.lower).all():
            notes.append(t("masked_scales"))
        spans = interval_strings(self.results.radii, env.side)
        if spans:
            notes.append("  ·  ".join(t("scale_span", direction=t(direction), lo=lo, hi=hi,
                                        unit=self.results.data.calibration.unit) for direction, lo, hi in spans[:4])
                         + ("  …" if len(spans) > 4 else ""))
            if adjusted > .05:
                notes.append(t("p_family_note"))
        return notes

    def show_figure(self, parent, figure, notes):
        self.figure = figure
        for note in notes:
            ttk.Label(parent, text=note, wraplength=690, style="Muted.TLabel").pack(fill="x", anchor="w", pady=1)
        canvas = FigureCanvasTkAgg(figure, master=parent)
        toolbar = PlotToolbar(canvas, parent, self.t)
        toolbar.pack(side="bottom", fill="x")
        canvas.get_tk_widget().pack(fill="both", expand=True, pady=(5, 0))
        canvas.draw_idle()

    def qc_table(self, parent, records):
        t = self.t
        if records.empty:
            ttk.Label(parent, text=t("no_qc")).pack(anchor="w")
            return
        panel = ttk.Frame(parent)
        panel.pack(fill="both", expand=True)
        fields = ("name", "row", "sample_id", "roi_id", "reason")
        tree = ttk.Treeview(panel, columns=fields, show="headings")
        scroll = ttk.Scrollbar(panel, command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        tree.pack(fill="both", expand=True)
        for field, width in zip(fields, (130, 58, 75, 68, 350)):
            tree.column(field, width=width, minwidth=45, stretch=field == "reason")
            tree.heading(field, text=t(field))
        if len(records) > 250:
            ttk.Label(parent, text=t("qc_limit", count=len(records)), style="Muted.TLabel").pack(anchor="w")
        columns = ["source", "row", "sample_id", "roi_id", "reason"]
        for source, row, sample, roi, reason in records.head(250).reindex(columns=columns).fillna("").itertuples(index=False, name=None):
            tree.insert("", "end", values=(Path(source).name if source else "", row, sample, roi, t(reason)))

    def export_pdf(self):
        if self.figure is None:
            return
        t = self.t
        filename = filedialog.asksaveasfilename(parent=self.root, title=t("pdf"), initialfile="spatial-pattern.pdf",
                                                defaultextension=".pdf", filetypes=[("PDF", "*.pdf")])
        if filename:
            try:
                save_pdf(self.figure, filename, t("title"))
                self.set_status("saved", file=Path(filename).name)
            except Exception as error:
                show_error(self.root, t, error)

    def clear_results(self):
        if self.job is None:
            self.results = None
            self.set_status("ready")
            self.render_main()

    def export_tables(self):
        if self.results is None:
            return
        t = self.t
        filename = filedialog.asksaveasfilename(parent=self.root, title=t("data_export"), initialfile="spatial-results.xlsx",
                                                defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx"), ("CSV (ZIP)", "*.zip")])
        if filename:
            try:
                export_data(self.results, filename)
                self.set_status("saved", file=Path(filename).name)
            except Exception as error:
                show_error(self.root, t, error)

    def methods(self):
        t = self.t
        dialog = modal(self.root, t("methods"))
        dialog.geometry("800x625")
        base = frame(dialog)
        from tkinter.scrolledtext import ScrolledText

        text = ScrolledText(base, wrap="word", padx=13, pady=11, bg=WHITE, fg=INK, relief="flat")
        text.pack(fill="both", expand=True)
        document = ASSETS / ("README.zh-TW.md" if t.lang == "zh-TW" else "README.md")
        text.insert("1.0", document.read_text(encoding="utf-8-sig"))
        text.configure(state="disabled")
        ttk.Button(base, text=t("close"), command=dialog.destroy).pack(anchor="e", pady=(8, 0))


def smoke_test(path):
    path.mkdir(parents=True, exist_ok=True)
    try:
        import xlrd

        assert xlrd.__version__
        root = tk.Tk()
        root.withdraw()
        app = SpatialAnalysisGUI(root)
        app.data = demo_data()
        app.results = run_analysis(app.data, Settings(roi_mode="rectangle", simulations=19))
        for lang in ("en", "zh-TW"):
            t = Translator(lang)
            save_pdf(group_figure(app.results, "", t), path / f"groups-{lang}.pdf", t("title"))
        export_data(app.results, path / "results.xlsx")
        root.update_idletasks()
        root.destroy()
    except Exception:
        from traceback import format_exc

        (path / "error.txt").write_text(format_exc(), encoding="utf-8")
        raise SystemExit(1) from None


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--smoke-test":
        smoke_test(Path(sys.argv[2]))
        return
    if len(sys.argv) != 1:
        raise SystemExit("Usage: spatial_analysis_gui.py [--smoke-test OUTPUT_DIRECTORY]")
    root = tk.Tk()
    root.geometry("1200x790")
    root.minsize(920, 650)
    language = locale.getlocale()[0] or ""
    zh = language.casefold().startswith(("zh_tw", "zh_hk", "chinese (traditional)"))
    SpatialAnalysisGUI(root, "zh-TW" if zh else "en")
    root.mainloop()


if __name__ == "__main__":
    main()
