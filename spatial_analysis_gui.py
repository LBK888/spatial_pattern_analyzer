"""PyQt6 desktop application for two-dimensional point-pattern analysis."""
from __future__ import annotations

import os
from pathlib import Path
import sys

if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--smoke-test":
    _SMOKE_DIR = Path(sys.argv[2].strip('"'))
    _SMOKE_DIR.mkdir(parents=True, exist_ok=True)
else:
    _SMOKE_DIR = None

try:
    from dataclasses import replace
    import locale
    from threading import Event

    import numpy as np
    from PyQt6.QtCore import Qt, QThread, pyqtSignal
    from PyQt6.QtGui import QFont, QKeySequence, QPalette, QPixmap, QShortcut
    from PyQt6.QtWidgets import (
        QAbstractItemView,
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFrame,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMainWindow,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    from i18n import Translator
    from ripley_analysis import Cancelled, make_window
    from spatial_data import (
        Calibration,
        ReadOptions,
        column_labels,
        infer_mapping,
        load_custom_roi,
        load_dataset,
        read_table,
    )
    from spatial_dialogs import (
        ImportDialog,
        PlotToolbar,
        advanced_dialog,
        geometry_dialog,
        help_button,
        mask_origin_dialog,
        methods_dialog,
        parse_number,
        rectangle_dialog,
        series_selector_dialog,
        show_error,
    )
    from spatial_roi_guide import ROIGuide
    from spatial_theme import apply_theme
    from spatial_plots import (
        contrast_figure,
        group_figure,
        omnibus_figure,
        roi_figure,
        save_figure,
        save_pdf,
        select_indices,
    )
    from spatial_workflow import (
        Settings,
        VERSION,
        build_natural_presets,
        compare_series,
        default_fixed_grid,
        export_batch_bundle,
        export_data,
        interval_strings,
        run_analysis,
        sorted_types,
        with_band,
    )
except BaseException:
    if _SMOKE_DIR is not None:
        from traceback import format_exc

        (_SMOKE_DIR / "error.txt").write_text(format_exc(), encoding="utf-8")
        raise SystemExit(1) from None
    raise


ASSETS = Path(__file__).resolve().parent
MODES = ("infer", "locked", "rectangle", "custom")
RECTANGLE_HULLS = ("convex", "concave")
EDGE_CHOICES = ("auto", "translation", "isotropic", "border")
EDGE_LABELS = ("edge_auto", "edge_translation", "edge_isotropic", "edge_border")
MARKS_MODES = ("all", "each", "each_and_merged", "cross", "cross_all")
VIEWS = ("view_groups", "view_roi", "view_contrast", "view_qc")
BANDS = ("sd", "se", "ci", "off")
PAIRING_MODES = ("auto", "independent", "paired")
PERM_STATS = ("welch", "diff")

def demo_data():
    path = ASSETS / "demo" / "points.csv"
    options = ReadOptions()
    raw = read_table(path, options, limit=4)
    mapping = infer_mapping(column_labels(raw, options.header))
    return load_dataset([path], options, mapping, Calibration(unit="µm"))


class WorkerThread(QThread):
    progress_changed = pyqtSignal(int, int, int)
    task_finished = pyqtSignal(int, str, object)

    def __init__(self, job_id: int, work_fn, stop_event: Event, parent=None):
        super().__init__(parent)
        self.job_id = job_id
        self.work_fn = work_fn
        self.stop_event = stop_event

    def run(self):
        try:
            result = self.work_fn(
                self.job_id,
                self.stop_event,
                lambda done, total: self.progress_changed.emit(self.job_id, int(done), int(total)),
            )
            if self.stop_event.is_set():
                self.task_finished.emit(self.job_id, "canceled", None)
            else:
                self.task_finished.emit(self.job_id, "success", result)
        except Cancelled:
            self.task_finished.emit(self.job_id, "canceled", None)
        except Exception as error:
            self.task_finished.emit(self.job_id, "error", error)


class SpatialAnalysisGUI(QMainWindow):
    """Main PyQt6 window for Spatial Pattern Analyzer."""

    def __init__(self, parent_or_lang="en", language: str | None = None):
        super().__init__(None)
        if language is not None:
            lang = language
        elif isinstance(parent_or_lang, str):
            lang = parent_or_lang
        else:
            lang = "en"
        self.t = Translator(lang)
        self.data = None
        self.results = None
        self.figure = None
        self.settings = Settings()
        self.dataset_name = ""
        self.last_import_paths = None
        self.last_import_state = None
        self.roi_path = ""
        self.intensity_path = ""
        self.view = "view_groups"
        self.selected: dict[str, int] = {}
        self.roi_filter_group: str | None = None
        self.roi_filter_type: str | None = None
        self.qc_search_text = ""
        self.active_series = []
        self.preset_index: int | None = 0
        self.show_samples = True
        self.job = None
        self.worker: WorkerThread | None = None
        self.next_job = 0
        self.stop: Event | None = None
        self.status = ("ready", {})

        self._shortcut_open = QShortcut(QKeySequence("Ctrl+O"), self)
        self._shortcut_open.activated.connect(self.open_data)

        self.resize(1280, 830)
        self.setMinimumSize(980, 660)
        app = QApplication.instance()
        hints = app.styleHints()
        hints.colorSchemeChanged.connect(self._apply_color_scheme)
        self._apply_color_scheme(hints.colorScheme())
        self.draw_shell()

    def _apply_color_scheme(self, scheme):
        apply_theme(QApplication.instance(), scheme)

    def setup_styles(self):
        app = QApplication.instance()
        family = "Microsoft JhengHei UI" if self.t.lang == "zh-TW" else "Segoe UI"
        app.setFont(QFont(family, 10))

    def _preserve_sidebar_draft(self):
        if hasattr(self, "fixed_r_cb") and self.fixed_r_cb is not None:
            try:
                self._collect_sidebar_inputs(strict=False)
            except Exception:
                pass

    def draw_shell(self):
        self._preserve_sidebar_draft()
        self.setup_styles()
        t = self.t
        self.setWindowTitle(f"{t('title')}  ·  {VERSION}")
        self.figure = None

        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(20, 12, 20, 10)
        outer.setSpacing(8)

        # Header
        header = QHBoxLayout()
        title_col = QVBoxLayout()
        title_col.setSpacing(1)
        title_lbl = QLabel(t("title"))
        title_lbl.setObjectName("TitleLabel")
        sub_lbl = QLabel(t("subtitle"))
        sub_lbl.setObjectName("MutedLabel")
        title_col.addWidget(title_lbl)
        title_col.addWidget(sub_lbl)
        header.addLayout(title_col)
        header.addStretch(1)

        self.open_btn = QPushButton(t("open"))
        self.open_btn.clicked.connect(self.open_data)
        header.addWidget(self.open_btn)

        self.reimport_btn = QPushButton(t("reconfigure_import"))
        self.reimport_btn.clicked.connect(self.reconfigure_import)
        header.addWidget(self.reimport_btn)

        methods_btn = QPushButton(t("methods"))
        methods_btn.clicked.connect(self.methods)
        header.addWidget(methods_btn)

        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["English", "繁體中文"])
        self.lang_combo.setCurrentIndex(1 if t.lang == "zh-TW" else 0)
        self.lang_combo.setFixedWidth(115)
        self.lang_combo.currentIndexChanged.connect(self.switch_language)
        header.addWidget(self.lang_combo)
        outer.addLayout(header)

        # Body: Left scrollable setup sidebar + Right content area
        body = QHBoxLayout()
        body.setSpacing(16)
        outer.addLayout(body, stretch=1)

        side_scroll = QScrollArea()
        side_scroll.setObjectName("AnalysisSidebar")
        side_scroll.setFixedWidth(366)
        side_scroll.setWidgetResizable(True)
        side_scroll.setFrameShape(QFrame.Shape.NoFrame)
        side_scroll.viewport().setBackgroundRole(QPalette.ColorRole.Window)
        side_widget = QWidget()
        side = QVBoxLayout(side_widget)
        side.setContentsMargins(0, 4, 8, 4)
        side.setSpacing(8)
        side_scroll.setWidget(side_widget)
        body.addWidget(side_scroll)

        wf_lbl = QLabel(t("workflow"))
        wf_lbl.setObjectName("SectionLabel")
        side.addWidget(wf_lbl)

        # Card 1: Dataset
        ds_card = QFrame()
        ds_card.setObjectName("CardFrame")
        ds_layout = QVBoxLayout(ds_card)
        ds_layout.setContentsMargins(12, 10, 12, 10)
        ds_hdr = QLabel(t("data"))
        ds_hdr.setObjectName("SectionLabel")
        ds_layout.addWidget(ds_hdr)
        self.dataset_label = QLabel("")
        self.dataset_label.setWordWrap(True)
        self.dataset_label.setObjectName("MutedLabel")
        ds_layout.addWidget(self.dataset_label)
        self.demo_btn = QPushButton(t("demo"))
        self.demo_btn.setObjectName("SmallBtn")
        self.demo_btn.clicked.connect(self.open_demo)
        ds_layout.addWidget(self.demo_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        side.addWidget(ds_card)

        # Card 2: Observation Window (ROI)
        roi_card = QFrame()
        roi_card.setObjectName("CardFrame")
        roi_layout = QVBoxLayout(roi_card)
        roi_layout.setContentsMargins(12, 10, 12, 10)
        roi_hdr = QHBoxLayout()
        roi_lbl = QLabel(t("roi"))
        roi_lbl.setObjectName("SectionLabel")
        roi_hdr.addWidget(roi_lbl)
        roi_hdr.addStretch(1)
        roi_guide_btn = QPushButton(t("roi_guide"))
        roi_guide_btn.setObjectName("SmallBtn")
        roi_guide_btn.setToolTip(t("roi_help"))
        roi_guide_btn.clicked.connect(lambda: ROIGuide(self, self.t.lang).exec())
        roi_hdr.addWidget(roi_guide_btn)
        roi_layout.addLayout(roi_hdr)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems([t(m) for m in MODES])
        self.mode_combo.setCurrentIndex(MODES.index(self.settings.roi_mode))
        self.mode_combo.currentIndexChanged.connect(self.change_roi)
        roi_layout.addWidget(self.mode_combo)

        self.roi_note = QLabel("")
        self.roi_note.setWordWrap(True)
        self.roi_note.setObjectName("MutedLabel")
        roi_layout.addWidget(self.roi_note)

        self.rectangle_source = QWidget()
        rectangle_row = QHBoxLayout(self.rectangle_source)
        rectangle_row.setContentsMargins(0, 0, 0, 0)
        rectangle_label = QLabel(t("rectangle_source"))
        self.rectangle_combo = QComboBox()
        self.rectangle_combo.addItems([t("source_" + key) for key in RECTANGLE_HULLS])
        self.rectangle_combo.setCurrentIndex(RECTANGLE_HULLS.index(self.settings.rectangle_hull))
        self.rectangle_combo.currentIndexChanged.connect(self._on_rectangle_source_changed)
        rectangle_label.setBuddy(self.rectangle_combo)
        rectangle_row.addWidget(rectangle_label)
        rectangle_row.addWidget(self.rectangle_combo, stretch=1)
        roi_layout.addWidget(self.rectangle_source)

        # Edge correction & hull area bias guard (Recommendations A & C)
        edge_row = QHBoxLayout()
        edge_row.addWidget(QLabel(t("edge_correction")))
        edge_row.addWidget(help_button(roi_card, t("edge_correction_help")))
        edge_row.addStretch(1)
        roi_layout.addLayout(edge_row)
        self.edge_combo = QComboBox()
        self.edge_combo.addItems([t(lbl) for lbl in EDGE_LABELS])
        cur_ec = EDGE_CHOICES.index(self.settings.edge_correction) if self.settings.edge_correction in EDGE_CHOICES else 0
        self.edge_combo.setCurrentIndex(cur_ec)
        self.edge_combo.currentIndexChanged.connect(self._on_edge_changed)
        roi_layout.addWidget(self.edge_combo)

        self.infer_options = QWidget()
        infer_layout = QVBoxLayout(self.infer_options)
        infer_layout.setContentsMargins(0, 0, 0, 0)
        self.convex_guard_cb = QCheckBox(t("convex_hull_guard"))
        self.convex_guard_cb.setChecked(self.settings.convex_guard)
        self.convex_guard_cb.toggled.connect(self._on_convex_guard_toggled)
        self.shared_hull_cb = QCheckBox(t("shared_sample_hull"))
        self.shared_hull_cb.setChecked(not self.settings.infer_per_label)
        self.shared_hull_cb.toggled.connect(self._on_shared_hull_toggled)
        for check, tip in ((self.convex_guard_cb, "convex_hull_guard_help"),
                           (self.shared_hull_cb, "shared_sample_hull_help")):
            row = QHBoxLayout()
            row.addWidget(check)
            row.addWidget(help_button(self.infer_options, t(tip)))
            row.addStretch(1)
            infer_layout.addLayout(row)
        roi_layout.addWidget(self.infer_options)

        self.custom_options = QWidget()
        custom_buttons = QVBoxLayout(self.custom_options)
        custom_buttons.setContentsMargins(0, 0, 0, 0)
        custom_row = QHBoxLayout()
        custom_buttons.addLayout(custom_row)
        for key, slot in (("load_roi", self.import_roi), ("draw_roi", self.draw_roi),
                          ("configure_roi", self.open_rectangle_dialog)):
            button = QPushButton(t(key))
            button.setObjectName("SmallBtn")
            button.clicked.connect(slot)
            if key == "configure_roi":
                custom_buttons.addWidget(button, alignment=Qt.AlignmentFlag.AlignLeft)
            else:
                custom_row.addWidget(button)
        custom_row.addStretch(1)
        roi_layout.addWidget(self.custom_options)
        side.addWidget(roi_card)

        # Card 3: Point-Type Mode (promoted to main sidebar for discoverability)
        marks_card = QFrame()
        marks_card.setObjectName("CardFrame")
        marks_layout = QVBoxLayout(marks_card)
        marks_layout.setContentsMargins(12, 10, 12, 10)
        m_hdr = QHBoxLayout()
        m_lbl = QLabel(t("marks"))
        m_lbl.setObjectName("SectionLabel")
        m_hdr.addWidget(m_lbl)
        m_hdr.addWidget(help_button(marks_card, t("marks_help")))
        m_hdr.addStretch(1)
        marks_layout.addLayout(m_hdr)

        self.marks_combo = QComboBox()
        self.marks_combo.addItems([t(m) for m in MARKS_MODES])
        cur_m = MARKS_MODES.index(self.settings.marks) if self.settings.marks in MARKS_MODES else 2
        self.marks_combo.setCurrentIndex(cur_m)
        self.marks_combo.currentIndexChanged.connect(self._on_marks_changed)
        marks_layout.addWidget(self.marks_combo)

        self.cross_box = QWidget()
        cross_layout = QHBoxLayout(self.cross_box)
        cross_layout.setContentsMargins(0, 2, 0, 0)
        self.focal_combo = QComboBox()
        self.neighbor_combo = QComboBox()
        for combo in (self.focal_combo, self.neighbor_combo):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(8)
            combo.currentTextChanged.connect(combo.setToolTip)
        cross_layout.addWidget(self.focal_combo, stretch=1)
        cross_layout.addWidget(QLabel("→"))
        cross_layout.addWidget(self.neighbor_combo, stretch=1)
        self.focal_combo.currentTextChanged.connect(self._on_cross_type_changed)
        self.neighbor_combo.currentTextChanged.connect(self._on_cross_type_changed)
        marks_layout.addWidget(self.cross_box)
        side.addWidget(marks_card)

        # Card 4: Fixed Radius Discretization
        r_card = QFrame()
        r_card.setObjectName("CardFrame")
        r_layout = QVBoxLayout(r_card)
        r_layout.setContentsMargins(12, 10, 12, 10)
        r_hdr = QHBoxLayout()
        r_lbl = QLabel(t("fixed_r_section"))
        r_lbl.setObjectName("SectionLabel")
        r_hdr.addWidget(r_lbl)
        r_hdr.addWidget(help_button(r_card, t("fixed_r_help")))
        r_hdr.addStretch(1)
        r_layout.addLayout(r_hdr)

        self.fixed_r_cb = QCheckBox(t("use_fixed_r"))
        self.fixed_r_cb.setChecked(self.settings.r_step is not None)
        self.fixed_r_cb.toggled.connect(self._sync_sidebar_r)
        r_layout.addWidget(self.fixed_r_cb)

        r_inputs = QHBoxLayout()
        r_inputs.addWidget(QLabel(t("r_range_inline")))
        self.rmax_edit = QLineEdit("" if self.settings.r_max is None else f"{self.settings.r_max:g}")
        self.rmax_edit.setFixedWidth(66)
        r_inputs.addWidget(self.rmax_edit)
        r_inputs.addWidget(QLabel(t("r_step_inline")))
        self.rstep_edit = QLineEdit("" if self.settings.r_step is None else f"{self.settings.r_step:g}")
        self.rstep_edit.setFixedWidth(56)
        r_inputs.addWidget(self.rstep_edit)
        r_inputs.addStretch(1)
        r_layout.addLayout(r_inputs)
        self.rmax_edit.textChanged.connect(self._update_r_note)
        self.rstep_edit.textChanged.connect(self._update_r_note)

        self.r_note = QLabel("")
        self.r_note.setWordWrap(True)
        self.r_note.setObjectName("MutedLabel")
        r_layout.addWidget(self.r_note)
        side.addWidget(r_card)

        # Action buttons
        self.advanced_btn = QPushButton(t("advanced"))
        self.advanced_btn.clicked.connect(self.open_advanced)
        side.addWidget(self.advanced_btn)

        self.run_btn = QPushButton(t("run"))
        self.run_btn.setObjectName("PrimaryBtn")
        self.run_btn.setMinimumHeight(34)
        self.run_btn.clicked.connect(self.start)
        side.addWidget(self.run_btn)

        self.cancel_btn = QPushButton(t("cancel_job"))
        self.cancel_btn.clicked.connect(self.cancel)
        side.addWidget(self.cancel_btn)
        side.addStretch(1)

        # Main right panel
        self.main_area = QWidget()
        self.main_layout = QVBoxLayout(self.main_area)
        self.main_layout.setContentsMargins(0, 4, 0, 0)
        body.addWidget(self.main_area, stretch=1)

        # Footer status + progress bar
        footer = QHBoxLayout()
        self.status_label = QLabel(t(self.status[0], **self.status[1]))
        self.status_label.setObjectName("MutedLabel")
        footer.addWidget(self.status_label)
        footer.addStretch(1)
        self.progress = QProgressBar()
        self.progress.setFixedWidth(210)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        footer.addWidget(self.progress)
        outer.addLayout(footer)

        self._populate_cross_combos()
        self.refresh_sidebar()
        self.render_main()

    def _populate_cross_combos(self):
        types = sorted_types(self.data.points.cell_type) if self.data is not None else []
        self.focal_combo.blockSignals(True)
        self.neighbor_combo.blockSignals(True)
        self.focal_combo.clear()
        self.neighbor_combo.clear()
        self.focal_combo.addItems(types)
        self.neighbor_combo.addItems(types)
        if self.settings.from_type in types:
            self.focal_combo.setCurrentText(self.settings.from_type)
        elif types:
            self.focal_combo.setCurrentIndex(0)
        if self.settings.to_type in types:
            self.neighbor_combo.setCurrentText(self.settings.to_type)
        elif len(types) > 1:
            self.neighbor_combo.setCurrentIndex(1)
        self.focal_combo.blockSignals(False)
        self.neighbor_combo.blockSignals(False)
        self.cross_box.setVisible(self.settings.marks == "cross")

    def _on_edge_changed(self, idx: int):
        if 0 <= idx < len(EDGE_CHOICES):
            self.settings = replace(self.settings, edge_correction=EDGE_CHOICES[idx])

    def _on_convex_guard_toggled(self, checked: bool):
        self.settings = replace(self.settings, convex_guard=checked)

    def _on_shared_hull_toggled(self, checked: bool):
        self.settings = replace(self.settings, infer_per_label=not checked)

    def _on_rectangle_source_changed(self, index):
        self.settings = replace(self.settings, rectangle_hull=RECTANGLE_HULLS[index])

    def _on_marks_changed(self):
        new_mark = MARKS_MODES[self.marks_combo.currentIndex()]
        self.settings = replace(
            self.settings,
            marks=new_mark,
            from_type=self.focal_combo.currentText(),
            to_type=self.neighbor_combo.currentText(),
        )
        self.cross_box.setVisible(new_mark == "cross")

    def _on_cross_type_changed(self):
        if self.settings.marks == "cross":
            self.settings = replace(
                self.settings,
                from_type=self.focal_combo.currentText(),
                to_type=self.neighbor_combo.currentText(),
            )

    def _sync_sidebar_r(self):
        if self.fixed_r_cb.isChecked():
            _, rmax_def, rstep_def = default_fixed_grid(
                self.data, use_known_bounds=self.settings.roi_mode == "custom"
            )
            if not self.rmax_edit.text().strip():
                self.rmax_edit.setText(f"{rmax_def:g}")
            if not self.rstep_edit.text().strip():
                self.rstep_edit.setText(f"{rstep_def:g}")
        self._update_r_note()
        self.refresh_sidebar()

    def _collect_sidebar_inputs(self, strict: bool = True):
        if not self.fixed_r_cb.isChecked():
            self.settings = replace(self.settings, r_step=None)
        else:
            if strict:
                rmax = parse_number(self.rmax_edit, float, self.t, "r_max")
                rstep = parse_number(self.rstep_edit, float, self.t, "r_step")
            else:
                rmax = float(self.rmax_edit.text().strip()) if self.rmax_edit.text().strip() else self.settings.r_max
                rstep = float(self.rstep_edit.text().strip()) if self.rstep_edit.text().strip() else self.settings.r_step
            rmin = self.settings.r_min if (self.settings.r_min is not None and self.settings.r_min > 0) else 0.0
            self.settings = replace(self.settings, r_min=rmin, r_max=rmax, r_step=rstep)
        if self.settings.marks == "cross":
            self.settings = replace(
                self.settings,
                from_type=self.focal_combo.currentText(),
                to_type=self.neighbor_combo.currentText(),
            )

    def _update_r_note(self):
        t = self.t
        if self.fixed_r_cb.isChecked():
            try:
                rmax = float(self.rmax_edit.text().strip())
                rstep = float(self.rstep_edit.text().strip())
                r0 = self.settings.r_min if (self.settings.r_min and self.settings.r_min > 0) else rstep
                if rstep > 0 and rmax > r0:
                    cnt = int(np.floor((rmax - r0) / rstep + 1e-9)) + 1
                    r_last = r0 + (cnt - 1) * rstep
                    self.r_note.setText(t("r_grid_summary", count=cnt, r0=r0, r1=r_last, step=rstep))
                    return
            except ValueError:
                pass
        self.r_note.setText(t("r_auto_summary", n=self.settings.n_radii))

    def set_status(self, key: str, **values):
        self.status = (key, values)
        self.status_label.setText(self.t(key, **values))

    def switch_language(self, selected_index: int):
        new_lang = "zh-TW" if selected_index else "en"
        if new_lang != self.t.lang:
            self.t = Translator(new_lang)
            self.draw_shell()

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            sub = item.layout()
            if w is not None:
                w.deleteLater()
            elif sub is not None:
                self._clear_layout(sub)

    def refresh_sidebar(self):
        t = self.t
        if self.data is None:
            text = t("no_data")
        else:
            data = self.data
            text = self.dataset_name + "\n" + t(
                "data_count",
                n=len(data.points),
                N=data.points.sample_id.nunique(),
                fields=len({(s, r) for _, s, r in data.fields}),
            )
            n_grp = len(set(data.points.group) - {""})
            n_typ = len(set(data.points.cell_type) - {""})
            if n_grp or n_typ:
                text += "\n" + t("data_meta_summary", groups=max(1, n_grp), types=max(1, n_typ))
            if len(data.issues):
                text += "\n" + t("qc_count", count=len(data.issues))
        self.dataset_label.setText(text)

        mode = self.settings.roi_mode
        note = t(mode + "_note")
        if mode == "custom":
            if self.settings.custom:
                note += "\n" + (Path(self.roi_path).name if self.roi_path else t("custom_count", n=len(self.settings.custom)))
            elif self.data is not None and self.data.bounds:
                note += "\n" + t("custom_bounds_count", n=len(self.data.bounds))
        self.roi_note.setText(note)

        busy = self.job is not None
        ready = self.data is not None and not busy
        self.rectangle_source.setVisible(mode == "rectangle")
        self.rectangle_source.setEnabled(ready)
        self.custom_options.setVisible(mode == "custom")
        self.custom_options.setEnabled(ready)
        self.infer_options.setVisible(mode == "infer")
        self.convex_guard_cb.setEnabled(ready)
        self.shared_hull_cb.setEnabled(ready)

        self.open_btn.setEnabled(not busy)
        self.reimport_btn.setEnabled(not busy and self.last_import_paths is not None)
        self.demo_btn.setEnabled(not busy)
        self.mode_combo.setEnabled(ready)
        self.edge_combo.setEnabled(ready)
        self.marks_combo.setEnabled(ready)
        self.focal_combo.setEnabled(ready)
        self.neighbor_combo.setEnabled(ready)
        self.fixed_r_cb.setEnabled(ready)
        self.rmax_edit.setEnabled(ready and self.fixed_r_cb.isChecked())
        self.rstep_edit.setEnabled(ready and self.fixed_r_cb.isChecked())
        self._update_r_note()
        self.advanced_btn.setEnabled(ready)
        self.run_btn.setEnabled(ready)
        self.cancel_btn.setEnabled(bool(busy and self.job[1] == "analysis"))
        if hasattr(self, "clear_btn") and self.clear_btn is not None:
            try:
                self.clear_btn.setEnabled(not busy)
            except RuntimeError:
                pass

    def change_roi(self, index: int):
        self.settings = replace(self.settings, roi_mode=MODES[index])
        self.refresh_sidebar()

    def open_data(self):
        if self.job is not None:
            return
        files, _ = QFileDialog.getOpenFileNames(
            self,
            self.t("open"),
            "",
            "CSV / TSV / Excel (*.csv *.tsv *.txt *.xlsx *.xls);;All files (*)",
        )
        if files:
            dlg = ImportDialog(self, self.t, files, self.import_points)
            dlg.exec()

    def reconfigure_import(self):
        if self.job is not None or self.last_import_paths is None:
            return
        dlg = ImportDialog(
            self, self.t, self.last_import_paths, self.import_points, initial_state=self.last_import_state
        )
        dlg.exec()

    def import_points(self, paths, options, mapping, calibration, duplicates):
        self.last_import_paths = tuple(paths)
        self.last_import_state = (options, dict(mapping), calibration, duplicates)
        base_name = Path(paths[0]).name if len(paths) == 1 else f"{Path(paths[0]).name} +{len(paths) - 1}"
        if options.sheets and len(options.sheets) > 1:
            base_name += f" ({len(options.sheets)} sheets)"
        self.launch(
            "import",
            lambda _job, _stop, _prog: (load_dataset(paths, options, mapping, calibration, duplicates), base_name),
        )

    def open_demo(self):
        if self.job is None:
            self.last_import_paths = None
            self.last_import_state = None
            self.launch("import", lambda _job, _stop, _prog: (demo_data(), "Demo · points.csv"))

    def open_rectangle_dialog(self):
        if self.data is None:
            return

        def save_bounds(geometry):
            self.settings = replace(self.settings, custom={**self.settings.custom, None: geometry})
            self.roi_path = ""
            self.refresh_sidebar()

        window = self.settings.custom.get(None)
        current = window.bounds if window is not None and window.equals(window.envelope) else None
        rectangle_dialog(self, self.t, self.data.calibration, current, save_bounds)

    def import_roi(self):
        if self.data is None:
            return
        t = self.t
        path, _ = QFileDialog.getOpenFileName(
            self,
            t("load_roi"),
            "",
            "ROI · GeoJSON / WKT / PNG / TIFF (*.geojson *.json *.wkt *.png *.tif *.tiff)",
        )
        if not path:
            return

        def load_with_origin(origin, parent_widget):
            try:
                geometry = load_custom_roi(path, self.data.calibration, origin)
                for window in geometry.values():
                    make_window("custom", [], geometry=window)
                self.settings = replace(self.settings, custom=geometry)
                self.roi_path = path
                self.refresh_sidebar()
                return True
            except Exception as error:
                show_error(parent_widget, t, error)
                return False

        if Path(path).suffix.lower() not in (".png", ".tif", ".tiff"):
            load_with_origin((0.0, 0.0), self)
        else:
            mask_origin_dialog(self, t, load_with_origin)

    def draw_roi(self):
        if self.data is None:
            return

        def save_geom(key, geometry):
            self.settings = replace(self.settings, custom={**self.settings.custom, key: geometry})
            self.refresh_sidebar()

        geometry_dialog(self, self.t, self.data, self.settings.custom, save_geom)

    def open_advanced(self):
        self._preserve_sidebar_draft()

        def save(new_settings, path):
            self.settings, self.intensity_path = new_settings, path
            self.edge_combo.blockSignals(True)
            if new_settings.edge_correction in EDGE_CHOICES:
                self.edge_combo.setCurrentIndex(EDGE_CHOICES.index(new_settings.edge_correction))
            self.edge_combo.blockSignals(False)
            self.convex_guard_cb.blockSignals(True)
            self.convex_guard_cb.setChecked(new_settings.convex_guard)
            self.convex_guard_cb.blockSignals(False)
            self.shared_hull_cb.blockSignals(True)
            self.shared_hull_cb.setChecked(not new_settings.infer_per_label)
            self.shared_hull_cb.blockSignals(False)
            self.marks_combo.blockSignals(True)
            if new_settings.marks in MARKS_MODES:
                self.marks_combo.setCurrentIndex(MARKS_MODES.index(new_settings.marks))
            self.marks_combo.blockSignals(False)
            self._populate_cross_combos()
            self.fixed_r_cb.blockSignals(True)
            self.fixed_r_cb.setChecked(new_settings.r_step is not None)
            self.fixed_r_cb.blockSignals(False)
            self.rmax_edit.setText("" if new_settings.r_max is None else f"{new_settings.r_max:g}")
            self.rstep_edit.setText("" if new_settings.r_step is None else f"{new_settings.r_step:g}")
            self.refresh_sidebar()

        advanced_dialog(self, self.t, self.data, self.settings, self.intensity_path, save)

    def launch(self, kind: str, work_fn):
        if self.job is not None:
            return
        self.next_job += 1
        job_id = self.next_job
        self.job = (job_id, kind)
        self.stop = Event()
        self.set_status("loading" if kind == "import" else "preparing")
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.refresh_sidebar()

        self.worker = WorkerThread(job_id, work_fn, self.stop, self)
        self.worker.progress_changed.connect(self._on_worker_progress)
        self.worker.task_finished.connect(self._on_worker_finished)
        self.worker.start()

    def _on_worker_progress(self, job_id: int, done: int, total: int):
        if self.job is None or self.job[0] != job_id:
            return
        if self.stop is not None and not self.stop.is_set():
            if done < total:
                self.set_status("progress", done=done, total=total)
            else:
                self.set_status("finalizing")
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)

    def _on_worker_finished(self, job_id: int, state: str, payload):
        if self.job is None or self.job[0] != job_id:
            return
        kind = self.job[1]
        canceled = (self.stop is not None and self.stop.is_set()) or state == "canceled"
        self.job, self.stop, self.worker = None, None, None
        try:
            if canceled:
                self.set_status("canceled")
            elif state == "error":
                self.set_status("failed")
                show_error(self, self.t, payload)
            elif kind == "import":
                self.data, self.dataset_name = payload
                self.results = None
                self._configure_defaults_for_dataset()
                self.roi_path, self.intensity_path = "", ""
                self.selected.clear()
                self.roi_filter_group = None
                self.roi_filter_type = None
                self.active_series.clear()
                self.set_status("ready")
                self.render_main()
            else:
                self.results = payload
                self.view = "view_groups"
                self.selected.clear()
                presets = self._build_presets()
                self.preset_index = 0
                if presets:
                    self._update_active_comparison(presets[0][1])
                else:
                    self.active_series = list(self.results.summaries)
                self.set_status("done")
                self.render_main()
        except Exception as error:
            self.set_status("failed")
            show_error(self, self.t, error)
        finally:
            self.refresh_sidebar()

    def start(self):
        if self.data is None or self.job is not None:
            return
        try:
            self._collect_sidebar_inputs(strict=True)
            self.settings.validate()
        except ValueError as error:
            show_error(self, self.t, error)
            return
        data, settings = self.data, self.settings
        roi_path, intensity_path = self.roi_path, self.intensity_path

        def analyze(_job, stop_ev, progress_cb):
            results = run_analysis(data, settings, progress_cb, stop_ev)
            results.source_roi, results.source_intensity = roi_path, intensity_path
            return results

        self.launch("analysis", analyze)

    def _configure_defaults_for_dataset(self):
        n_samples = self.data.points.sample_id.nunique()
        types = sorted_types(self.data.points.cell_type)
        roi_mode = "custom" if self.data.bounds else "infer"
        marks = "each_and_merged" if len(types) >= 2 else "all"
        if n_samples > 1:
            r0, rmax, rstep = default_fixed_grid(self.data)
            self.settings = Settings(
                roi_mode=roi_mode,
                edge_correction="translation",
                r_min=r0,
                r_max=rmax,
                r_step=rstep,
                marks=marks,
                from_type=types[0] if types else "",
                to_type=types[1] if len(types) > 1 else "",
            )
        else:
            self.settings = Settings(
                roi_mode=roi_mode,
                marks=marks,
                from_type=types[0] if types else "",
                to_type=types[1] if len(types) > 1 else "",
            )
        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentIndex(MODES.index(self.settings.roi_mode))
        self.mode_combo.blockSignals(False)
        self.edge_combo.blockSignals(True)
        self.edge_combo.setCurrentIndex(EDGE_CHOICES.index(self.settings.edge_correction))
        self.edge_combo.blockSignals(False)
        self.convex_guard_cb.blockSignals(True)
        self.convex_guard_cb.setChecked(self.settings.convex_guard)
        self.convex_guard_cb.blockSignals(False)
        self.shared_hull_cb.blockSignals(True)
        self.shared_hull_cb.setChecked(not self.settings.infer_per_label)
        self.shared_hull_cb.blockSignals(False)
        self.rectangle_combo.blockSignals(True)
        self.rectangle_combo.setCurrentIndex(0)
        self.rectangle_combo.blockSignals(False)
        self.marks_combo.blockSignals(True)
        self.marks_combo.setCurrentIndex(MARKS_MODES.index(self.settings.marks))
        self.marks_combo.blockSignals(False)
        self._populate_cross_combos()
        self.fixed_r_cb.blockSignals(True)
        self.fixed_r_cb.setChecked(self.settings.r_step is not None)
        self.fixed_r_cb.blockSignals(False)
        self.rmax_edit.setText("" if self.settings.r_max is None else f"{self.settings.r_max:g}")
        self.rstep_edit.setText("" if self.settings.r_step is None else f"{self.settings.r_step:g}")

    def _build_presets(self):
        t = self.t
        res = self.results
        if res is None or not res.summaries:
            return []
        presets = build_natural_presets(res, t)
        if len(res.summaries) > 1 and len(presets) > 1:
            presets.append((t("preset_all_series", n=len(res.summaries)), list(res.summaries)))
        return presets

    def _update_active_comparison(self, series_list):
        self.active_series = list(series_list)
        res = self.results
        if res is None:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            if len(self.active_series) >= 2:
                res.comparison = compare_series(
                    self.active_series,
                    res.radii,
                    data=res.data,
                    simulations=min(res.settings.simulations, 999),
                    seed=res.settings.seed,
                    pairing_mode=res.settings.pairing_mode,
                    perm_stat=res.settings.perm_stat,
                    all_text=self.t("all"),
                )
            else:
                res.comparison = None
            self.selected.pop("view_contrast", None)
        finally:
            QApplication.restoreOverrideCursor()

    def cancel(self):
        if self.stop is not None:
            self.stop.set()
            self.set_status("canceled")

    def closeEvent(self, event):
        if self.stop is not None:
            self.stop.set()
        if self.worker is not None and self.worker.isRunning():
            self.worker.wait(1500)
        super().closeEvent(event)

    def render_main(self):
        t = self.t
        self.figure = None
        self._clear_layout(self.main_layout)
        if self.results is None:
            self._render_welcome()
            return

        bar = QHBoxLayout()
        res_lbl = QLabel(t("results"))
        res_lbl.setObjectName("SectionLabel")
        bar.addWidget(res_lbl)
        bar.addStretch(1)

        self.clear_btn = QPushButton(t("clear_results"))
        self.clear_btn.setEnabled(self.job is None)
        self.clear_btn.clicked.connect(self.clear_results)
        bar.addWidget(self.clear_btn)

        export_data_btn = QPushButton(t("data_export"))
        export_data_btn.clicked.connect(self.export_tables)
        bar.addWidget(export_data_btn)

        batch_btn = QPushButton(t("batch_export"))
        batch_btn.clicked.connect(self.export_batch)
        bar.addWidget(batch_btn)

        self.pdf_btn = QPushButton(t("export_figure"))
        self.pdf_btn.setObjectName("PrimaryBtn")
        self.pdf_btn.clicked.connect(self.export_pdf)
        bar.addWidget(self.pdf_btn)
        self.main_layout.addLayout(bar)

        self.tabs = QTabWidget()
        self.pages: dict[str, QWidget] = {}
        for key in VIEWS:
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(12, 10, 12, 10)
            page_layout.setSpacing(6)
            self.pages[key] = page
            self.tabs.addTab(page, t(key))
        self.tabs.setCurrentIndex(VIEWS.index(self.view))
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.main_layout.addWidget(self.tabs, stretch=1)
        self.render_view()

    def _on_tab_changed(self, index: int):
        if 0 <= index < len(VIEWS):
            self.view = VIEWS[index]
            self.render_view()

    def _render_welcome(self):
        t = self.t
        hero = QFrame()
        hero.setObjectName("CardFrame")
        layout = QVBoxLayout(hero)
        layout.setContentsMargins(36, 36, 36, 36)
        layout.setSpacing(12)
        title = QLabel(t("welcome_title"))
        title.setObjectName("HeroTitle")
        title.setWordWrap(True)
        body = QLabel(t("welcome_body"))
        body.setObjectName("HeroBody")
        body.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(body)
        if self.data is not None and len(self.data.issues):
            qc_hdr = QLabel(t("qc_before_run"))
            qc_hdr.setObjectName("SectionLabel")
            layout.addWidget(qc_hdr)
            self._build_qc_table(layout, self.data.issues)
        else:
            layout.addStretch(1)
        self.main_layout.addWidget(hero, stretch=1)

    def _grouping_toolbar(self, page_layout: QVBoxLayout, include_band: bool = True, include_design: bool = False):
        t = self.t
        res = self.results
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)

        presets = self._build_presets()
        preset_names = [p[0] for p in presets]
        if self.preset_index is None:
            preset_names.append(t("preset_custom", n=len(self.active_series)))
            cur_idx = len(preset_names) - 1
        else:
            cur_idx = min(self.preset_index, max(0, len(preset_names) - 1))

        if preset_names:
            toolbar.addWidget(QLabel(t("grouping_label")))
            p_combo = QComboBox()
            p_combo.addItems(preset_names)
            p_combo.setCurrentIndex(cur_idx)
            p_combo.setMinimumWidth(240)

            def on_preset(idx: int):
                if 0 <= idx < len(presets):
                    self.preset_index = idx
                    self._update_active_comparison(presets[idx][1])
                    self.render_view()

            p_combo.currentIndexChanged.connect(on_preset)
            toolbar.addWidget(p_combo)

        def open_selector():
            uids = [(s.series_id, s.group, s.label) for s in self.active_series]

            def apply_custom(picked):
                self.preset_index = None
                self._update_active_comparison(picked)
                self.render_view()

            series_selector_dialog(self, t, res, uids, apply_custom)

        choose_btn = QPushButton(t("choose_groups"))
        choose_btn.setObjectName("SmallBtn")
        choose_btn.clicked.connect(open_selector)
        toolbar.addWidget(choose_btn)

        if include_band:
            b_combo = QComboBox()
            b_combo.addItems([t(f"band_short_{b}") for b in BANDS])
            cur_b = BANDS.index(res.settings.band) if res.settings.band in BANDS else 0
            b_combo.setCurrentIndex(cur_b)

            def on_band(idx: int):
                new_band = BANDS[idx]
                res.settings = replace(res.settings, band=new_band)
                self.settings = replace(self.settings, band=new_band)
                res.summaries = [with_band(s, new_band, res.settings.seed) for s in res.summaries]
                res.custom_summaries = [with_band(s, new_band, res.settings.seed) for s in res.custom_summaries]
                self.active_series = [with_band(s, new_band, res.settings.seed) for s in self.active_series]
                self.render_view()

            b_combo.currentIndexChanged.connect(on_band)
            toolbar.addWidget(b_combo)

            samples_cb = QCheckBox(t("show_samples"))
            samples_cb.setChecked(self.show_samples)

            def on_samples(checked: bool):
                self.show_samples = checked
                self.render_view()

            samples_cb.toggled.connect(on_samples)
            toolbar.addWidget(samples_cb)

        if include_design:
            d_keys = ["design_auto", "design_independent", "design_paired"]
            d_combo = QComboBox()
            d_combo.addItems([t(k) for k in d_keys])
            cur_d = PAIRING_MODES.index(res.settings.pairing_mode) if res.settings.pairing_mode in PAIRING_MODES else 0
            d_combo.setCurrentIndex(cur_d)

            def on_design(idx: int):
                new_pm = PAIRING_MODES[idx]
                res.settings = replace(res.settings, pairing_mode=new_pm)
                self.settings = replace(self.settings, pairing_mode=new_pm)
                self._update_active_comparison(self.active_series)
                self.render_view()

            d_combo.currentIndexChanged.connect(on_design)
            toolbar.addWidget(d_combo)

            ps_combo = QComboBox()
            ps_combo.addItems([t("perm_welch"), t("perm_diff")])
            cur_ps = PERM_STATS.index(res.settings.perm_stat) if res.settings.perm_stat in PERM_STATS else 0
            ps_combo.setCurrentIndex(cur_ps)

            def on_pstat(idx: int):
                new_ps = PERM_STATS[idx]
                res.settings = replace(res.settings, perm_stat=new_ps)
                self.settings = replace(self.settings, perm_stat=new_ps)
                self._update_active_comparison(self.active_series)
                self.render_view()

            ps_combo.currentIndexChanged.connect(on_pstat)
            toolbar.addWidget(ps_combo)

        toolbar.addStretch(1)
        page_layout.addLayout(toolbar)

    def _format_omnibus_banner(self, om):
        t = self.t
        if om is None:
            return None
        sig = t("sig_Yes") if (np.isfinite(om.p_global) and om.p_global <= 0.05) else t("sig_No")
        return t(
            "omnibus_stat_line",
            design=t("design_tag_" + om.design),
            k=om.k_groups,
            p=om.p_global,
            sig=sig,
            ptest=t(om.parametric_test) if om.parametric_test else "—",
            pp=om.parametric_p,
            nptest=t(om.nonparametric_test) if om.nonparametric_test else "—",
            npp=om.nonparametric_p,
        )

    def render_view(self):
        t = self.t
        page = self.pages[self.view]
        page_layout = page.layout()
        self._clear_layout(page_layout)
        self.figure = None
        result = self.results

        if self.view == "view_groups":
            if result.summaries:
                if not self.active_series:
                    presets = self._build_presets()
                    self._update_active_comparison(presets[0][1] if presets else result.summaries)
                self._grouping_toolbar(page_layout, include_band=True, include_design=False)
                om = result.comparison.omnibus if result.comparison else None
                banner = self._format_omnibus_banner(om)
                if banner:
                    b_lbl = QLabel(banner)
                    b_lbl.setWordWrap(True)
                    b_lbl.setObjectName("BannerLabel")
                    page_layout.addWidget(b_lbl)
                notes = [t("group_caption") + "  ·  " + t("group_caption_n")]
                req_n = 3 if result.settings.band == "ci" else 2
                if result.settings.band != "off" and any(
                    s.n.min() < req_n for s in self.active_series if len(s.n)
                ):
                    notes.append(t("band_na"))
                fig = group_figure(
                    result,
                    self.active_series,
                    t,
                    band=result.settings.band,
                    show_samples=self.show_samples,
                    omnibus=om,
                )
                self._show_figure(page_layout, fig, notes)
            else:
                page_layout.addWidget(QLabel(t("no_summary")))
                page_layout.addStretch(1)

        elif self.view == "view_roi":
            self._render_roi_view(page_layout, result)

        elif self.view == "view_contrast":
            self._grouping_toolbar(page_layout, include_band=False, include_design=True)
            comp = result.comparison
            contrasts = result.contrasts or (comp.contrasts if comp else [])
            om = comp.omnibus if (not result.contrasts and comp) else None
            if contrasts:
                items = []
                if om is not None and om.k_groups >= 3:
                    items.append(("omnibus", om, t("omnibus_view_item", k=om.k_groups, p=om.p_global)))
                for c in contrasts:
                    items.append(
                        ("pair", c, t("pairwise_view_item", a=c.a, b=c.b, p=c.envelope.p, holm=c.p_adjusted))
                    )
                idx = min(self.selected.get("view_contrast", 0), len(items) - 1)
                if len(items) >= 2:
                    sel_combo = QComboBox()
                    sel_combo.addItems([it[2] for it in items])
                    sel_combo.setCurrentIndex(idx)
                    sel_combo.setMaximumWidth(560)

                    def on_item_select(new_i: int):
                        self.selected["view_contrast"] = new_i
                        self.render_view()

                    sel_combo.currentIndexChanged.connect(on_item_select)
                    page_layout.addWidget(sel_combo)

                kind, obj, _ = items[idx]
                if kind == "omnibus":
                    notes = []
                    banner = self._format_omnibus_banner(obj)
                    if banner:
                        notes.append(banner)
                    fig = omnibus_figure(result, obj, t)
                else:
                    c = obj
                    design_txt = t("design_tag_" + c.design)
                    n_txt = (
                        f"N={c.n_paired} paired"
                        if c.design == "paired"
                        else f"N_A={c.n_a}, N_B={c.n_b}" + (f" ({c.n_paired} shared)" if c.n_paired else "")
                    )
                    notes = [f"{c.a} vs {c.b}  ·  {design_txt} ({n_txt})"]
                    notes.extend(self.envelope_notes(c.envelope, c.p_adjusted))
                    fig = contrast_figure(result, c, t)

                self._show_figure(page_layout, fig, notes)
                self._build_stats_table(page_layout, om, contrasts, active_row=idx)
            else:
                msg = QLabel(t("stats_need_two_groups"))
                msg.setWordWrap(True)
                page_layout.addWidget(msg)
                page_layout.addStretch(1)

        else:
            self._build_qc_table(page_layout, result.qc)

        self.pdf_btn.setEnabled(self.figure is not None)

    def _render_roi_view(self, page_layout: QVBoxLayout, result):
        t = self.t
        all_patterns = result.patterns
        groups = list(dict.fromkeys(p.group for p in all_patterns if p.group != ""))
        types = list(dict.fromkeys(p.label for p in all_patterns))

        filter_bar = QHBoxLayout()
        if len(groups) >= 2:
            filter_bar.addWidget(QLabel(t("filter_group")))
            g_combo = QComboBox()
            g_items = [t("filter_all")] + groups
            g_combo.addItems(g_items)
            if self.roi_filter_group in groups:
                g_combo.setCurrentIndex(groups.index(self.roi_filter_group) + 1)

            def on_g(i: int):
                self.roi_filter_group = None if i == 0 else groups[i - 1]
                self.selected["view_roi"] = 0
                self.render_view()

            g_combo.currentIndexChanged.connect(on_g)
            filter_bar.addWidget(g_combo)

        if len(types) >= 2:
            filter_bar.addWidget(QLabel(t("filter_type")))
            t_combo = QComboBox()
            t_disp = [t("filter_all")] + [tp or t("all") for tp in types]
            t_combo.addItems(t_disp)
            if self.roi_filter_type is not None and self.roi_filter_type in types:
                t_combo.setCurrentIndex(types.index(self.roi_filter_type) + 1)

            def on_t(i: int):
                self.roi_filter_type = None if i == 0 else types[i - 1]
                self.selected["view_roi"] = 0
                self.render_view()

            t_combo.currentIndexChanged.connect(on_t)
            filter_bar.addWidget(t_combo)

        filtered = [
            p
            for p in all_patterns
            if (self.roi_filter_group is None or p.group == self.roi_filter_group)
            and (self.roi_filter_type is None or p.label == self.roi_filter_type)
        ]
        if not filtered:
            filtered = all_patterns

        idx = min(self.selected.get("view_roi", 0), len(filtered) - 1)
        roi_combo = QComboBox()
        roi_combo.addItems(
            [
                f"{p.group + ' · ' if p.group else ''}{p.sample} / {p.roi}  ·  {p.label or t('all')}"
                for p in filtered
            ]
        )
        roi_combo.setCurrentIndex(idx)
        roi_combo.setMinimumWidth(320)

        def on_roi(new_i: int):
            self.selected["view_roi"] = new_i
            self.render_view()

        roi_combo.currentIndexChanged.connect(on_roi)
        filter_bar.addWidget(roi_combo, stretch=1)
        page_layout.addLayout(filter_bar)

        p = filtered[idx]
        note = t("inference", status=t(p.status), null=t(p.null), n=p.count)
        if p.target_count is not None:
            note += t("n_targets", n=p.target_count)
        notes = [note + "  ·  " + t(p.window.correction)]
        notes.extend(self.envelope_notes(p.envelope, p.p_adjusted))
        shown = len(select_indices(len(p.xy)))
        if shown < len(p.xy):
            notes.append(t("point_count_drawn", n=shown, total=len(p.xy)))
        notes.extend(t(qc) for qc in p.qc)
        self._show_figure(page_layout, roi_figure(result, p, t), notes)

    def _build_stats_table(self, page_layout: QVBoxLayout, omnibus, contrasts, active_row: int = 0):
        t = self.t
        unit = self.results.data.calibration.unit
        hint = QLabel(t("click_row_hint"))
        hint.setObjectName("MutedLabel")
        page_layout.addWidget(hint)

        rows_data = []

        def fmt_p(p):
            if p is None or not np.isfinite(p):
                return "—"
            return f"{p:.4g}{' *' if p <= 0.05 else ''}"

        def fmt_spans(spans):
            if not spans:
                return "—"
            txt = ", ".join(f"{lo:g}–{hi:g}" for _, lo, hi in spans[:2])
            return f"{txt} {unit}" + (" …" if len(spans) > 2 else "")

        if omnibus is not None and omnibus.k_groups >= 3:
            rows_data.append(
                (
                    f"[Omnibus] {' / '.join(omnibus.series_names)}",
                    f"{t('design_tag_' + omnibus.design)} ({omnibus.k_groups}g)",
                    fmt_p(omnibus.p_global),
                    fmt_p(omnibus.p_global),
                    fmt_spans(omnibus.significant_spans),
                    f"{t(omnibus.parametric_test)}: {fmt_p(omnibus.parametric_p)}" if omnibus.parametric_test else "—",
                    f"{t(omnibus.nonparametric_test)}: {fmt_p(omnibus.nonparametric_p)}"
                    if omnibus.nonparametric_test
                    else "—",
                )
            )
        for c in contrasts:
            spans = interval_strings(self.results.radii, c.envelope.side)
            d_str = (
                f"{t('design_tag_paired')} (N={c.n_paired})"
                if c.design == "paired"
                else f"{t('design_tag_' + c.design)} ({c.n_a},{c.n_b})"
            )
            p_used = (
                c.parametric_p_holm
                if len(contrasts) > 1 and np.isfinite(c.parametric_p_holm)
                else c.parametric_p
            )
            np_used = (
                c.nonparametric_p_holm
                if len(contrasts) > 1 and np.isfinite(c.nonparametric_p_holm)
                else c.nonparametric_p
            )
            rows_data.append(
                (
                    f"{c.a} − {c.b}",
                    d_str,
                    fmt_p(c.envelope.p),
                    fmt_p(c.p_adjusted),
                    fmt_spans(spans),
                    f"{t(c.parametric_test)}: {fmt_p(p_used)}" if c.parametric_test else "—",
                    f"{t(c.nonparametric_test)}: {fmt_p(np_used)}" if c.nonparametric_test else "—",
                )
            )

        headers = [
            t("col_comparison"),
            t("col_design"),
            t("col_p_global"),
            t("col_p_holm"),
            t("col_sig_r"),
            t("col_param"),
            t("col_nonparam"),
        ]
        table = QTableWidget(len(rows_data), len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)
        table.verticalHeader().setDefaultSectionSize(25)
        table.setMaximumHeight(min(185, 34 + 26 * len(rows_data)))

        for r_idx, r_vals in enumerate(rows_data):
            for c_idx, val in enumerate(r_vals):
                table.setItem(r_idx, c_idx, QTableWidgetItem(val))

        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        for col in (1, 2, 3, 5, 6):
            table.resizeColumnToContents(col)
        if 0 <= active_row < len(rows_data):
            table.selectRow(active_row)

        def on_row_clicked(row_idx: int, _col: int):
            if row_idx != self.selected.get("view_contrast", 0):
                self.selected["view_contrast"] = row_idx
                self.render_view()

        table.cellClicked.connect(on_row_clicked)
        page_layout.addWidget(table)

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
            notes.append(
                "  ·  ".join(
                    t("scale_span", direction=t(direction), lo=lo, hi=hi, unit=self.results.data.calibration.unit)
                    for direction, lo, hi in spans[:4]
                )
                + ("  …" if len(spans) > 4 else "")
            )
            if adjusted > .05:
                notes.append(t("p_family_note"))
        return notes

    def _show_figure(self, layout: QVBoxLayout, figure, notes: list[str]):
        self.figure = figure
        for note in notes:
            lbl = QLabel(note)
            lbl.setWordWrap(True)
            lbl.setObjectName("MutedLabel")
            layout.addWidget(lbl)
        canvas = FigureCanvasQTAgg(figure)
        toolbar = PlotToolbar(canvas, self, self.t)
        layout.addWidget(canvas, stretch=1)
        layout.addWidget(toolbar)
        canvas.draw_idle()

    def _build_qc_table(self, layout: QVBoxLayout, records):
        t = self.t
        if records.empty:
            layout.addWidget(QLabel(t("no_qc")))
            layout.addStretch(1)
            return

        search_edit = QLineEdit(self.qc_search_text)
        search_edit.setPlaceholderText(t("search_placeholder"))
        layout.addWidget(search_edit)

        headers = [t("name"), t("row"), t("sample_id"), t("roi_id"), t("reason")]
        cols = ["source", "row", "sample_id", "roi_id", "reason"]
        subset = records.head(1000).reindex(columns=cols).fillna("")
        rows = [
            (Path(src).name if src else "", str(row), str(sample), str(roi), t(str(reason)))
            for src, row, sample, roi, reason in subset.itertuples(index=False, name=None)
        ]

        table = QTableWidget(len(rows), len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setAlternatingRowColors(True)
        table.setSortingEnabled(True)
        table.verticalHeader().setVisible(False)
        table.verticalHeader().setDefaultSectionSize(25)
        for r_idx, r_vals in enumerate(rows):
            for c_idx, val in enumerate(r_vals):
                table.setItem(r_idx, c_idx, QTableWidgetItem(val))
        table.resizeColumnsToContents()
        table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)

        def filter_rows(query: str):
            self.qc_search_text = query
            q = query.strip().casefold()
            for r_idx in range(table.rowCount()):
                match = not q or any(
                    q in (table.item(r_idx, c).text().casefold() if table.item(r_idx, c) else "")
                    for c in range(table.columnCount())
                )
                table.setRowHidden(r_idx, not match)

        search_edit.textChanged.connect(filter_rows)
        if self.qc_search_text:
            filter_rows(self.qc_search_text)

        if len(records) > 1000:
            lim = QLabel(t("qc_limit", count=len(records)))
            lim.setObjectName("MutedLabel")
            layout.addWidget(lim)
        layout.addWidget(table, stretch=1)

    def export_pdf(self):
        if self.figure is None:
            return
        t = self.t
        filename, _ = QFileDialog.getSaveFileName(
            self,
            t("export_figure"),
            "spatial-pattern.pdf",
            "PDF (*.pdf);;SVG (*.svg);;PNG 300 DPI (*.png)",
        )
        if filename:
            try:
                save_figure(self.figure, filename, t("title"))
                self.set_status("saved", file=Path(filename).name)
            except Exception as error:
                show_error(self, t, error)

    def clear_results(self):
        if self.job is None:
            self.results = None
            self.active_series.clear()
            self.set_status("ready")
            self.render_main()

    def export_tables(self):
        if self.results is None:
            return
        t = self.t
        filename, _ = QFileDialog.getSaveFileName(
            self,
            t("data_export"),
            "spatial-results.xlsx",
            "Excel (*.xlsx);;CSV ZIP (*.zip)",
        )
        if filename:
            try:
                export_data(self.results, filename)
                self.set_status("saved", file=Path(filename).name)
            except Exception as error:
                show_error(self, t, error)

    def export_batch(self):
        if self.results is None:
            return
        t = self.t
        folder = QFileDialog.getExistingDirectory(self, t("batch_export_title"))
        if folder:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                written = export_batch_bundle(
                    self.results,
                    folder,
                    t,
                    band=self.results.settings.band,
                    show_samples=self.show_samples,
                )
                self.set_status("batch_saved", n=len(written), folder=Path(folder).name)
            except Exception as error:
                show_error(self, t, error)
            finally:
                QApplication.restoreOverrideCursor()

    def methods(self):
        document = ASSETS / ("README.zh-TW.md" if self.t.lang == "zh-TW" else "README.md")
        methods_dialog(self, self.t, document)


def smoke_test(path: Path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    try:
        if sys.platform != "win32" and not os.environ.get("DISPLAY"):
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        if getattr(sys, "frozen", False):
            import xlrd

            assert xlrd.__version__
        app = QApplication.instance() or QApplication([])
        window = SpatialAnalysisGUI("en")
        assert not QPixmap(str(ASSETS / "ui" / "checked.svg")).isNull()
        guide = ROIGuide(window, "zh-TW")
        assert not guide.grab().isNull()
        window.data = demo_data()
        window._configure_defaults_for_dataset()
        window.results = run_analysis(window.data, Settings(roi_mode="custom", simulations=19))
        one_field = replace(window.data, points=window.data.points.iloc[:72], fields=window.data.fields[:1])
        for mode in ("infer", "rectangle"):
            reference = run_analysis(one_field, Settings(roi_mode=mode, simulations=19)).patterns[0].envelope
            assert np.isfinite(reference.lower).any() and np.isnan(reference.p)
        window.render_main()
        for lang in ("en", "zh-TW"):
            t = Translator(lang)
            save_pdf(group_figure(window.results, "", t), path / f"groups-{lang}.pdf", t("title"))
        export_data(window.results, path / "results.xlsx")
        app.processEvents()
        window.close()
        window.deleteLater()
        app.processEvents()
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
    app = QApplication(sys.argv)
    language = locale.getlocale()[0] or ""
    zh = language.casefold().startswith(("zh_tw", "zh_hk", "chinese (traditional)"))
    window = SpatialAnalysisGUI("zh-TW" if zh else "en")
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
