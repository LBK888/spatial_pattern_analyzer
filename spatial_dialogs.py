"""PyQt6 dialogs and reusable form controls for Spatial Pattern Analyzer."""
from __future__ import annotations

from dataclasses import replace
import html
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from shapely.geometry import Polygon

from ripley_analysis import IntensityGrid, make_window, valid_bounds_box
from spatial_data import (
    COLUMNS,
    RECT_COLUMNS,
    Calibration,
    ReadOptions,
    column_labels,
    default_excel_sheets,
    detect_block_width,
    excel_sheets,
    infer_mapping,
    read_table,
)
from spatial_plots import PALETTE, select_indices, spatial_axes, use_cjk_font
from spatial_workflow import build_custom_summary, merged_label, sorted_types


class PlotToolbar(NavigationToolbar2QT):
    def __init__(self, canvas, parent, t):
        names = {
            "Home": "plot_home",
            "Back": "plot_back",
            "Forward": "plot_forward",
            "Pan": "plot_pan",
            "Zoom": "plot_zoom",
        }
        self.toolitems = [
            (name, t(names[name]), icon, method)
            for name, _, icon, method in self.toolitems
            if name in names
        ]
        super().__init__(canvas, parent)
        self.setIconSize(self.iconSize() * 0.85)
        self.setStyleSheet("QToolBar { border: none; background: transparent; spacing: 4px; }")


def rich_tip(text: str) -> str:
    escaped = html.escape(str(text)).replace("\n", "<br>")
    return f"<div style='max-width:320px; line-height:1.4;'>{escaped}</div>"


def help_button(parent: QWidget, text: str) -> QToolButton:
    btn = QToolButton(parent)
    btn.setText("ⓘ")
    btn.setObjectName("HelpBtn")
    btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
    tip = rich_tip(text)
    btn.setToolTip(tip)
    btn.clicked.connect(
        lambda: QToolTip.showText(btn.mapToGlobal(QPoint(0, btn.height() + 2)), tip, btn)
    )
    return btn


def labeled_row(layout: QVBoxLayout, t, label_key: str, widget: QWidget, help_key: str | None = None):
    row = QHBoxLayout()
    row.setContentsMargins(0, 2, 0, 2)
    lbl_wrap = QHBoxLayout()
    lbl_wrap.setSpacing(4)
    lbl = QLabel(t(label_key))
    lbl_wrap.addWidget(lbl)
    lbl_wrap.addWidget(help_button(widget, t(help_key or f"{label_key}_help")))
    lbl_wrap.addStretch(1)
    row.addLayout(lbl_wrap, stretch=3)
    row.addWidget(widget, stretch=2)
    layout.addLayout(row)
    return widget


def make_entry(layout: QVBoxLayout, t, label_key: str, value="", help_key: str | None = None) -> QLineEdit:
    edit = QLineEdit(str(value))
    edit.setMinimumWidth(110)
    return labeled_row(layout, t, label_key, edit, help_key)


def make_choice(
    layout: QVBoxLayout,
    t,
    label_key: str,
    values: list[str],
    current_value: str = "",
    help_key: str | None = None,
    translated: bool = True,
) -> QComboBox:
    combo = QComboBox()
    display = [t(x) if translated else str(x) for x in values]
    combo.addItems(display)
    if current_value in values:
        combo.setCurrentIndex(values.index(current_value))
    elif values:
        combo.setCurrentIndex(0)
    return labeled_row(layout, t, label_key, combo, help_key)


def make_flag(layout: QVBoxLayout, t, label_key: str, checked: bool, help_key: str | None = None) -> QCheckBox:
    row = QHBoxLayout()
    row.setContentsMargins(0, 2, 0, 2)
    cb = QCheckBox(t(label_key))
    cb.setChecked(bool(checked))
    row.addWidget(cb)
    row.addWidget(help_button(cb, t(help_key or f"{label_key}_help")))
    row.addStretch(1)
    layout.addLayout(row)
    return cb


def dialog_buttons(dialog: QDialog, layout: QVBoxLayout, t, accept_cb, accept_key: str = "accept"):
    bar = QHBoxLayout()
    bar.setContentsMargins(0, 10, 0, 0)
    bar.addStretch(1)
    cancel_btn = QPushButton(t("cancel"))
    cancel_btn.clicked.connect(dialog.reject)
    ok_btn = QPushButton(t(accept_key))
    ok_btn.setObjectName("PrimaryBtn")
    ok_btn.setDefault(True)
    ok_btn.clicked.connect(accept_cb)
    bar.addWidget(cancel_btn)
    bar.addWidget(ok_btn)
    layout.addLayout(bar)


def parse_number(edit: QLineEdit, kind, t, field_key: str, optional: bool = False):
    value = edit.text().strip()
    if optional and not value:
        return None
    try:
        return kind(value)
    except ValueError as error:
        raise ValueError(t("numeric_value", field=t(field_key))) from error


def show_error(parent: QWidget | None, t, error: Exception):
    QMessageBox.critical(parent, t("failed"), t.error(error))


class ImportDialog(QDialog):
    """Multi-sheet, block-unstacking, column-mapping & coordinate-calibration dialog."""

    def __init__(self, parent, t, paths, callback, initial_state=None):
        super().__init__(parent)
        self.t = t
        self.paths = tuple(paths)
        self.callback = callback
        self.setWindowTitle(t("import_title"))
        self.resize(1080, 700)
        self.setMinimumSize(820, 540)

        self.is_excel = Path(self.paths[0]).suffix.lower() in (".xls", ".xlsx")
        self.all_sheets: list[str] = []
        self.sheet_checks: dict[str, QCheckBox] = {}
        self.combos: dict[str, QComboBox] = {}
        self.more_combos: dict[str, QComboBox] = {}
        self.labels: list[str] = []
        self._suppress_refresh = True

        if initial_state is not None:
            self.options, init_map, self.calibration, self.duplicates = initial_state
            self.mapping = dict(init_map)
            if self.is_excel:
                try:
                    self.all_sheets = excel_sheets(self.paths[0])
                except Exception:
                    self.all_sheets = []
        else:
            self.calibration = Calibration()
            self.duplicates = "keep"
            self.mapping = {}
            initial_sheet = 0
            initial_sheets = None
            initial_sheet_group = False
            initial_unstack = False
            if self.is_excel:
                try:
                    self.all_sheets = excel_sheets(self.paths[0])
                    auto = default_excel_sheets(self.paths[0], header=True)
                    initial_sheets = auto
                    initial_sheet = auto[0] if auto else self.all_sheets[0]
                    initial_sheet_group = len(auto) > 1
                    probe = pd.read_excel(
                        self.paths[0], sheet_name=initial_sheet, header=None,
                        dtype=str, keep_default_na=False, nrows=8
                    )
                    initial_unstack = detect_block_width(probe, True) is not None
                except Exception:
                    initial_sheet = 0
            self.options = ReadOptions(
                sheet=initial_sheet,
                sheets=initial_sheets,
                sheet_as_group=initial_sheet_group,
                unstack_blocks=initial_unstack,
            )

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(16, 14, 16, 14)

        top_bar = QHBoxLayout()
        title_lbl = QLabel(t("batch_files", count=len(self.paths)))
        title_lbl.setObjectName("SectionLabel")
        top_bar.addWidget(title_lbl)
        top_bar.addStretch(1)
        self.cal_summary_lbl = QLabel("")
        self.cal_summary_lbl.setObjectName("MutedLabel")
        top_bar.addWidget(self.cal_summary_lbl)
        root_layout.addLayout(top_bar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter, stretch=1)

        # Left preview pane
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 4, 8, 0)
        self.preview_title = QLabel("")
        self.preview_title.setObjectName("MutedLabel")
        left_layout.addWidget(self.preview_title)
        self.preview_table = QTableWidget()
        self.preview_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.verticalHeader().setDefaultSectionSize(25)
        left_layout.addWidget(self.preview_table, stretch=1)
        splitter.addWidget(left)

        # Right inspector tabs: [Sheets & Columns] | [Calibration & Format]
        right_tabs = QTabWidget()
        splitter.addWidget(right_tabs)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        # --- Tab 1: Sheets & Columns ---
        map_scroll = QScrollArea()
        map_scroll.setWidgetResizable(True)
        map_scroll.setFrameShape(QFrame.Shape.NoFrame)
        map_container = QWidget()
        self.map_tab_layout = QVBoxLayout(map_container)
        self.map_tab_layout.setContentsMargins(10, 10, 10, 10)
        map_scroll.setWidget(map_container)
        right_tabs.addTab(map_scroll, t("tab_mapping"))

        if self.is_excel and self.all_sheets:
            ws_group = QGroupBox(t("worksheets"))
            ws_layout = QVBoxLayout(ws_group)
            btn_row = QHBoxLayout()
            btn_row.addWidget(help_button(ws_group, t("worksheets_help")))
            btn_row.addStretch(1)
            if len(self.all_sheets) > 1:
                sel_all_btn = QPushButton(t("select_all_sheets"))
                sel_all_btn.setObjectName("SmallBtn")
                sel_all_btn.clicked.connect(lambda: self._set_all_sheets(True))
                clr_all_btn = QPushButton(t("deselect_all_sheets"))
                clr_all_btn.setObjectName("SmallBtn")
                clr_all_btn.clicked.connect(lambda: self._set_all_sheets(False))
                btn_row.addWidget(sel_all_btn)
                btn_row.addWidget(clr_all_btn)
            ws_layout.addLayout(btn_row)

            grid = QGridLayout()
            chosen = set(self.options.sheets or [self.options.sheet])
            for idx, sh_name in enumerate(self.all_sheets):
                cb = QCheckBox(sh_name)
                cb.setChecked(sh_name in chosen)
                cb.toggled.connect(self._on_sheet_toggled)
                self.sheet_checks[sh_name] = cb
                grid.addWidget(cb, idx // 2, idx % 2)
            ws_layout.addLayout(grid)

            if len(self.all_sheets) > 1:
                cur_sh = (
                    self.all_sheets[self.options.sheet]
                    if isinstance(self.options.sheet, int)
                    else str(self.options.sheet)
                )
                self.preview_sheet_combo = make_choice(
                    ws_layout, t, "preview_sheet", self.all_sheets, cur_sh, translated=False
                )
                self.preview_sheet_combo.currentIndexChanged.connect(self._on_preview_sheet_changed)
            else:
                self.preview_sheet_combo = None

            self.sheet_group_cb = make_flag(ws_layout, t, "sheet_as_group", self.options.sheet_as_group)
            self.unstack_cb = make_flag(ws_layout, t, "unstack_blocks", self.options.unstack_blocks)
            self.unstack_cb.toggled.connect(lambda _: self.refresh(keep_mapping=False))
            self.map_tab_layout.addWidget(ws_group)
        else:
            self.preview_sheet_combo = None
            self.sheet_group_cb = None
            self.unstack_cb = make_flag(self.map_tab_layout, t, "unstack_blocks", self.options.unstack_blocks)
            self.unstack_cb.toggled.connect(lambda _: self.refresh(keep_mapping=False))

        self.header_cb = make_flag(self.map_tab_layout, t, "has_header", self.options.header)
        self.header_cb.toggled.connect(lambda _: self.refresh(keep_mapping=False))

        self.primary_map_box = QGroupBox(t("import_title"))
        self.primary_map_layout = QVBoxLayout(self.primary_map_box)
        self.map_tab_layout.addWidget(self.primary_map_box)

        self.extra_map_box = QGroupBox(t("more_columns"))
        self.extra_map_box.setCheckable(True)
        self.extra_map_box.setChecked(any(k in self.mapping for k in COLUMNS[6:]))
        self.extra_map_layout = QVBoxLayout(self.extra_map_box)
        self.map_tab_layout.addWidget(self.extra_map_box)
        self.map_tab_layout.addStretch(1)

        # --- Tab 2: Coordinate Calibration & Format ---
        cal_container = QWidget()
        cal_layout = QVBoxLayout(cal_container)
        cal_layout.setContentsMargins(12, 12, 12, 12)
        right_tabs.addTab(cal_container, t("tab_calibration"))

        cal_box = QGroupBox(t("coordinates").rstrip("…"))
        cal_box_layout = QVBoxLayout(cal_box)
        self.x_scale_edit = make_entry(cal_box_layout, t, "x_scale", self.calibration.x, "scale_help")
        self.y_scale_edit = make_entry(cal_box_layout, t, "y_scale", self.calibration.y, "scale_help")
        self.unit_edit = make_entry(cal_box_layout, t, "unit", self.calibration.unit)

        unit_bar = QHBoxLayout()
        unit_bar.addWidget(QLabel(t("unit_presets")))
        for u in ("units", "µm", "nm", "mm", "px"):
            ub = QPushButton(u)
            ub.setObjectName("SmallBtn")
            ub.clicked.connect(lambda _, val=u: self._set_unit(val))
            unit_bar.addWidget(ub)
        unit_bar.addStretch(1)
        cal_box_layout.addLayout(unit_bar)

        self.y_down_cb = make_flag(cal_box_layout, t, "y_down", self.calibration.y_down)
        self.dup_values = ["keep", "drop"]
        self.dup_combo = make_choice(cal_box_layout, t, "duplicates", self.dup_values, self.duplicates)
        for w in (self.x_scale_edit, self.y_scale_edit, self.unit_edit):
            w.textChanged.connect(self._update_cal_summary)
        self.y_down_cb.toggled.connect(self._update_cal_summary)
        cal_layout.addWidget(cal_box)

        fmt_box = QGroupBox(t("file_format"))
        fmt_layout = QVBoxLayout(fmt_box)
        self.decimal_combo = make_choice(fmt_layout, t, "decimal", [".", ","], self.options.decimal, translated=False)
        self.delimiters = ("auto", ",", ";", "\t")
        self.delim_names = ["auto", "comma", "semicolon", "tab"]
        self.encodings = ("auto", "utf-8-sig", "cp950", "utf-16")
        if not self.is_excel:
            cur_d = self.delim_names[self.delimiters.index(self.options.delimiter)]
            self.sep_combo = make_choice(fmt_layout, t, "delimiter", self.delim_names, cur_d)
            enc_disp = [t("auto"), "UTF-8", "CP950", "UTF-16"]
            cur_e = enc_disp[self.encodings.index(self.options.encoding)]
            self.enc_combo = make_choice(fmt_layout, t, "encoding", enc_disp, cur_e, translated=False)
        else:
            self.sep_combo = None
            self.enc_combo = None
        refresh_btn = QPushButton(t("apply_preview"))
        refresh_btn.clicked.connect(lambda: self.refresh(keep_mapping=True))
        fmt_layout.addWidget(refresh_btn)
        cal_layout.addWidget(fmt_box)
        cal_layout.addStretch(1)

        dialog_buttons(self, root_layout, t, self._on_accept, "load")

        self._suppress_refresh = False
        self._update_cal_summary()
        self.refresh(keep_mapping=bool(initial_state))

    def _set_unit(self, unit_str: str):
        self.unit_edit.setText(unit_str)
        self._update_cal_summary()

    def _update_cal_summary(self):
        unit = self.unit_edit.text().strip() or "units"
        xs = self.x_scale_edit.text().strip() or "1"
        ys = self.y_scale_edit.text().strip() or "1"
        yd = " · Y↓" if self.y_down_cb.isChecked() else ""
        self.cal_summary_lbl.setText(f"Scale: {xs} × {ys} {unit}{yd}")

    def _set_all_sheets(self, state: bool):
        self._suppress_refresh = True
        for cb in self.sheet_checks.values():
            cb.setChecked(state)
        if state and len(self.sheet_checks) > 1 and self.sheet_group_cb is not None:
            self.sheet_group_cb.setChecked(True)
        self._suppress_refresh = False
        self._sync_options()

    def _on_sheet_toggled(self):
        if self._suppress_refresh:
            return
        selected = [sh for sh, cb in self.sheet_checks.items() if cb.isChecked()]
        if len(selected) > 1 and self.sheet_group_cb is not None and "group" not in self.mapping:
            self.sheet_group_cb.setChecked(True)
        self._sync_options()
        if selected and self.options.sheet not in selected:
            self.options = replace(self.options, sheet=selected[0])
            if self.preview_sheet_combo is not None:
                self._suppress_refresh = True
                self.preview_sheet_combo.setCurrentIndex(self.all_sheets.index(selected[0]))
                self._suppress_refresh = False
            self.refresh(keep_mapping=True)

    def _on_preview_sheet_changed(self):
        if self._suppress_refresh or self.preview_sheet_combo is None:
            return
        chosen = self.all_sheets[self.preview_sheet_combo.currentIndex()]
        self.options = replace(self.options, sheet=chosen)
        self.refresh(keep_mapping=True)

    def _sync_options(self):
        selected = tuple(sh for sh, cb in self.sheet_checks.items() if cb.isChecked()) if self.is_excel else None
        sheet_grp = self.sheet_group_cb.isChecked() if self.sheet_group_cb is not None else False
        unstack = self.unstack_cb.isChecked() if self.unstack_cb is not None else False
        delim = self.delimiters[self.sep_combo.currentIndex()] if self.sep_combo is not None else "auto"
        enc = self.encodings[self.enc_combo.currentIndex()] if self.enc_combo is not None else "auto"
        self.options = replace(
            self.options,
            header=self.header_cb.isChecked(),
            decimal=self.decimal_combo.currentText(),
            delimiter=delim,
            encoding=enc,
            sheets=selected or None,
            sheet_as_group=sheet_grp,
            unstack_blocks=unstack,
        )

    def _collect_mapping(self):
        for key, combo in {**self.combos, **self.more_combos}.items():
            self.mapping.pop(key, None)
            if combo.currentIndex() > 0:
                self.mapping[key] = combo.currentIndex() - 1
        if not self.extra_map_box.isChecked():
            for key in COLUMNS[6:]:
                self.mapping.pop(key, None)

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            sub = item.layout()
            if w is not None:
                w.deleteLater()
            elif sub is not None:
                self._clear_layout(sub)

    def refresh(self, keep_mapping: bool = False):
        if self._suppress_refresh:
            return
        t = self.t
        prev_map = {}
        if keep_mapping and (self.combos or self.mapping):
            if self.combos:
                self._collect_mapping()
            prev_map = dict(self.mapping)
        self._sync_options()
        self._clear_layout(self.primary_map_layout)
        self._clear_layout(self.extra_map_layout)
        self.combos.clear()
        self.more_combos.clear()
        self.preview_table.clear()
        self.preview_table.setRowCount(0)
        self.preview_table.setColumnCount(0)

        path = self.paths[0]
        try:
            raw = read_table(path, self.options, limit=51, sheet=self.options.sheet)
            if raw.empty:
                raise ValueError("empty_file")
        except Exception as error:
            self.preview_title.setText(t.error(error))
            return

        self.labels = column_labels(raw, self.options.header)
        if prev_map and all(0 <= idx < len(self.labels) for idx in prev_map.values()):
            self.mapping = prev_map
        else:
            self.mapping = infer_mapping(self.labels, raw, self.options.header, self.options.decimal)

        file_tag = Path(path).name + (f" [{self.options.sheet}]" if self.is_excel and self.all_sheets else "")
        self.preview_title.setText(t("max_preview", file=file_tag))
        data_rows = raw.iloc[int(self.options.header):][:50]
        self.preview_table.setColumnCount(len(self.labels))
        self.preview_table.setHorizontalHeaderLabels(self.labels)
        self.preview_table.setRowCount(len(data_rows))
        for r_idx, row in enumerate(data_rows.itertuples(index=False, name=None)):
            for c_idx, val in enumerate(row):
                self.preview_table.setItem(r_idx, c_idx, QTableWidgetItem(str(val)))
        self.preview_table.resizeColumnsToContents()
        for c_idx in range(len(self.labels)):
            if self.preview_table.columnWidth(c_idx) > 180:
                self.preview_table.setColumnWidth(c_idx, 180)

        choices = [t("none")] + [f"{i + 1} · {lbl}" for i, lbl in enumerate(self.labels)]
        for field in COLUMNS[:6]:
            idx = self.mapping.get(field, -1) + 1
            cur = choices[idx] if 0 <= idx < len(choices) else choices[0]
            self.combos[field] = make_choice(self.primary_map_layout, t, field, choices, cur, translated=False)
        for field in COLUMNS[6:]:
            idx = self.mapping.get(field, -1) + 1
            cur = choices[idx] if 0 <= idx < len(choices) else choices[0]
            hkey = "bounds_columns_help" if field in RECT_COLUMNS else None
            self.more_combos[field] = make_choice(
                self.extra_map_layout, t, field, choices, cur, help_key=hkey, translated=False
            )

    def _on_accept(self):
        t = self.t
        try:
            self._collect_mapping()
            self._sync_options()
            scales = [
                parse_number(self.x_scale_edit, float, t, "x_scale"),
                parse_number(self.y_scale_edit, float, t, "y_scale"),
            ]
            if not np.all(np.isfinite(scales)) or min(scales) <= 0:
                raise ValueError("invalid_scale")
            self.calibration = Calibration(
                scales[0],
                scales[1],
                self.unit_edit.text().strip() or "units",
                self.y_down_cb.isChecked(),
            )
            self.duplicates = self.dup_values[self.dup_combo.currentIndex()]
            if self.is_excel and self.sheet_checks:
                selected = tuple(sh for sh, cb in self.sheet_checks.items() if cb.isChecked())
                if not selected:
                    selected = (self.options.sheet,)
                self.options = replace(self.options, sheets=selected, sheet=selected[0])
            if (
                self.mapping.get("x") is None
                or self.mapping.get("y") is None
                or self.mapping["x"] == self.mapping["y"]
            ):
                raise ValueError("select_xy")
        except ValueError as error:
            show_error(self, t, error)
            return
        self.accept()
        self.callback(self.paths, self.options, dict(self.mapping), self.calibration, self.duplicates)


def advanced_dialog(parent, t, data, settings, intensity_path, callback):
    dialog = QDialog(parent)
    dialog.setWindowTitle(t("advanced"))
    dialog.setMinimumSize(620, 490)
    root = QVBoxLayout(dialog)
    root.setContentsMargins(16, 14, 16, 14)

    tabs = QTabWidget()
    root.addWidget(tabs, stretch=1)

    # Tab 1: Distance range & Edge correction
    dist_page = QWidget()
    dist_layout = QVBoxLayout(dist_page)
    dist_layout.setContentsMargins(12, 12, 12, 12)
    edge_keys = ["auto", "translation", "isotropic", "border"]
    edge_labels = ["edge_auto", "edge_translation", "edge_isotropic", "edge_border"]
    cur_edge = settings.edge_correction if settings.edge_correction in edge_keys else "auto"
    edge_combo = make_choice(
        dist_layout, t, "edge_correction", edge_labels, edge_labels[edge_keys.index(cur_edge)]
    )
    use_fixed_cb = make_flag(dist_layout, t, "use_fixed_r", settings.r_step is not None)
    keys = ["r_min", "r_max", "r_step", "n_radii", "local_radius"]
    if (settings.roi_mode == "infer" and not settings.convex_guard) or (
        settings.roi_mode == "rectangle" and settings.rectangle_hull == "concave"
    ):
        keys.insert(-1, "hull_ratio")
    scale_edits = {
        key: make_entry(
            dist_layout, t, key, "" if getattr(settings, key) is None else getattr(settings, key)
        )
        for key in keys
    }
    dist_layout.addStretch(1)
    tabs.addTab(dist_page, t("distance"))

    # Tab 2: Pattern test
    test_page = QWidget()
    test_layout = QVBoxLayout(test_page)
    test_layout.setContentsMargins(12, 12, 12, 12)
    run_test_cb = make_flag(test_layout, t, "run_test", settings.test)
    sim_edit = make_entry(test_layout, t, "simulations", settings.simulations)
    seed_edit = make_entry(test_layout, t, "seed", settings.seed)
    workers_edit = make_entry(test_layout, t, "workers", settings.workers)
    modes = ["all", "each", "each_and_merged", "cross", "cross_all"]
    cur_mark = settings.marks if settings.marks in modes else "each_and_merged"
    marks_combo = make_choice(test_layout, t, "marks", modes, cur_mark)
    merge_req_cb = make_flag(test_layout, t, "merge_require_all", settings.merge_require_all)
    categories = sorted_types(data.points.cell_type) if data is not None else []
    focal_combo = make_choice(
        test_layout, t, "from_type", categories, settings.from_type, "cross_help", translated=False
    )
    neighbor_combo = make_choice(
        test_layout, t, "to_type", categories, settings.to_type, "cross_help", translated=False
    )
    if len(categories) > 1 and not settings.to_type:
        neighbor_combo.setCurrentIndex(1)

    def update_cross_state():
        enabled = marks_combo.currentIndex() == 3
        focal_combo.setEnabled(enabled)
        neighbor_combo.setEnabled(enabled)

    marks_combo.currentIndexChanged.connect(update_cross_state)
    update_cross_state()

    intensity = settings.intensity
    cur_intensity_path = intensity_path
    grid_row = QHBoxLayout()
    grid_btn = QPushButton(t("intensity"))
    grid_row.addWidget(grid_btn)
    grid_row.addWidget(help_button(test_page, t("intensity_help")))
    grid_row.addStretch(1)
    clr_btn = QPushButton(t("clear_file"))
    grid_row.addWidget(clr_btn)
    test_layout.addLayout(grid_row)
    grid_lbl = QLabel(Path(cur_intensity_path).name if cur_intensity_path else t("none"))
    grid_lbl.setObjectName("MutedLabel")
    test_layout.addWidget(grid_lbl)
    test_layout.addStretch(1)

    def open_grid():
        nonlocal intensity, cur_intensity_path
        path, _ = QFileDialog.getOpenFileName(dialog, t("intensity"), "", "CSV (*.csv)")
        if path:
            try:
                intensity = IntensityGrid(np.loadtxt(path, delimiter=",", ndmin=2, encoding="utf-8-sig"))
                cur_intensity_path = path
                grid_lbl.setText(Path(path).name)
            except (OSError, ValueError):
                show_error(dialog, t, ValueError("invalid_intensity"))

    def clear_grid():
        nonlocal intensity, cur_intensity_path
        intensity, cur_intensity_path = None, ""
        grid_lbl.setText(t("none"))

    grid_btn.clicked.connect(open_grid)
    clr_btn.clicked.connect(clear_grid)
    tabs.addTab(test_page, t("tests"))

    # Tab 3: Group summaries & biostatistics
    grp_page = QWidget()
    grp_layout = QVBoxLayout(grp_page)
    grp_layout.setContentsMargins(12, 12, 12, 12)
    bands = ["sd", "se", "ci", "off"]
    band_combo = make_choice(grp_layout, t, "band", bands, settings.band if settings.band in bands else "sd")
    p_modes = ["auto", "independent", "paired"]
    p_labels = ["design_auto", "design_independent", "design_paired"]
    cur_pm = settings.pairing_mode if settings.pairing_mode in p_modes else "auto"
    pairing_combo = make_choice(grp_layout, t, "pairing_mode", p_labels, p_labels[p_modes.index(cur_pm)])
    perm_keys = ["welch", "diff"]
    perm_labels = ["perm_welch", "perm_diff"]
    cur_ps = settings.perm_stat if settings.perm_stat in perm_keys else "welch"
    perm_combo = make_choice(grp_layout, t, "perm_stat", perm_labels, perm_labels[perm_keys.index(cur_ps)])
    contrast_cb = make_flag(grp_layout, t, "contrast", settings.contrast)
    paired_cb = make_flag(grp_layout, t, "paired", settings.paired)
    grp_names = sorted(set(data.points.group) - {""}) if data is not None else []
    a_combo = make_choice(grp_layout, t, "group_a", grp_names, settings.group_a, "group_choice_help", translated=False)
    b_combo = make_choice(grp_layout, t, "group_b", grp_names, settings.group_b, "group_choice_help", translated=False)
    if len(grp_names) > 1 and not settings.group_b:
        b_combo.setCurrentIndex(1)
    grp_layout.addStretch(1)
    tabs.addTab(grp_page, t("groups"))

    def on_apply():
        try:
            values = {
                key: parse_number(
                    edit,
                    int if key == "n_radii" else float,
                    t,
                    key,
                    optional=key in ("r_min", "r_max", "r_step", "local_radius"),
                )
                for key, edit in scale_edits.items()
            }
            if not use_fixed_cb.isChecked():
                values["r_step"] = None
            elif values["r_step"] is None:
                values["r_step"] = 2.5
            new = replace(
                settings,
                **values,
                edge_correction=edge_keys[edge_combo.currentIndex()],
                test=run_test_cb.isChecked(),
                marks=modes[marks_combo.currentIndex()],
                merge_require_all=merge_req_cb.isChecked(),
                simulations=parse_number(sim_edit, int, t, "simulations"),
                seed=parse_number(seed_edit, int, t, "seed"),
                workers=parse_number(workers_edit, int, t, "workers"),
                from_type=focal_combo.currentText(),
                to_type=neighbor_combo.currentText(),
                intensity=intensity,
                band=bands[band_combo.currentIndex()],
                pairing_mode=p_modes[pairing_combo.currentIndex()],
                perm_stat=perm_keys[perm_combo.currentIndex()],
                contrast=contrast_cb.isChecked(),
                paired=paired_cb.isChecked(),
                group_a=a_combo.currentText(),
                group_b=b_combo.currentText(),
            )
            new.validate()
            if new.contrast and (not new.group_a or not new.group_b or new.group_a == new.group_b):
                raise ValueError("choose_two_groups")
        except ValueError as error:
            show_error(dialog, t, error)
            return
        dialog.accept()
        callback(new, cur_intensity_path)

    dialog_buttons(dialog, root, t, on_apply)
    dialog.exec()


def series_selector_dialog(parent, t, results, selected_ids, callback):
    dialog = QDialog(parent)
    dialog.setWindowTitle(t("select_groups_title"))
    dialog.resize(720, 610)
    dialog.setMinimumSize(580, 460)
    root = QVBoxLayout(dialog)
    root.setContentsMargins(16, 14, 16, 14)

    hint = QLabel(t("select_groups_hint"))
    hint.setWordWrap(True)
    hint.setObjectName("MutedLabel")
    root.addWidget(hint)

    all_series = list(results.summaries) + list(results.custom_summaries)
    checks: list[QCheckBox] = []

    def series_uid(s):
        return (s.series_id, s.group, s.label)

    chosen_uids = set(selected_ids)

    preset_row = QHBoxLayout()
    preset_row.addWidget(QLabel(t("quick_presets")))
    sel_btn = QPushButton(t("select_all_sheets"))
    sel_btn.setObjectName("SmallBtn")
    sel_btn.clicked.connect(lambda: [cb.setChecked(True) for cb in checks])
    clr_btn = QPushButton(t("clear_all"))
    clr_btn.setObjectName("SmallBtn")
    clr_btn.clicked.connect(lambda: [cb.setChecked(False) for cb in checks])
    preset_row.addWidget(sel_btn)
    preset_row.addWidget(clr_btn)
    preset_row.addStretch(1)
    root.addLayout(preset_row)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    list_widget = QWidget()
    list_widget.setObjectName("CardFrame")
    grid = QGridLayout(list_widget)
    grid.setContentsMargins(12, 10, 12, 10)
    scroll.setWidget(list_widget)
    root.addWidget(scroll, stretch=1)

    def rebuild():
        while grid.count():
            it = grid.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        checks.clear()
        for idx, s in enumerate(all_series):
            cb = QCheckBox(f"{s.format_name(t('all'))}   (N = {len(s.sample_ids)})")
            cb.setChecked(series_uid(s) in chosen_uids)
            checks.append(cb)
            grid.addWidget(cb, idx // 2, idx % 2)

    rebuild()

    point_types = sorted_types(results.data.points.cell_type)
    treatments = sorted(set(results.data.points.group) - {""})
    if len(point_types) >= 2:
        merge_box = QGroupBox(t("custom_merge_section"))
        merge_layout = QVBoxLayout(merge_box)
        top_line = QHBoxLayout()
        top_line.addWidget(QLabel(t("target_treatment")))
        trt_combo = QComboBox()
        trt_items = (
            [t("each_treatment_separately"), t("all_treatments_pooled")] + treatments
            if treatments
            else [t("all_treatments_pooled")]
        )
        trt_combo.addItems(trt_items)
        top_line.addWidget(trt_combo)
        req_cb = QCheckBox(t("merge_require_all"))
        req_cb.setChecked(results.settings.merge_require_all)
        top_line.addWidget(req_cb)
        top_line.addStretch(1)
        merge_layout.addLayout(top_line)

        pt_grid = QGridLayout()
        pt_checks: dict[str, QCheckBox] = {}
        for idx, pt in enumerate(point_types):
            cb = QCheckBox(pt)
            cb.setChecked(True)
            pt_checks[pt] = cb
            pt_grid.addWidget(cb, idx // 4, idx % 4)
        merge_layout.addLayout(pt_grid)

        add_btn = QPushButton(t("add_merged_series"))
        add_btn.setObjectName("SmallBtn")

        def on_add_merged():
            chosen_pts = tuple(pt for pt, cb in pt_checks.items() if cb.isChecked())
            if len(chosen_pts) < 2:
                show_error(dialog, t, ValueError("need_two_types_merge"))
                return
            chosen_uids.clear()
            for s_item, cb in zip(all_series, checks):
                if cb.isChecked():
                    chosen_uids.add(series_uid(s_item))
            sel_idx = trt_combo.currentIndex()
            if treatments and sel_idx == 0:
                target_groups = [(g,) for g in treatments]
            elif (treatments and sel_idx == 1) or not treatments:
                target_groups = [None]
            else:
                target_groups = [(treatments[sel_idx - 2],)]
            added = 0
            for grp_filter in target_groups:
                pt_text = merged_label(chosen_pts) + ("" if req_cb.isChecked() else " (any)")
                grp_name = grp_filter[0] if grp_filter else t("all")
                custom_title = f"{grp_name} · {pt_text}" if grp_filter else pt_text
                existing = next(
                    (
                        s
                        for s in all_series
                        if s.series_id == custom_title
                        or (grp_filter and s.group == grp_filter[0] and s.label == pt_text)
                    ),
                    None,
                )
                if existing is not None:
                    chosen_uids.add(series_uid(existing))
                    added += 1
                    continue
                built = build_custom_summary(
                    results, grp_filter, chosen_pts, require_all=req_cb.isChecked(), title=custom_title
                )
                if built is not None:
                    results.custom_summaries.append(built)
                    all_series.append(built)
                    chosen_uids.add(series_uid(built))
                    added += 1
            if not added:
                show_error(dialog, t, ValueError("no_matching_samples"))
                return
            rebuild()

        add_btn.clicked.connect(on_add_merged)
        merge_layout.addWidget(add_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        root.addWidget(merge_box)

    def on_accept():
        picked = [s for s, cb in zip(all_series, checks) if cb.isChecked()]
        if not picked:
            show_error(dialog, t, ValueError("need_at_least_one_group"))
            return
        dialog.accept()
        callback(picked)

    dialog_buttons(dialog, root, t, on_accept)
    dialog.exec()


def geometry_dialog(parent, t, data, existing_custom, callback):
    dialog = QDialog(parent)
    dialog.setWindowTitle(t("draw_roi"))
    dialog.resize(800, 650)
    root = QVBoxLayout(dialog)
    root.setContentsMargins(16, 14, 16, 14)

    help_lbl = QLabel(t("draw_help"))
    help_lbl.setWordWrap(True)
    help_lbl.setObjectName("MutedLabel")
    root.addWidget(help_lbl)

    fields = list(data.points.groupby(["sample_id", "roi_id"], sort=False))
    names = [
        f"{'✓ ' if key in existing_custom else ''}{i + 1} · {key[0]} / {key[1]}"
        for i, (key, _) in enumerate(fields)
    ]
    selector = make_choice(root, t, "field", names, names[0], translated=False)

    fig = Figure(figsize=(7.2, 4.4), layout="constrained")
    ax = fig.add_subplot()
    canvas = FigureCanvasQTAgg(fig)
    toolbar = PlotToolbar(canvas, dialog, t)
    root.addWidget(toolbar)
    root.addWidget(canvas, stretch=1)

    vertices: list[tuple[float, float]] = []
    status_row = QHBoxLayout()
    v_count_lbl = QLabel(t("vertex_count", n=0))
    v_count_lbl.setObjectName("MutedLabel")
    undo_btn = QPushButton(t("undo_vertex"))
    undo_btn.setObjectName("SmallBtn")
    reset_btn = QPushButton(t("reset"))
    reset_btn.setObjectName("SmallBtn")
    status_row.addWidget(v_count_lbl)
    status_row.addStretch(1)
    status_row.addWidget(undo_btn)
    status_row.addWidget(reset_btn)
    root.addLayout(status_row)

    def update_line():
        shown = vertices + ([vertices[0]] if len(vertices) >= 3 else [])
        if shown:
            ax.lines[-1].set_data(*zip(*shown))
        else:
            ax.lines[-1].set_data([], [])
        v_count_lbl.setText(t("vertex_count", n=len(vertices)))
        canvas.draw_idle()

    def redraw():
        vertices.clear()
        ax.clear()
        key, grp = fields[selector.currentIndex()]
        xy = grp[["x", "y"]].to_numpy()
        ax.scatter(*xy[select_indices(len(xy))].T, s=9, color=PALETTE[0], alpha=.6)
        if key in existing_custom:
            existing_geom = existing_custom[key]
            parts = (existing_geom,) if existing_geom.geom_type == "Polygon" else existing_geom.geoms
            for part in parts:
                gx, gy = part.exterior.xy
                ax.plot(gx, gy, "--", color=PALETTE[1], linewidth=1.2, alpha=.7)
        ax.margins(.12)
        spatial_axes(ax, xy, None, data.calibration, t)
        ax.plot([], [], "-o", markersize=4.5, color=PALETTE[3], linewidth=1.5)
        use_cjk_font(fig, t)
        v_count_lbl.setText(t("vertex_count", n=0))
        canvas.draw_idle()

    def undo_last():
        if vertices:
            vertices.pop()
            update_line()

    def on_click(event):
        if toolbar.mode or event.inaxes is not ax or event.xdata is None or event.ydata is None:
            return
        if event.button == 1:
            vertices.append((float(event.xdata), float(event.ydata)))
            update_line()
        elif event.button == 3:
            undo_last()

    def on_accept():
        try:
            if len(vertices) < 3:
                raise ValueError("invalid_roi")
            geometry = make_window("custom", [], geometry=Polygon(vertices)).geometry
        except ValueError as error:
            show_error(dialog, t, error)
            return
        dialog.accept()
        callback(fields[selector.currentIndex()][0], geometry)

    selector.currentIndexChanged.connect(lambda _: redraw())
    undo_btn.clicked.connect(undo_last)
    reset_btn.clicked.connect(redraw)
    canvas.mpl_connect("button_press_event", on_click)
    dialog_buttons(dialog, root, t, on_accept)
    redraw()
    dialog.exec()


def rectangle_dialog(parent, t, calibration, current_rect, callback):
    dialog = QDialog(parent)
    dialog.setWindowTitle(t("configure_roi"))
    dialog.setMinimumWidth(380)
    root = QVBoxLayout(dialog)
    root.setContentsMargins(16, 14, 16, 14)
    factors = calibration.factors
    vals = (
        [v / factors[i % 2] for i, v in enumerate(current_rect)]
        if current_rect is not None
        else ["", "", "", ""]
    )
    edits = [make_entry(root, t, key, val, "bounds_help") for key, val in zip(RECT_COLUMNS, vals)]

    def on_accept():
        try:
            bounds = tuple(
                parse_number(ed, float, t, key) * factors[i % 2]
                for i, (key, ed) in enumerate(zip(RECT_COLUMNS, edits))
            )
            geometry = valid_bounds_box(bounds)
        except ValueError as error:
            show_error(dialog, t, error)
            return
        dialog.accept()
        callback(geometry)

    dialog_buttons(dialog, root, t, on_accept)
    dialog.exec()


def mask_origin_dialog(parent, t, callback):
    dialog = QDialog(parent)
    dialog.setWindowTitle(t("mask_origin"))
    dialog.setMinimumWidth(360)
    root = QVBoxLayout(dialog)
    root.setContentsMargins(16, 14, 16, 14)
    x_ed = make_entry(root, t, "x", "0", "origin_help")
    y_ed = make_entry(root, t, "y", "0", "origin_help")

    def on_accept():
        try:
            ox = parse_number(x_ed, float, t, "x")
            oy = parse_number(y_ed, float, t, "y")
            if callback((ox, oy), dialog):
                dialog.accept()
        except ValueError as error:
            show_error(dialog, t, error)

    dialog_buttons(dialog, root, t, on_accept)
    dialog.exec()


def methods_dialog(parent, t, markdown_path: Path):
    dialog = QDialog(parent)
    dialog.setWindowTitle(t("methods"))
    dialog.resize(840, 640)
    root = QVBoxLayout(dialog)
    root.setContentsMargins(16, 14, 16, 14)
    viewer = QTextBrowser()
    viewer.setOpenExternalLinks(True)
    viewer.setMarkdown(markdown_path.read_text(encoding="utf-8-sig"))
    root.addWidget(viewer, stretch=1)
    close_row = QHBoxLayout()
    close_row.addStretch(1)
    close_btn = QPushButton(t("close"))
    close_btn.clicked.connect(dialog.accept)
    close_row.addWidget(close_btn)
    root.addLayout(close_row)
    dialog.exec()
