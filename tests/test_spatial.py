from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
from numpy.testing import assert_allclose
from openpyxl import load_workbook
import pandas as pd
from PIL import Image
from shapely.geometry import Polygon

from i18n import Translator
from ripley_analysis import (Cancelled, IntensityGrid, boundary_points, global_envelope,
                             inside, local_neighbors, make_window, ripley, uniform)
from spatial_analysis_gui import demo_data
from spatial_data import (Calibration, ReadOptions, column_labels, default_excel_sheets,
                          detect_block_width, infer_mapping, load_custom_roi, load_dataset, read_table)
from spatial_plots import contrast_figure, group_figure, omnibus_figure, save_pdf
from spatial_workflow import (Settings, build_custom_summary, compare_groups, compare_series,
                              data_tables, default_fixed_grid, export_data, run_analysis, summarize, with_band)


class RipleyTests(unittest.TestCase):
    def setUp(self):
        self.rect = make_window("rectangle", [], (0, 0, 10, 10))

    def test_ordered_pairs_and_exact_translation_weight(self):
        points = [[3, 4], [4, 4], [8, 8]]
        result = ripley(points, self.rect, [.9, 1.0, 2.0])
        assert_allclose(result, [0, 2 * 100 * (100 / 90) / (3 * 2), 2 * 100 * (100 / 90) / (3 * 2)])

    def test_coincident_distinct_points_and_cross_k(self):
        assert_allclose(ripley([[4, 4], [4, 4]], self.rect, [1]), [100])
        assert_allclose(ripley([[4, 4]], self.rect, [1], targets=[[4, 4], [8, 8]]), [50])

    def test_border_correction_counts_only_eligible_focal_points(self):
        geometry = Polygon([(0, 0), (0, 10), (10, 10), (10, 0)],
                           [[(4, 4), (4, 6), (6, 6), (6, 4)]])
        window = make_window("custom", [], geometry=geometry)
        points = [[1.5, 2], [.6, 2], [3.4, 5]]
        assert_allclose(ripley(points, window, [1, 2]), [48, np.nan], equal_nan=True)
        xy = uniform(window, 400, np.random.default_rng(3))
        self.assertTrue(inside(window, xy).all())
        self.assertFalse(inside(window, [[5, 5]])[0])

    def test_csr_average_matches_theory(self):
        rng = np.random.default_rng(6)
        r = np.array([.5, 1.3, 2])
        curves = [ripley(uniform(self.rect, 50, rng), self.rect, r) for _ in range(150)]
        assert_allclose(np.mean(curves, axis=0), np.pi * r**2, rtol=.08)

    def test_inhomogeneous_weights_and_simulation(self):
        rng = np.random.default_rng(23)
        equal = IntensityGrid(np.ones((4, 5)))
        variable = IntensityGrid(np.array([[1, 4], [2, 8]]))
        xy = equal.sample(70, self.rect, rng)
        r = np.array([.8, 1.7])
        assert_allclose(ripley(xy, self.rect, r, intensity=equal), ripley(xy, self.rect, r))
        curves = [ripley(variable.sample(65, self.rect, rng), self.rect, r, intensity=variable)
                  for _ in range(150)]
        assert_allclose(np.mean(curves, axis=0), np.pi * r**2, rtol=.13)

    def test_local_expected_neighbors_with_one_radius(self):
        xy = np.array([[4, 4], [4.5, 4]])
        assert_allclose(local_neighbors(xy, self.rect, 1), [1 - np.pi / 100] * 2, atol=.00003)

    def test_global_rank_ties_and_both_sides(self):
        tied = global_envelope(np.ones((20, 3)))
        self.assertEqual(tied.p, 1)
        self.assertEqual(tied.status, "compatible")
        curves = np.r_[[[10, -10]], np.zeros((19, 2))]
        mixed = global_envelope(curves)
        self.assertEqual(mixed.p, .05)
        self.assertEqual(mixed.status, "mixed")
        assert_allclose(mixed.side, [1, -1])

    def test_global_rank_excludes_unsupported_radii(self):
        result = global_envelope(np.array([[5, 9], [1, np.nan], [0, 0]]))
        self.assertTrue(np.isnan(result.upper[1]))
        self.assertTrue(np.isfinite(result.p))


class InputTests(unittest.TestCase):
    def test_headerless_first_record_and_coordinate_calibration(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "points.csv"
            path.write_text("1,2\n2,3\n4,5\n", encoding="utf-8")
            options = ReadOptions(header=False)
            raw = read_table(path, options)
            data = load_dataset([path], options, infer_mapping(column_labels(raw, False), raw, False),
                                Calibration(.5, 2, "µm"))
            assert_allclose(data.points[["x", "y"]], [[.5, 4], [1, 6], [2, 10]])
            self.assertEqual(data.points.row.tolist(), [1, 2, 3])

    def test_types_duplicates_and_nonfinite_coordinates(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "table.csv"
            path.write_text("x;y;sample_id;roi_id;cell_type\n0,5;1,5;01;R;A\n0,5;1,5;01;R;B\n0,5;1,5;01;R;B\n"
                            "0,5;1,5;01;S;B\nInf;1,0;01;S;A\n1,0;2,0;01;S;\n", encoding="utf-8")
            options = ReadOptions(delimiter=";", decimal=",")
            raw = read_table(path, options)
            data = load_dataset([path], options, infer_mapping(column_labels(raw, True)), Calibration(), "drop")
            self.assertEqual(data.points.cell_type.tolist(), ["A", "B", "B", ""])
            self.assertEqual(set(data.points.sample_id), {"01"})
            self.assertEqual(data.issues.reason.value_counts()["invalid_coordinates"], 1)
            self.assertEqual(data.issues.reason.value_counts()["duplicate_removed"], 1)

    def test_excel_second_worksheet_and_text_identifiers(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "table.xlsx"
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                pd.DataFrame({"Wrong": ["ignore"]}).to_excel(writer, sheet_name="Notes", index=False)
                pd.DataFrame({"x": [1, 2], "y": [3, 5], "sample_id": ["001", "002"]}).to_excel(
                    writer, sheet_name="Points", index=False)
            options = ReadOptions(sheet="Points")
            raw = read_table(path, options)
            result = load_dataset([path], options, infer_mapping(column_labels(raw, True)), Calibration())
            self.assertEqual(result.points.sample_id.tolist(), ["001", "002"])

    def test_mask_with_hole_and_calibration(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "mask.png"
            Image.fromarray(np.array([[255, 255, 255], [255, 0, 255], [255, 255, 255]], dtype=np.uint8)).save(path)
            w = make_window("custom", [], geometry=load_custom_roi(path, Calibration(2, 3), (10, 20))[None])
            self.assertEqual(w.area, 48)
            self.assertEqual(w.geometry.bounds, (20, 60, 26, 69))
            self.assertFalse(inside(w, [[23, 64.5]])[0])


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = demo_data()
        cls.settings = Settings(roi_mode="rectangle", simulations=39, n_radii=20)
        cls.results = run_analysis(cls.data, cls.settings)

    def test_demo_has_nine_independent_samples(self):
        self.assertEqual(len(self.data.points), 1296)
        self.assertEqual(len(self.results.patterns), 18)
        for s in self.results.summaries:
            self.assertEqual(len(s.sample_ids), 3)
            self.assertTrue((s.n == 3).all())
            assert_allclose(s.high - s.mean, s.curves.std(axis=0, ddof=1))

    def test_demo_csr_cluster_and_regular_known_directions(self):
        self.assertTrue(any(p.status == "cluster" for p in self.results.patterns if p.group == "Clustered"))
        self.assertTrue(any(p.status == "regular" for p in self.results.patterns if p.group == "Regular"))

    def test_locked_keeps_original_boundary_during_all_simulations(self):
        data = replace(self.data, points=self.data.points.iloc[:72],
                       fields=(self.data.fields[0],), bounds={})
        from spatial_workflow import ripley as estimator
        records = []
        def record(points, *args, **kwargs):
            records.append(points.copy())
            return estimator(points, *args, **kwargs)
        with patch("spatial_workflow.ripley", side_effect=record):
            result = run_analysis(data, replace(self.settings, roi_mode="locked", simulations=19))
        pattern = result.patterns[0]
        fixed = pattern.xy[boundary_points(pattern.window, pattern.xy)]
        self.assertGreater(len(fixed), 0)
        self.assertEqual(len(records), 20)
        for xy in records[1:]:
            assert_allclose(xy[:len(fixed)], fixed)

    def test_inferred_mode_is_descriptive_and_cross_labels_are_tested(self):
        data = replace(self.data, points=self.data.points.iloc[:72], fields=(self.data.fields[0],))
        settings = replace(self.settings, roi_mode="infer")
        described = run_analysis(data, settings).patterns[0]
        self.assertIsNone(described.envelope)
        cross = run_analysis(data, replace(settings, marks="cross", from_type="A", to_type="B")).patterns[0]
        self.assertIsNotNone(cross.envelope)
        self.assertEqual((cross.count, cross.target_count), (36, 36))
        self.assertEqual(cross.null, "random_label")

    def test_one_sample_sd_is_unavailable_and_ci_uses_whole_samples(self):
        first = self.results.patterns[:2]
        one = summarize(first, self.settings, np.random.default_rng(1), None)[0]
        self.assertTrue(np.isnan(one.low).all())
        ci = summarize(self.results.patterns, replace(self.settings, band="ci"), np.random.default_rng(1), None)
        self.assertTrue(all(np.isfinite(s.high).all() for s in ci))
        self.assertTrue(all(np.all(s.high >= s.low) for s in ci))

    def test_contrast_uses_independent_groups_and_paired_order(self):
        a, b = self.results.summaries[:2]
        settings = replace(self.settings, group_a=a.group, group_b=b.group)
        result = compare_groups(self.data, [a, b], settings, np.random.default_rng(5), None)[0]
        self.assertEqual((result.n_a, result.n_b), (3, 3))
        assert_allclose(result.effect, a.mean - b.mean)
        frame = self.data.points.copy()
        pairs = dict(zip(a.sample_ids + b.sample_ids[::-1], ["1", "2", "3"] * 2))
        frame["pair_id"] = frame.sample_id.map(pairs).fillna("")
        paired = compare_groups(replace(self.data, points=frame), [a, b],
                                replace(settings, paired=True), np.random.default_rng(5), None)[0]
        assert_allclose(paired.effect, result.effect)

    def test_cancel_before_work_starts(self):
        stop = Event()
        stop.set()
        with self.assertRaises(Cancelled):
            run_analysis(self.data, self.settings, stop=stop)

    def test_export_has_raw_values_metadata_and_pdf(self):
        table = data_tables(self.results)
        self.assertEqual(set(table), {"curves", "summary", "metadata", "qc", "roi", "local"})
        roi = table["curves"].query("row_type == 'roi'")
        assert_allclose(roi.L - roi.radius, roi.L_minus_r)
        self.assertEqual(set(table["curves"].query("row_type == 'group'").N), {3})
        with TemporaryDirectory() as d:
            d = Path(d)
            export_data(self.results, d / "results.xlsx")
            workbook = load_workbook(d / "results.xlsx", read_only=True)
            self.assertIn("metadata", workbook.sheetnames)
            workbook.close()
            export_data(self.results, d / "results.zip")
            with ZipFile(d / "results.zip") as archive:
                self.assertTrue(archive.read("summary.csv").startswith(b"\xef\xbb\xbf"))
            translator = Translator()
            save_pdf(group_figure(self.results, "", translator), d / "figure.pdf", "Plot")
            pdf = (d / "figure.pdf").read_bytes()
            self.assertTrue(pdf.startswith(b"%PDF"))
            self.assertIn(b"/FontFile2", pdf)
            self.assertIn(b"/ToUnicode", pdf)


class MultiSheetFixedRadiusAndBiostatsTests(unittest.TestCase):
    def test_multisheet_excel_a_to_i_and_side_by_side_blocks(self):
        with TemporaryDirectory() as d:
            d = Path(d)
            # 1. Stacked A-I format across 3 treatment sheets, with repeated headers & blank padding rows
            stacked_path = d / "stacked_ai.xlsx"
            with pd.ExcelWriter(stacked_path, engine="openpyxl") as writer:
                for trt in ("Control", "Colchicine", "Cytochalasin D"):
                    rows = [
                        ["HRC(HA)", f" {trt} 1", "Area", "Mean", "StdDev", "Min", "Max", "X", "Y"],
                        ["HRC(HA)", f" {trt} 1", 0, 70, 0, 70, 70, 500, 600],
                        ["HRC(HA)", f" {trt} 1", 0, 72, 0, 72, 72, 540, 620],
                        ["HRC(HA)", f" {trt} 1", "", "", "", "", "", "", ""],  # padding row
                        ["NaRC(α5 )", f"{trt} 1", "Area", "Mean", "StdDev", "Min", "Max", "X", "Y"],  # repeated header
                        ["NaRC(α5 )", f"{trt} 1", 0, 65, 0, 65, 65, 510, 610],
                        ["NaRC(α5 )", f"{trt} 1", 0, 66, 0, 66, 66, 560, 640],
                        ["NCC(ISH)", f"{trt} 1", "Area", "Mean", "StdDev", "Min", "Max", "X", "Y"],
                        ["NCC(ISH)", f"{trt} 1", 0, 80, 0, 80, 80, 520, 615],
                        ["NCC(ISH)", f"{trt} 1", 0, 81, 0, 81, 81, 580, 650],
                    ]
                    pd.DataFrame(rows).to_excel(writer, sheet_name=trt, index=False, header=False)

            sheets = default_excel_sheets(stacked_path)
            self.assertEqual(sheets, ("Control", "Colchicine", "Cytochalasin D"))
            opts = ReadOptions(sheet=sheets[0], sheets=sheets, sheet_as_group=True)
            raw = read_table(stacked_path, opts, limit=20)
            mapping = infer_mapping(column_labels(raw, True), raw, True)
            self.assertEqual(mapping["x"], 7)
            self.assertEqual(mapping["y"], 8)
            self.assertEqual(mapping["cell_type"], 0)
            self.assertEqual(mapping["sample_id"], 1)
            ds = load_dataset([stacked_path], opts, mapping, Calibration())
            self.assertEqual(set(ds.points.group), {"Control", "Colchicine", "Cytochalasin D"})
            self.assertEqual(set(ds.points.cell_type), {"HRC(HA)", "NaRC(α5 )", "NCC(ISH)"})
            self.assertEqual(ds.points.sample_id.nunique(), 3)
            self.assertEqual(len(ds.points), 18)
            self.assertEqual(len(ds.issues), 0)

            # 2. Wide side-by-side 9-column blocks (unstack_blocks=True)
            wide_path = d / "wide_blocks.xlsx"
            with pd.ExcelWriter(wide_path, engine="openpyxl") as writer:
                wide_rows = [
                    ["HRC", "S1", "Area", "Mean", "StdDev", "Min", "Max", "X", "Y",
                     "HRC", "S2", "Area", "Mean", "StdDev", "Min", "Max", "X", "Y"],
                    ["HRC", "S1", 0, 50, 0, 50, 50, 100, 200,
                     "HRC", "S2", "", "", "", "", "", "", ""],
                    ["HRC", "S1", 0, 55, 0, 55, 55, 130, 240,
                     "HRC", "S2", 0, 60, 0, 60, 60, 300, 400],
                ]
                pd.DataFrame(wide_rows).to_excel(writer, sheet_name="Control", index=False, header=False)
            raw_probe = pd.read_excel(wide_path, header=None, dtype=str, keep_default_na=False)
            self.assertEqual(detect_block_width(raw_probe, True), 9)
            wide_opts = ReadOptions(sheet="Control", sheets=("Control",), sheet_as_group=True, unstack_blocks=True)
            unstacked = read_table(wide_path, wide_opts)
            w_map = infer_mapping(column_labels(unstacked, True), unstacked, True)
            w_ds = load_dataset([wide_path], wide_opts, w_map, Calibration())
            self.assertEqual(sorted(w_ds.points.sample_id.unique()), ["S1", "S2"])
            self.assertEqual(len(w_ds.points), 3)

    def test_same_sample_number_across_sheets_is_disambiguated(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "numeric_ids.xlsx"
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                for trt in ("Control", "Drug"):
                    pd.DataFrame({
                        "點名稱": ["HRC", "HRC"],
                        "樣品編號": ["1", "1"],
                        "X": [10, 20],
                        "Y": [30, 40],
                    }).to_excel(writer, sheet_name=trt, index=False)
            opts = ReadOptions(sheets=("Control", "Drug"), sheet_as_group=True)
            raw = read_table(path, opts)
            mapping = infer_mapping(column_labels(raw, True), raw, True)
            ds = load_dataset([path], opts, mapping, Calibration())
            self.assertEqual(set(ds.points.sample_id), {"Control 1", "Drug 1"})

    def test_fixed_r_step_0_to_250_by_2_5_gives_100_points_and_handles_mixed_sizes(self):
        rng = np.random.default_rng(42)
        rows = []
        # Sample 1: large window (supports up to r=250); Sample 2: smaller window (drops out before 250)
        for sid, span in (("C1", 1200.0), ("C2", 500.0), ("C3", 1100.0)):
            xy = rng.uniform(0, span, size=(80, 2))
            for x, y in xy:
                rows.append(dict(source="mem", row=1, sample_id=sid, roi_id="1",
                                 cell_type="HRC", group="Control", pair_id="", block_id="", x=x, y=y))
        from spatial_data import Dataset
        ds = Dataset(pd.DataFrame(rows), pd.DataFrame(), ("mem",), Calibration(),
                     tuple(("mem", s, "1") for s in ("C1", "C2", "C3")))
        r0, rmax, rstep = default_fixed_grid(ds)
        self.assertEqual((r0, rmax, rstep), (0.0, 250.0, 2.5))
        settings = Settings(roi_mode="infer", r_min=0.0, r_max=250.0, r_step=2.5,
                            test=False, marks="each_and_merged", band="se")
        res = run_analysis(ds, settings)
        self.assertEqual(len(res.radii), 100)
        assert_allclose(res.radii[0], 2.5)
        assert_allclose(res.radii[-1], 250.0)
        assert_allclose(np.diff(res.radii), 2.5)
        s = res.summaries[0]
        # SE band equals SD / sqrt(N)
        valid = s.n > 1
        assert_allclose(s.high[valid] - s.mean[valid], s.sd[valid] / np.sqrt(s.n[valid]))
        # CI band also works even when smaller sample drops out at large r
        s_ci = with_band(s, "ci", seed=42)
        self.assertTrue(np.isfinite(s_ci.high[0]))

    def test_flexible_grouping_merged_points_and_biostatistics(self):
        rng = np.random.default_rng(99)
        rows = []
        # Create 2 treatments x 4 samples with 3 point types (HRC clustered, NaRC & NCC uniform)
        from convex_hull_pattern_generator import clustered
        win = make_window("rectangle", [], (0, 0, 600, 600))
        for trt, is_clust in (("Control", True), ("Colchicine", False)):
            for i in range(1, 5):
                sid = f"{trt} {i}"
                pts_hrc = clustered(win, 60, rng) if is_clust else uniform(win, 60, rng)
                pts_narc = uniform(win, 60, rng)
                # Leave NCC out of sample 4 in Control to exercise partial overlap & merge_require_all!
                has_ncc = not (trt == "Control" and i == 4)
                pts_ncc = uniform(win, 60, rng) if has_ncc else np.empty((0, 2))
                for ptype, arr in (("HRC", pts_hrc), ("NaRC", pts_narc), ("NCC", pts_ncc)):
                    for x, y in arr:
                        rows.append(dict(source="mem", row=1, sample_id=sid, roi_id="1",
                                         cell_type=ptype, group=trt, pair_id="", block_id="",
                                         x=float(x), y=float(y), xmin=0.0, ymin=0.0, xmax=600.0, ymax=600.0))
        from spatial_data import Dataset
        df = pd.DataFrame(rows)
        bounds = {(sid, "1"): (0.0, 0.0, 600.0, 600.0) for sid in df.sample_id.unique()}
        ds = Dataset(df, pd.DataFrame(), ("mem",), Calibration(),
                     tuple(("mem", sid, "1") for sid in df.sample_id.unique()), bounds=bounds)
        settings = Settings(roi_mode="rectangle", r_min=0.0, r_max=100.0, r_step=5.0,
                            simulations=99, test=False, marks="each_and_merged")
        res = run_analysis(ds, settings)
        self.assertEqual(len(res.radii), 20)
        by_key = {(s.group, s.label): s for s in res.summaries}
        # Control merged (HRC + NaRC + NCC) excludes Control 4 when merge_require_all=True -> N=3!
        self.assertEqual(len(by_key[("Control", "HRC + NaRC + NCC")].sample_ids), 3)
        self.assertEqual(len(by_key[("Control", "HRC")].sample_ids), 4)

        # 1. Compare Control HRC vs Colchicine HRC (independent groups, clustered vs uniform -> significant!)
        comp_ind = compare_series(
            [by_key[("Control", "HRC")], by_key[("Colchicine", "HRC")]],
            res.radii, data=ds, simulations=199, seed=11
        )
        self.assertEqual(comp_ind.omnibus.design, "independent")
        self.assertLess(comp_ind.omnibus.p_global, 0.05)
        self.assertEqual(comp_ind.contrasts[0].parametric_test, "welch_t")

        # 2. Compare within Control: HRC (N=4), NaRC (N=4), NCC (N=3), HRC+NaRC+NCC (N=3) -> partial_overlap!
        ctrl_series = [by_key[("Control", k)] for k in ("HRC", "NaRC", "NCC", "HRC + NaRC + NCC")]
        comp_within = compare_series(ctrl_series, res.radii, data=ds, simulations=199, seed=11)
        self.assertEqual(comp_within.omnibus.design, "partial_overlap")
        self.assertEqual(comp_within.omnibus.k_groups, 4)
        self.assertEqual(len(comp_within.contrasts), 6)
        self.assertLess(comp_within.omnibus.p_global, 0.05)

        # 3. Compare fully paired subset (Colchicine: HRC, NaRC, NCC, Merged all have N=4 identical sample_ids)
        col_series = [by_key[("Colchicine", k)] for k in ("HRC", "NaRC", "NCC")]
        comp_paired = compare_series(col_series, res.radii, data=ds, simulations=199, seed=11)
        self.assertEqual(comp_paired.omnibus.design, "paired")
        self.assertEqual(comp_paired.omnibus.parametric_test, "rm_anova")
        self.assertEqual(comp_paired.omnibus.nonparametric_test, "friedman")

        # 4. Custom 2-type merge (HRC + NaRC) & plotting
        custom = build_custom_summary(res, ("Control",), ("HRC", "NaRC"), require_all=True, title="Control · HRC+NaRC")
        self.assertEqual(len(custom.sample_ids), 4)
        t_zh = Translator("zh-TW")
        fig1 = group_figure(res, ctrl_series, t_zh, band="se", omnibus=comp_within.omnibus)
        fig2 = omnibus_figure(res, comp_within.omnibus, t_zh)
        fig3 = contrast_figure(res, comp_within.contrasts[0], t_zh)
        self.assertIsNotNone(fig1)
        self.assertIsNotNone(fig2)
        self.assertIsNotNone(fig3)


class PyQt6UITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_pyqt_main_window_language_switch_preserves_draft_and_renders_tabs(self):
        from spatial_analysis_gui import SpatialAnalysisGUI
        from spatial_plots import save_figure

        win = SpatialAnalysisGUI("en")
        win.data = demo_data()
        win._configure_defaults_for_dataset()
        win.rmax_edit.setText("45")
        win.rstep_edit.setText("0.5")
        # Switch to Traditional Chinese and verify draft values are preserved
        win.lang_combo.setCurrentIndex(1)
        self.assertEqual(win.t.lang, "zh-TW")
        self.assertEqual(win.rmax_edit.text(), "45")
        self.assertEqual(win.rstep_edit.text(), "0.5")

        win.results = run_analysis(
            win.data,
            Settings(roi_mode="rectangle", r_min=0.0, r_max=40.0, r_step=2.0, simulations=19, marks="each_and_merged"),
        )
        presets = win._build_presets()
        self.assertGreaterEqual(len(presets), 2)
        win._update_active_comparison(presets[0][1])
        win.render_main()

        for idx, key in enumerate(("view_groups", "view_roi", "view_contrast", "view_qc")):
            win.tabs.setCurrentIndex(idx)
            self.assertEqual(win.view, key)

        with TemporaryDirectory() as d:
            d = Path(d)
            win.tabs.setCurrentIndex(0)
            save_figure(win.figure, d / "out.svg", "SVG Test")
            save_figure(win.figure, d / "out.png", "PNG Test")
            self.assertTrue((d / "out.svg").exists())
            self.assertTrue((d / "out.png").exists())
        win.close()

    def test_pyqt_import_dialog_multisheet_and_smoke_test(self):
        from spatial_analysis_gui import smoke_test
        from spatial_dialogs import ImportDialog

        with TemporaryDirectory() as d:
            d = Path(d)
            smoke_test(d / "smoke")
            self.assertTrue((d / "smoke" / "groups-en.pdf").exists())
            self.assertTrue((d / "smoke" / "groups-zh-TW.pdf").exists())
            self.assertTrue((d / "smoke" / "results.xlsx").exists())

            xlsx = d / "multi.xlsx"
            with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
                for sh in ("Control", "Colchicine"):
                    pd.DataFrame({
                        "點名稱": ["HRC", "NaRC"],
                        "樣品編號": [f"{sh} 1", f"{sh} 1"],
                        "X": [100, 150],
                        "Y": [200, 250],
                    }).to_excel(writer, sheet_name=sh, index=False)
            captured = []
            dlg = ImportDialog(None, Translator("zh-TW"), [xlsx], lambda *args: captured.append(args))
            self.assertEqual(set(dlg.sheet_checks.keys()), {"Control", "Colchicine"})
            self.assertTrue(dlg.sheet_group_cb.isChecked())
            dlg.unit_edit.setText("µm")
            dlg.x_scale_edit.setText("0.5")
            dlg.y_scale_edit.setText("0.5")
            dlg._on_accept()
            self.assertEqual(len(captured), 1)
            paths, options, mapping, cal, dup = captured[0]
            self.assertEqual(options.sheets, ("Control", "Colchicine"))
            self.assertEqual(cal.unit, "µm")
            self.assertEqual(cal.x, 0.5)
            dlg.close()

    def test_recommendations_a_to_d_edge_seed_welch_cross_and_batch_export(self):
        from spatial_workflow import export_batch_bundle

        # A: Elongated non-rectangular polygon (1400 x 380, like zebrafish yolk sac) at r=250
        elongated = Polygon([(0, 40), (700, 0), (1400, 40), (1400, 340), (700, 380), (0, 340)])
        rng = np.random.default_rng(77)
        w_border = make_window("custom", [], geometry=elongated, edge_mode="border")
        w_trans = make_window("custom", [], geometry=elongated, edge_mode="translation")
        w_iso = make_window("custom", [], geometry=elongated, edge_mode="isotropic")
        pts = uniform(w_trans, 160, rng)
        r_grid = np.array([50.0, 150.0, 250.0])
        k_border = ripley(pts, w_border, r_grid)
        k_trans = ripley(pts, w_trans, r_grid)
        k_iso = ripley(pts, w_iso, r_grid)
        self.assertTrue(np.isnan(k_border[-1]))       # Border loses all focal points at r=250 > 190
        self.assertTrue(np.isfinite(k_trans[-1]))     # Polygon translation retains full support at r=250
        self.assertTrue(np.isfinite(k_iso[-1]))       # Ripley isotropic retains full support at r=250

        # Verify CSR unbiasedness of polygon translation & isotropic on non-rectangular polygon
        r_csr = np.array([30.0, 80.0, 220.0])
        trans_sims = [ripley(uniform(w_trans, 90, rng), w_trans, r_csr) for _ in range(80)]
        iso_sims = [ripley(uniform(w_iso, 90, rng), w_iso, r_csr) for _ in range(80)]
        assert_allclose(np.nanmean(trans_sims, axis=0), np.pi * r_csr ** 2, rtol=0.12)
        assert_allclose(np.nanmean(iso_sims, axis=0), np.pi * r_csr ** 2, rtol=0.12)

        # B: SeedSequence order & thread-count independence
        demo = demo_data()
        sub_two = replace(
            demo,
            points=demo.points[demo.points.sample_id.isin(["U1", "C1"])].reset_index(drop=True),
            fields=tuple(f for f in demo.fields if f[1] in ("U1", "C1")),
        )
        sub_one = replace(
            demo,
            points=demo.points[demo.points.sample_id == "C1"].reset_index(drop=True),
            fields=tuple(f for f in demo.fields if f[1] == "C1"),
        )
        st_seq = Settings(roi_mode="rectangle", simulations=39, n_radii=15, seed=999, workers=1)
        st_par = replace(st_seq, workers=4)
        res_seq = run_analysis(sub_two, st_seq)
        res_par = run_analysis(sub_two, st_par)
        res_solo = run_analysis(sub_one, st_seq)
        p_seq = next(p for p in res_seq.patterns if p.sample == "C1" and p.roi == "F1")
        p_par = next(p for p in res_par.patterns if p.sample == "C1" and p.roi == "F1")
        p_solo = next(p for p in res_solo.patterns if p.sample == "C1" and p.roi == "F1")
        assert_allclose(p_seq.envelope.lower, p_par.envelope.lower)
        assert_allclose(p_seq.envelope.lower, p_solo.envelope.lower)
        self.assertEqual(p_seq.envelope.p, p_solo.envelope.p)

        # C & D: Convex guard, Welch studentized permutation, Cross-all matrix, and Batch export
        st_cross_all = Settings(
            roi_mode="infer", edge_correction="translation", hull_ratio=1.0,
            r_min=0.0, r_max=40.0, r_step=5.0, simulations=29, marks="cross_all", perm_stat="welch",
        )
        res_cross = run_analysis(demo, st_cross_all)
        cross_labels = {s.label for s in res_cross.summaries}
        self.assertEqual(cross_labels, {"A → B", "B → A"})

        with TemporaryDirectory() as d:
            written = export_batch_bundle(res_cross, Path(d) / "bundle", Translator("zh-TW"), formats=("pdf", "png"))
            self.assertGreaterEqual(len(written), 5)
            self.assertTrue((Path(d) / "bundle" / "spatial-results.xlsx").exists())



