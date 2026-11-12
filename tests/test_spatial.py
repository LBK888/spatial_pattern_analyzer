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
from spatial_data import Calibration, ReadOptions, column_labels, infer_mapping, load_custom_roi, load_dataset, read_table
from spatial_plots import group_figure, save_pdf
from spatial_workflow import Settings, compare_groups, data_tables, export_data, run_analysis, summarize


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
