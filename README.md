# Spatial Pattern Analyzer

[繁體中文說明](README.zh-TW.md) · [Windows downloads](https://github.com/LBK888/spatial_pattern_analyzer/releases) · [BibTeX](CITATION.bib)

Analyze two-dimensional point locations in an observation window with Ripley's K and L. The desktop interface and PDF figures are available in English and Traditional Chinese.

## Get started

1. Download the Windows x64 EXE from Releases and open it. For a source installation, use Python 3.10 or later with Tk, run setup.bat, then start.bat.
2. Click **Try demo**, then **Run analysis**. The bundled synthetic data contain 9 independent samples, 18 ROIs, and 1,296 points across uniform, clustered and regular groups. Each group has N=3. The point types A and B are randomly assigned within ROIs. Actual rectangular bounds are supplied in the table.
3. Open **Groups** to compare curves and between-sample variation. Select a view in **ROI detail** to inspect its point map and null envelope. **Quality control** lists input and analysis exclusions. Click ⓘ next to a setting for help.

## Data and observation windows

CSV, TSV, TXT, XLSX and XLS are supported. Map X/Y and, when available, sample_id, roi_id, cell_type and group. Pair and block IDs, plus the rectangle columns xmin, ymin, xmax and ymax, are under More columns. Batch files share the mapped layout and worksheet. If sample ID is omitted, the filename stem identifies an independent sample. If ROI ID is omitted, the file contributes one ROI per sample. Use the original coordinate system for bounds and ROI files. The separate X and Y calibration factors apply to both points and windows. Supply a physical minimum radius if localization or segmentation makes shorter distances unreliable. Image Y orientation controls map display.

The four ROI modes answer different questions:

- **Infer from points** uses a concave hull. The hull looseness is configurable. The location of the hull depends on the observations, so the point-location curve is descriptive. Fixed-location random-label tests remain available.
- **Infer and fix boundary points** uses a convex hull. In each conditional CSR simulation all observed hull-boundary positions stay fixed, and only interior positions are redrawn uniformly. This conditional model assumes continuous uniform sampling over a convex support and addresses the interior arrangement conditional on the observed hull. Choose a known ROI to test the whole observed window, especially with concave sampling regions or image masks. Very few interior points, coincident points and per-type curves in this mode have descriptive results.
- **Rectangle** uses the four table columns for each ROI when present and valid, or manually entered bounds as a common fallback. Use actual acquisition or crop bounds.
- **Custom ROI** accepts planar WKT, GeoJSON polygons or multipolygons (including holes), binary PNG/TIFF masks, and polygons drawn in the app. A GeoJSON FeatureCollection can identify a window with sample_id and roi_id properties. A feature with neither property, a single WKT, or an image mask is a common window. Nonzero mask pixels are inside. Enter the origin of the upper-left pixel edge in the original coordinates. Drawn polygons are associated with the selected ROI.

The importer reports nonfinite coordinates, missing identifiers, coincident points, and points outside the ROI. Coincident points are kept by default. An optional policy removes repeated positions of the same type within each sample and ROI. An exactly coincident pattern can be described and can be tested by random labeling; continuous-position CSR is withheld. Record numbers identify logical table records.

## Analysis and interpretation

- **K and L.** Rectangles use translation correction and ordered distinct pairs with n(n−1) normalization. Other shapes use border correction with focal points at least r from the boundary, including hole boundaries. Unsupported radii are missing. Default maximum distance is one quarter of the shorter bounding-box dimension, capped further for irregular windows based on interior clearance. Runs use a common physical radius grid. K has squared distance units, L = sqrt(K/π), and plots display L(r) − r.
- **Global pattern test.** Fixed-count CSR places the observed number of points independently and uniformly within the known ROI. Each observed and simulated curve uses the same estimator. The two-sided 95% global extreme rank length envelope controls the selected radius family, with ties handled conservatively. The default is 999 simulations with a reproducible seed. The raw global p and Holm-adjusted p are shown separately. Use adjusted p when interpreting the family of tests from a run. Excursions indicate scales with more or fewer close pairs under the chosen model. Failure to reject expresses the available evidence. Report the radius range and chosen null along with the result.
- **Inhomogeneity and labels.** For known rectangular windows, a positive numeric CSV matrix can specify an independently measured relative intensity on equal-sized cells. Rows run from low to high Y. The grid applies to each rectangle in normalized coordinates. Simulation uses the fixed grid, and pair contributions are intensity reweighted. The model assumes this external intensity is fixed. For cross-type K, coordinates are held fixed and the chosen two labels are permuted within each ROI with their counts preserved. Interpret its curve against the random-label envelope, particularly when the pooled locations have a spatial gradient.
- **Replicates.** Analyze ROIs first. Within a sample average the available ROI curves at each radius, then give each independent sample equal weight within its group. Plot individual sample curves, group means and effective N(r). The default band is ± one sample SD; the optional 95% percentile bootstrap CI is pointwise, resampling whole sample curves at radii supported by all samples and with N≥3. The null envelope serves the separate purpose of model testing. With one sample, there is no sample variability band.
- **Group contrast.** The curve is mean A minus mean B at jointly supported radii. The permutation test shuffles independent sample labels within block, or swaps conditions within matched pairs, according to the selected study design. This implementation uses the raw mean difference curve with global rank calibration. It requires exchangeable labels within blocks or pairs under the null. Use the exported Holm p when interpreting several contrasts by type.
- **Local map.** The optional radius shows each point's neighbor count minus its uniform expectation in the intersection of its search disk and the ROI. It is a descriptive map, with a color scale centered at zero. The disk area is approximated by a 96-sided polygon. This map applies to univariate curves under a uniform reference.

## Export and reproduce

**Export PDF** saves the current figure with vector lines and embedded searchable fonts. It is designed for editing plot text in vector/PDF editors. Windows uses installed Traditional Chinese fonts. **Export data** saves an Excel workbook or a ZIP of UTF-8 CSV tables. curves contains ROI, sample, group and contrast curves. summary holds tests and excursions. metadata captures settings, import mapping, calibration, versions and seeds. roi includes window WKT, qc records excluded or flagged observations, and local contains optional neighbor deviations. The exported seed, data order, settings and software versions allow Monte Carlo runs to be reproduced.

For a paper, report the observation window and calibration, edge correction, independent sample and ROI counts, analysis radii, chosen null model, simulation count, and multiple-testing correction.

## Methods and citations

| Reference | Where it is used |
|---|---|
| [Ripley (1976), The second-order analysis of stationary point processes](https://doi.org/10.2307/3212829) and [Besag (1977), discussion of Ripley's Modelling spatial patterns](https://doi.org/10.1111/j.2517-6161.1977.tb01616.x) | K, border correction and the square-root L transformation. Plots center L by subtracting r. |
| [Ohser (1983), On estimators for the reduced second moment measure of point processes](https://doi.org/10.1080/02331888308801687) | Translation edge weights for rectangles. |
| [Myllymäki et al. (2017), Global envelope tests for spatial processes](https://doi.org/10.1111/rssb.12172) and [Myllymäki & Mrkvička (2024), GET: Global Envelopes in R](https://doi.org/10.18637/jss.v111.i03) | Global Monte Carlo and permutation testing with extreme rank length. The app implements this method in Python. |
| [Holm (1979), A simple sequentially rejective multiple test procedure](https://www.jstor.org/stable/4615733) | Adjusts ROI p values and group-contrast p values in separate families. |
| [Baddeley, Møller & Waagepetersen (2000), Non- and semi-parametric estimation of interaction in inhomogeneous point patterns](https://doi.org/10.1111/1467-9574.00144) | Inverse-intensity pair weighting. The app uses an external grid and fixed-count normalization. |
| [Diggle, Mateu & Clough (2000), A comparison between parametric and non-parametric approaches to the analysis of replicated spatial point patterns](https://doi.org/10.1239/aap/1013540166) and [Efron (1979), Bootstrap methods: Another look at the jackknife](https://doi.org/10.1214/aos/1176344552) | Replicate-level functional summaries and resampling whole independent sample curves. |
| [spatstat Kcross and rlabel documentation](https://search.r-project.org/CRAN/refmans/spatstat.explore/html/Kcross.html) · [rlabel](https://search.r-project.org/CRAN/refmans/spatstat.random/html/rlabel.html) | Cross-type K and the fixed-position, fixed-count random-label null. |
| [GEOS 3.11 ConcaveHull](https://libgeos.org/posts/2022-07-01-geos-3-11-0-released/) and [Shapely concave_hull](https://shapely.readthedocs.io/en/stable/reference/shapely.concave_hull.html) | Delaunay-based point-inferred ROI. Conditional simulation in fixed-boundary mode uses the convex hull; interior points conditional on the fixed observed hull are uniform under the stated convex-support null. |

## Build on Windows

Install requirements.txt and PyInstaller into an isolated 64-bit Python environment, then run build-windows.ps1 with -Python pointing to that interpreter. The build runs the test suite, checks the packaged demo and bilingual PDF/Excel outputs, and writes SHA256.txt beside the EXE. MIT license.
