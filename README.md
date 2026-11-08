# Open-Data Landslide Susceptibility Mapping in Kodagu, Karnataka

Code and data for: Shankarnarayan, V. K. *Open-Data Landslide Susceptibility Mapping in Kodagu, Karnataka: A Spatially Cross-Validated Assessment.* Arab Gulf Journal of Scientific Research (manuscript AGJSR-02-2026-0031, revision 0711).

Author: Vinay Kellengere Shankarnarayan, RV Institute of Management, Bengaluru · ORCID [0000-0002-5759-0262](https://orcid.org/0000-0002-5759-0262)
Archived release: Zenodo DOI **[to be minted]**

## Reproduce everything

```bash
pip install -r requirements.txt
bash run_all.sh
```

`run_all.sh` regenerates every table, figure and supplementary result in the paper into `outputs/` and `figures/`. Random seed 42 throughout (pseudo-absence sensitivity draws: seeds 100–104).

## Inputs (`data/`)

| File | Source | Notes |
|---|---|---|
| `Kodagu_landslide_candidates_v2.csv` | Google Earth Engine export of `gee/gee_inventory_script.js` (run 7 Nov 2026) | 5,247 candidate scar patches: centroid lon/lat, patch area (m²) |
| `Kodagu_SRTM_merged.tif` | USGS EarthExplorer, SRTMGL1 v3 | Merged 1 arc-second tiles N11–N12, E075–E076 |
| `boundary/Kodagu.shp` | Survey of India / KSRSAC | District boundary used as the analysis mask |

## Scripts (`scripts/`, run in order)

| Script | What it does | Main outputs |
|---|---|---|
| `00_prepare_dem.py` | Clip SRTM, bilinear reprojection to 30 m UTM 43N | `data/dem_utm.tif` |
| `01_factors.py` | Slope (Horn), total curvature (Zevenbergen–Thorne), elevation, northness, eastness, TWI (D8) | `data/f_*.tif` |
| `02_samples.py` | Inventory screens (district; ≤ 2 ha; centroid slope ≥ 15°) → 325 scars; 325 pseudo-absences (slope ≥ 5°, > 500 m from scars) | `inventory_points.csv`, `pseudo_absences.csv`, `model_dataset.csv`, `inventory_screening_log.csv` |
| `03_models.py` | VIF; IV, LR, RF with/without slope; random 70/30 hold-out, random 5-fold, 4 km spatial blocks (+1 km buffer; 10 km) | `holdout_metrics.csv`, `cv_summary.csv`, `cv_fold_auc.csv`, `spatial_folds_4km.csv`, `spatial_fold_composition.csv`, `iv_class_table_training.csv`, `vif_and_designs.json` |
| `04_map_shap.py` | LR coefficients; susceptibility rasters (equal-interval classes); concentration ratio; SHAP | `lr_coefficients.csv`, `lsi_lr_probability.tif`, `lsi_lr_classes.tif`, `susceptibility_class_statistics.csv`, `concentration_statistics.json`, `shap_*.csv` |
| `06_figures.py` | Figures 1–3 | `figures/Figure_*.png` |
| `07_sensitivity_overlay.py` | Pseudo-absence exclusion-distance sensitivity (250/500/750 m × 5 draws) | `pseudoabsence_sensitivity.csv` |

## Google Earth Engine

`gee/gee_inventory_script.js` maps the candidate scars (Sentinel-2 Level-1C, Cloud Score+, NDVI change between 1 Oct 2017 – 28 Feb 2018 and 1 Oct 2018 – 28 Feb 2019). Paste it into https://code.earthengine.google.com, run it, and start the export task. It is the only step outside Python; no manual GIS editing was applied.

## Licence

Code: MIT. Author-generated data (inventory, pseudo-absences, outputs): CC BY 4.0.
