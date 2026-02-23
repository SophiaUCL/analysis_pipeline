# HCT hippocampal unit classification (pyramidal vs interneuron)

This repo reproduces a paper-style classification of hippocampal units into **pyramidal vs interneuron** using a 4-metric feature space and manual gating in PCA space.

## Summary of method (paper-faithful)

Cells are classified (or left unclassified) using PCA on four per-unit metrics:

1. **Spike width**: `peak_to_trough_ms`
2. **Mean firing rate**: `firing_rate_hz`
3. **Burst index** (paper definition):  
   `burst_index = ACG(0–10 ms) / ACG(40–50 ms)`
4. **Theta rhythmicity**: `oscillation_score` (FFT-based ACG theta-band peak / mean power)

Workflow:
- Compute/assemble the 4 metrics per unit
- Run PCA on the 4 metrics
- Plot PC1 vs PC2
- **Manually draw a polygon** around the pyramidal cluster
- Optionally verify visually using waveform + ACG + ratemap “inspection sheets”

## Data expectations

### Read-only inputs (session derivatives)
This pipeline expects a session `derivatives_base` containing:
- SpikeInterface analyzer folder:  
  `analysis/cell_characteristics/unit_features/spikeinterface_data`
- Unit overview CSVs (from the existing pipeline):  
  `analysis/cell_characteristics/unit_features/all_units_overview/unit_waveform_metrics.csv`  
  `analysis/cell_characteristics/unit_features/all_units_overview/unit_metrics.csv`
- Phy curation output (for `phy_group`):  
  a `cluster_group.tsv` somewhere under `derivatives_base` (the loader searches for it)

### Optional read-only inputs (for inspection sheets)
Existing image folders (if available):
- Waveform + ACG images:  
  `analysis/cell_characteristics/unit_features/auto_and_wv/`  
  filenames are commonly zero-padded, e.g. `unit_001.png`
- Ratemap images:  
  `analysis/cell_characteristics/spatial_features/ratemaps_and_hd/`  
  filenames commonly like `unit_1_rm_hd.png`

## Outputs (what this repo produces)

- 4-metric tables:
  - `unit_features_4metrics_all_units.csv`
  - `unit_features_4metrics_good_units.csv` (Phy good only)

- Paper ISI QC audit (2 ms rule):
  - `paper_qc_2ms_all_units.csv`
  - `paper_qc_2ms_good_units.csv`
  These include `paper_exclude`; `meets_paper_isi = ~paper_exclude` is derived/merged later.

- Classification tables for PCA / gating:
  - `unit_classification_features_noISI_allunits.csv` (no ISI filtering, no min spike filter)
  - `unit_classification_features_noISI_min500spikes.csv` (**min spikecount applied**, see below)
  - `unit_classification_features_noISI_min500spikes_withISI.csv` (adds `meets_paper_isi` annotation)

- PCA outputs:
  - `pca_coords_noISI_min500.csv` (unit_id + PC1/PC2 + metadata)
  - `PC1_PC2_noISI_min500.png`

- Polygon labels:
  - `labels_polygon_noISI_min500.csv` (unit_id → pyramidal/interneuron)

- Inspection sheets (PNGs):
  - one per unit, combining waveform+ACG image + ratemap image + metric readout

## Important: min spikecount filter (>= 500 spikes)

In the “noISI_min500” workflow, units are filtered by spike count **before PCA**:

- Implemented in: `build_features_noISI_min500.py`
- Criterion: `n_spikes >= 500`
- Effect: PCA + polygon gating (and the inspection sheets tied to that run) only include units passing this filter.
- Note: ISI QC is NOT used for inclusion in this branch; it is merged as an annotation (`meets_paper_isi`) for reference.

## Scripts (high-level)

### `audit_qc_2ms_paper.py`
Paper-style ISI QC audit: fraction of spikes whose next ISI < 2 ms; exclude if >1%.
Writes `paper_qc_2ms_*_units.csv`.

### `make_unit_features_4metrics_good_and_all_units.py`
Computes burst index + oscillation score from spike trains and merges:
- `peak_to_trough_ms` from waveform metrics CSV
- `firing_rate_hz` from unit metrics CSV
- `phy_group` from Phy `cluster_group.tsv`
Writes `unit_features_4metrics_*_units.csv`.

### `build_features_noISI_min500.py`
Builds the simplified classification feature tables for PCA:
- a noISI all-units table
- a **noISI + min500** table (`n_spikes >= 500`) used for PCA

### `pca_noISI_min500.py`
Runs PCA on the 4 metrics and saves PC1/PC2 coordinates.

### `gate_pyramidal_polygon_with_density_contours.py`
Interactive polygon selection on the PC1/PC2 scatter (optionally with density contours).
Outputs `labels_polygon_noISI_min500.csv` (unit_id → pyramidal/interneuron).

### `add_isi_annotation.py`
Merges `meets_paper_isi` into the min500 feature table for inspection outputs.

### `make_inspection_sheets_from_existing_images.py`
Creates per-unit inspection sheets from existing waveform+ACG and ratemap images plus metric text.

## Typical run order (new dataset)

1. (Optional) ISI QC audit:
   - `audit_qc_2ms_paper.py`
2. Build 4-metric tables:
   - `make_unit_features_4metrics_good_and_all_units.py`
3. Build PCA feature table (min500):
   - `build_features_noISI_min500.py`
4. PCA:
   - `pca_noISI_min500.py`
5. Polygon gate:
   - `gate_pyramidal_polygon_with_density_contours.py`
6. Merge ISI annotation:
   - `add_isi_annotation.py`
7. Inspection sheets:
   - `make_inspection_sheets_from_existing_images.py`

## Notes
- Phy “good” is a curation label and is kept explicitly as `phy_group`.
- Paper ISI QC is also kept explicitly as `meets_paper_isi`.
- The polygon gate is intentionally manual to match the paper’s approach.
