# Fault Detection for Solar PV Deployment

This MSc project investigates fault detection for solar photovoltaic (PV)
deployment. It examines how measurement availability affects PV fault analysis,
same-time irradiance estimation, expected AC-power modelling and cross-domain
screening.

## Research question

How can measurement-aware machine-learning and physics-informed PV analysis
support fault detection and diagnostic screening when sensor availability,
data quality and operating domains differ?

## Objectives

1. **Brazil sensor ablation and five-class PV fault classification.** Evaluate
   Normal, Short-Circuit, Degradation, Open-Circuit and Shadowing classes with
   recording-group-disjoint validation.
2. **Hong Kong SQ1 same-time irradiance estimation and expected AC-power
   modelling.** Compare a measured-irradiance Reference arm, pvlib clear-sky
   irradiance, a machine-learned same-time irradiance estimate and a hybrid
   representation. This is not an ahead-of-time forecasting task.
3. **Brazil-to-Hong-Kong cross-domain screening.** Apply a Brazil-fitted
   classifier using the restricted proxy-comparable shared representation. Hong
   Kong outputs are diagnostic screening results, not validated target-domain
   fault classifications.

## Repository layout

```text
01_HK_Detection/                 Hong Kong Objective 2 implementation
  raw_data/                      Repository-local Hong Kong raw source subset
  outputs/objective2_modeling/   Objective 2 grid, metrics, residuals, and reports
  outputs/objective2_figures/    Objective 2 figures and summary
  scripts/                       Preparation, modelling, diagnostics, and plotting

02_Fault_Classification/         Brazil Objective 1 implementation
  raw_data/                      Source MAT files and converted CSV intermediates
  data/                          Processed Brazil feature dataset
  outputs/                       Grouped-validation results and figures
  scripts/                       Conversion, preprocessing, models, diagnostics, plots

Objective3_Outputs/              Brazil-to-Hong-Kong Objective 3 evidence
  objective3_design/             Feature-contract and comparability diagnostics
  objective3_transfer/           Brazil-fitted transfer screening pipeline and outputs
  objective3_figures/            Objective 3 figures and summary
```

The root directory names `01_HK_Detection/` and `02_Fault_Classification/` are
intentional project identities and are retained as-is.

## Data availability and reproduction

Large source datasets and reproducible generated artefacts are excluded from
the public Git repository but remain in the local working copy. The ignore
rules affect future Git tracking only.

### Brazil

Obtain the source MAT files from the authoritative
[PV fault dataset repository](https://github.com/clayton-h-costa/pv_fault_dataset)
and place them at:

```text
02_Fault_Classification/raw_data/dataset_amb.mat
02_Fault_Classification/raw_data/dataset_elec.mat
```

The reproduction chain is:

```text
dataset_amb.mat + dataset_elec.mat
→ mat_to_csv.py
→ raw_data/converted_csv/
→ fault_preprocessing.py
→ data/fault_dataset.csv
```

The converted CSVs and `fault_dataset.csv` are generated locally and are not
included in the public repository.

### Hong Kong

Obtain the required SQ1 subset from the authoritative
[Dryad dataset](https://doi.org/10.5061/dryad.m37pvmd99) and place it at:

```text
01_HK_Detection/raw_data/dryad_dataset/
```

Only the required subset is needed: `SQ1.csv`, `SQ1_Inverter.csv`, the PV
generation-system metadata TTL and the 2021–2023 meteorological files for
irradiance, temperature, relative humidity, sea-level pressure, visibility,
wind and rainfall. The original Dryad README is retained at:

```text
01_HK_Detection/raw_data/README.md
```

The active chain is:

```text
raw_data/dryad_dataset/
→ objective2_prepare.py
→ outputs/objective2_modeling/objective2_grid.csv
→ Objective 2
→ Objective 3
```

`objective2_grid.csv` is generated locally and is intentionally not stored in
the public repository.

### Generated artefacts intentionally excluded from public Git

These large, reproducible artefacts are generated during execution and excluded
from public Git. They are not missing project files:

- `02_Fault_Classification/raw_data/converted_csv/`
- `02_Fault_Classification/data/fault_dataset.csv`
- `02_Fault_Classification/outputs/grouped_baseline/out_of_fold_predictions.csv`
- `01_HK_Detection/outputs/objective2_modeling/objective2_grid.csv`

## Environment

Use Python 3.9.6. From the repository root, install the direct project
dependencies with:

```bash
pip install -r requirements.txt
```

`requirements.txt` includes `python-docx` for the supporting Hong Kong audit
helper. The submitted Objective 1 workflow used scikit-learn 1.5.2. The
original project did not preserve a complete frozen environment capture for
every package; do not assume that the package versions on a new machine
reproduce every floating-point value exactly.

`fault_preprocessing.py` calls `pvlib.pvsystem.pvwatts_dc` using the
`effective_irradiance` keyword. That interface was verified with pvlib 0.13.0
in an isolated verification runtime; it is not evidence of a complete frozen
submission environment.

The classifier and neural-model fits are resource-intensive. The commands below
show how to regenerate the workflow, but do not guarantee byte-identical
numerical re-creation on every machine.

## Execution order

Run commands from the repository root. Each script uses paths relative to its
own file, so it does not depend on a particular working directory.

### Objective 1 — Brazil grouped classification

```bash
python 02_Fault_Classification/scripts/mat_to_csv.py
python 02_Fault_Classification/scripts/fault_preprocessing.py
python 02_Fault_Classification/scripts/pv_fault_classifier.py
python 02_Fault_Classification/scripts/extended_ablation.py
python 02_Fault_Classification/scripts/event_support_analysis.py
python 02_Fault_Classification/scripts/event_size_diagnostic.py
python 02_Fault_Classification/scripts/split_effect_diagnostic.py
python 02_Fault_Classification/scripts/plot_objective1_results.py
```

### Objective 2 — Hong Kong irradiance and expected power

After placing the raw Dryad source release as described above:

```bash
python 01_HK_Detection/scripts/objective2_prepare.py
python 01_HK_Detection/scripts/objective2_expected_power_arms.py
python 01_HK_Detection/scripts/residual_trend_diagnostic.py
python 01_HK_Detection/scripts/plot_objective2_results.py
```

`hk_data_audit.py` and `objective2_design_checks.py` are read-only supporting
audits for the raw Hong Kong data.

### Objective 3 — Brazil-to-Hong-Kong screening

Objective 3 requires the Objective 1 processed dataset and Objective 2
modelling outputs.

```bash
python Objective3_Outputs/objective3_design/objective3_gate_check.py
python Objective3_Outputs/objective3_design/objective3_feature_audit.py
python Objective3_Outputs/objective3_transfer/objective3_transfer.py
python Objective3_Outputs/objective3_transfer/plot_objective3_results.py
```

## Outputs and interpretation

- Objective 1 uses recording-group-disjoint folds. `recording_group` is a
  split key, never a model feature.
- Objective 2 evaluates all four arms on a common sensor-free population.
  Measured irradiance is unavailable at inference in the no-sensor arms, and
  power is the prediction target rather than a predictor.
- Objective 2's Saola period is a proxy event window. It has no compatible
  timestamp-level fault ground truth.
- Objective 3 is restricted to the shared power-plus-measured-irradiance
  representation. Capacity normalisation makes the Brazil DC and Hong Kong AC
  power measures proxy-comparable; it does not make them physically identical.

## GitHub scope

Keep concise reports, figures, source code and compact summary tables that
support the dissertation. Do not upload the full external Hong Kong source
release, large regenerable Brazil intermediates, virtual environments, Python
caches or local machine files without confirming the relevant redistribution
terms and repository-size limits. Make the largest retained data products
available through an appropriate data release or documented download route when
they exceed practical Git hosting limits.
