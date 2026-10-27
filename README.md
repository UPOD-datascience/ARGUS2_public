# ARGUS2

ARGUS2 is a clinical decision support tool wich estimates whether coronary artery disease (CAD) can be safely excluded in patients who present at cardiology with CAD related symptoms, Patients with an estimated risk below 15% are classified as low risk, for whom further diagnostic testing is not indicated. Separate calibrated XGBoost models are used for women and men.

This repository contains the code to:

1. [`Model development`](model_development/README.md): derive the outcome label and develop the models,
2. [`ARGUS_python_package`](src/argus/): Software (python package) used to run the models in *shadow mode* during the silent trial: predictions are made automatically and stored, but are not shown to clinicians, and
3. [`Silent trial results`](analysis_scripts/README.md): analyse the results (logs and predictions) of the silent trial

The code was developed at UMC Utrecht for the model version `v2_ARGUS25`.

## Data availability

The models were developed on pseudonymised patient data from UMC Utrecht. This data cannot be shared publicly because of patient privacy regulations. The repository therefore contains **no patient data and no trained models**. The code shows how the data were processed and how the models were built, but it cannot be run end to end outside the UMC Utrecht environment without adapting the data loaders.


## Repository structure

```
├── configs/             YAML configuration per processing step (columns, file names, filters, outlier limits)
├── assets/              Word lists used for text cleaning of clinical notes
├── model_development/   Outcome labelling and model development (see notebooks/README.md)
├── analysis_scripts/    Analysis of the silent trial (see analysis_scripts//README.md)
├── src/argus/           Python package
├── tests/               Unit tests
├── pyproject.toml       Package definition and dependencies
├── run_uv.sh            Start the API locally with uv
├── write_manifest.sh    Create the deployment manifest for Posit Connect
└── deploy.sh            Deploy the API to Posit Connect
```

### The `argus` package

| Module | Purpose |
|---|---|
| `Pipeline/` | Entry points. `Load_and_preprocess.py` runs all processing steps and merges the results; `Run_train_preprocess.py` builds the training dataset; `Run.py` runs inference. |
| `data_loaders/` | Load raw data. Training loaders read SAS exports from the research drive; inference loaders query the hospital data platform. `factory.py` selects the right loader for the mode. |
| `ETL/` | Processing per data source: patient selection (`ETL_patsel`), selection of chest-pain consultations (`ETL_pob_filter`) and exclusion of patients with a cardiac history (`ETL_vg_filter`), both from consultation text; laboratory results (`ETL_lab`), ECG (`ETL_ecg`), body and blood-pressure measurements (`ETL_metingen`), risk factors (`ETL_risk_factors`); and merging (`merged_data`). |
| `process/` | Outlier detection for measurements. |
| `label/` | Text mining of radiology reports to extract CAD findings (stenosis, CAD-RADS, calcium score) for the outcome label. |
| `data_models/` | Schemas that validate each dataset and the model output. |
| `api/` | FastAPI application for running inference in shadow mode. |
| `database/`, `utils/` | Database connections, configuration loading, logging, file paths and shared helpers. |

## Pipeline overview

The same processing pipeline is used for training and inference. Only the data source differs; you choose it with `mode="train"` or `mode="inference"`.

```
patient selection ─┐
  (chest pain,     │
   no CAD history) │
lab results ───────┤
ECG ───────────────┼──► merge ──► one row per visit ──► model (female / male) ──► predicted risk
body / BP ─────────┘    (+ outlier handling)
```

1. **Patient selection.** Outpatient cardiology visits are selected when the consultation note mentions chest pain and does not mention a history of CAD.
2. **Feature extraction.** Laboratory values, ECG features, BMI and blood pressure measured around the visit are added.
3. **Merge.** All sources are merged into one row per visit, with 35 features.
4. **Prediction.** The sex-specific calibrated XGBoost model gives the predicted risk. The fraction of missing features is also stored.

Configuration for each step is in `configs/config_<step>.yaml`. Each file contains a section for each model version and mode.

## Installation

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync          # creates .venv and installs the package with its dependencies
```

## Usage

### 1. Build the training dataset

```bash
uv run python -m argus.Pipeline.Run_train_preprocess
```

This reads the raw data exports listed under `train` in `configs/config_general.yaml` from the UMC Utrecht research drive (see `utils/PathManager.py`). It writes `data/female_argus_data.parquet` and `data/male_argus_data.parquet`. To use other data, adapt the training loaders in `data_loaders/training.py` and the file names in the config.

### 2. Label the data and develop the models

Radiology reports are text-mined with `src/argus/label/labeling_script.py`; run it with `--help` to see the options. The notebooks in [`notebooks/`](notebooks/README.md) then derive the outcome label and train, calibrate and evaluate the models. The notebook README describes the methods and the final performance.

Trained models are stored as `models/argus_calibrated_xgb_model_f.pkl` and `models/argus_calibrated_xgb_model_m.pkl`.

### 3. Run inference

From Python:

```python
from argus.Pipeline.Run import RunInference

df_female, df_male = RunInference(mode="inference").run()
```

As an API (used for the shadow-mode deployment):

```bash
./run_uv.sh      # starts the API at http://127.0.0.1:8000, docs at /docs
```

| Endpoint | Description |
|---|---|
| `POST /run_from_last` | Predict for all visits since the last stored prediction |
| `POST /run_all` | Predict for all available visits |
| `GET /job_status/{job_id}`, `GET /jobs` | Follow inference jobs, which run in the background |
| `GET /shadow_test/latest`, `/stats`, `/by_run/{run_id}` | Read stored predictions |
| `GET /health` | Health check |

The `POST` endpoints require an `X-API-KEY` header. Predictions and logs are written to the `shadow_test` and `logging_shadow_test` tables.

### Environment variables

Inference, the API and the labelling script read their settings from a `.env` file in the project root. This file is not part of the repository.

## Development

```bash
uv run pytest         # run tests
uv run ruff check .   # lint
```

The GitHub Actions workflows lint the code and check that notebooks contain no output, so that no patient data are committed.

## Outputs are stripped
Notebook outputs are stripped automatically on commit. After cloning, run once:

    uv tool install nbstripout
    nbstripout --install

## License

This project is licensed under the GNU General Public License v3.0; see [LICENSE](LICENSE).

Copyright © 2026 UMC Utrecht
