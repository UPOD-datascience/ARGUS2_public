# Analysis scripts – ARGUS silent trial

This folder holds the data analysis scripts used for the evaluation of the ARGUS silent trial. The silent trial ran the ARGUS software in the background for about 180 days without showing predictions to clinicians. The notebooks should be used in numbered order: notebook 2 creates the labelled dataset that notebook 3 uses.

| Notebook | Purpose |
|---|---|
| [1_Analyse_log_data.ipynb](1_Analyse_log_data.ipynb) | Analyse the technical performance during inference (patient flow, errors, latency). Plots latency per module (absolute and normalized).
| [2_Add_labels_to_silent_trial_data.ipynb](2_Add_labels_to_silent_trial_data.ipynb) | Determines the true label per row and stores the results in `../data/df_labels_final_<date>_info.parquet` |
| [3_Performance_assesment_silent_trial.ipynb](3_Performance_assesment_silent_trial.ipynb) | Analyses how well the model performed (discriminative performance and calibration), and includes the failure analysis. (Table 1, metrics tables)

Shared database helpers are in [functions/analysis_functions.py](functions/analysis_functions.py). Generated tables and figures go to [output/](output/).

---

## 1. Log data analysis – `1_Analyse_log_data.ipynb`

Input: `logging.parquet` (software logs from the silent trial).

| Section | What it does |
|---|---|
| Trial duration | Measures the time from the first to the last log entry |
| Drop duplicate runs | Excludes days with more than one `run_id` (early cycles up to 13 Jan, and the fix on 11 May) |
| Number of patients during preprocessing | Uses regex on the log messages to count patients at each step: initial selection, age filter, CAD-related complaints, and lab, ECG, BMI and blood-pressure availability. Also counts by sex, flags runs with errors and measures the downtime caused by errors |
| Duplicates within runs | Explains why availability is reported as percentages: consecutive runs overlap by one day |
| Latency analysis | Splits runtime per run into ETL/ECG, measurements, preprocessing and prediction. Includes a check on the high latency between 23 Jan and 7 Feb |
| Latency figures | Boxplots of latency per module, raw and normalised (Figure 3) |
| Latency vs batch size | Batch-level latency against batch size for the preprocessing modules and the patient-selection modules |

## 2. Outcome labelling – `2_Add_labels_to_silent_trial_data.ipynb`

Input: silent-trial predictions plus hospital data exports from 2026-08-03 (`.sas7bdat`). Output: a labelled dataset with one row per patient and index date.

| Section | Used for |
|---|---|
| Load OR interventions: planned / performed OR | Generate binary indicator columns `Order_Ok`, `Verrichting_Ok` |
| RAD verrichtingen | source: `rad_verrichting` --> Cardiac imaging performed, split into MRI/PET/SPECT and CT |
| Check radiology rapport for all CTs | source: `rad_verslagen` --> Checks that every CT has a report |
| Check radiology orders | source: `rad_orders` --> MRI/CT orders. Flags orders planned more than 180 days ahead (`Long_rad_waitinglist`) |
| Load CT rapports / labels Bram | source: `labels_from_model.parquet` --> Findings extracted from CT reports by a model (stenosis, plaque, CAD-RADS, CACS, MESA) |
| Create label logic | – | See the rules below |
| Add prediction label | – | `pred_label = predicted_risk >= 0.15` |

**Label rules**
1. **Label = 1** if there is an OR order or procedure, or an MRI/PET/SPECT procedure or order.
2. Otherwise, use the CT report: **1** if it shows stenosis, CAD-RADS, MESA or CACS > 400; **0** if CACS = 0.
3. **Label = 0** if there is no record of any OR or rediology procedure or order.
4. Records that are still inconclusive are dropped.

## 3. Performance assessment – `3_Performance_assesment_silent_trial.ipynb`

Input: the labelled dataset from notebook 2, plus the calibrated XGBoost models (`../data/models/argus_calibrated_xgb_model_{f,m}.pkl`). Excludes records on the long radiology waiting list. All analyses are split by sex.

| Section | What it does | 
|---|---|
| Table 1 info | Cohort characteristics by sex, with missingness markers. |
| Label info| Breaks down where each label came from | 
| Performance assessment | Confusion matrix, age and sex distribution, distribution of predicted risk. | 
| Calibration | LOESS-smoothed calibration curves with bootstrap confidence intervals | 
| Discrimination | AUC and ROC curves. Sensitivity, specificity, PPV, NPV and F1, with bootstrap and Wilson confidence intervals | 
| Failure analysis | Compares missingness and predictor values between false negatives and true positives, and between false positives and true negatives | 
| Compare data availability with development set | Compares the label distribution and missingness of the silent trial with the development set used for retraining | 
| Compare values across all datasets | Tests whether predictor values differ between the development set and silent-trial data | 
| Test feature importance | Permutation importance on the silent-trial data, per sex | 
