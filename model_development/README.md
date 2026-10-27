# ARGUS2 Model Development

This folder contains the notebooks used to develop the ARGUS2 (`v2_ARGUS25`) models. The models estimate whether coronary artery disease (CAD) can be safely excluded in patients who present at cardiology with CAD related symptoms, such as chest pain and dyspnea. Patients whose estimated risk is below **15%** are classified as low risk. The models are intended to rule CAD out, so negative predictive value (NPV) is the primary performance measures.

## Notebooks

| Notebook | Description |
|---|---|
| `1_add_labels_to_data.ipynb` | Derivation of the outcome label |
| `2_ARGUS_model_build.ipynb` | Data split, baselines, model training, calibration, evaluation and feature importance |

The input dataset is produced by the training pipeline ([`src/argus/Pipeline/Run_train_preprocess.py`](../src/argus/Pipeline/Run_train_preprocess.py)). That pipeline selects consultations for chest pain without a documented history of CAD, then adds laboratory, ECG and body- and blood-pressure measurements, giving one row per patient visit. Notebook 1 reads `data/female_argus_data.parquet` and `data/male_argus_data.parquet` and writes the label column back to the datasets, which notebook 2 then reads.

## Outcome definition

Each visit is labelled from three sources:

- **Radiology reports**, text-mined with [`src/argus/label/labeling_script.py`](../src/argus/label/labeling_script.py). A report is linked to a visit if the scan was performed 0–365 days after the consultation. If several reports are linked to one visit, the highest value of each extracted field is kept, so that risk is not underestimated.
- **Cardiology interventions** (OR procedures) within one year after the consultation.
- **Radiological procedures** performed (CT, MRI, PET/CT, SPECT) within one year after the consultation.

Rules are applied in order, and a later rule overrides an earlier one:

| Condition | Label |
|---|---|
| No diagnostic information available | 0 |
| Coronary artery calcium score (CACS) = 0 | 0 |
| Cardiology intervention within one year | 1 |
| CAD on CT (stenosis, stenosis grade, CAD-RADS or MESA flag, or CACS > 400) | 1 |
| MR angiography, cardiac MRI, PET/CT or SPECT performed | 1 |
| Otherwise (inconclusive) | excluded |

(https://github.com/UPOD-datascience/ARGUS2_public/blob/main/model_development/Fig%20S1_Flowchart%20outcome%20definition%20ARGUS.png?raw=true)

## Model development

**Study population.** The labelled dataset contained 6,332 visits from 6,076 patients. After excluding visits with more than 80% missing values, 5,990 visits from 5,755 patients remained: 2,626 visits from 2,512 women and 3,364 visits from 3,243 men. CAD prevalence was 30.4%. Mean age was 56.7 (SD 17.9) for women and 54.5 (SD 16.7) for men.

**Features (35).** Missing values were left as missing and handled natively by XGBoost.
- Age, BMI, systolic and diastolic blood pressure
- 21 laboratory values: lipids, glucose, kidney and liver function, complete blood count, electrolytes, CRP, TSH
- 10 ECG features: infarction classification, plus ST-deviation, T-morphology and regional-ischemia scores for the anterior, inferior and lateral regions

**Data split.** Separate models were developed for women and men. Each dataset was split at patient level into 85% training and 15% test data (`random_state=42`, not stratified on outcome), so that no patient appears in both sets. The patient IDs are saved to `data/models/train_patients.csv` and `data/models/test_patients.csv`. The training uses 10-fold cross-validation.

| | Women | Men |
|---|---|---|
| Training visits | 2,230 | 2,863 |
| Test visits | 396 | 501 |

**Baseline.**:
- Classifying every patient as low risk. On the training data this gives an NPV of 0.71 for women and 0.68 for men.

**Model.** An XGBoost classifier for each sex:

```python
XGBClassifier(n_estimators=10, max_depth=5, learning_rate=0.1,
              reg_alpha=1, reg_lambda=2, scale_pos_weight=n_neg / n_pos,
              booster="gbtree", eval_metric="logloss", random_state=42)
```

The hyperparameters follow the original ARGUS model/ In 10-fold stratified grouped cross-validation, the 10-tree models reached an NPV of 0.87 (women) and 0.84 (men), a sensitivity of 0.79 and 0.77, and a ROC AUC of 0.73 and 0.72. 

**Calibration.** Predicted probabilities were calibrated with Platt scaling (`CalibratedClassifierCV`, sigmoid, 10-fold) on the training data. On the test set, calibration lowered the Brier score from 0.197 to 0.172 for women and from 0.209 to 0.178 for men. Calibration intercept and slope were estimated by logistic recalibration, and calibration curves were smoothed with LOESS.

**Decision threshold.** A calibrated probability ≥ 0.15 is classified as *possible CAD, further diagnostics indicated*.

**Feature importance.** SHAP values were computed for the uncalibrated XGBoost models on the test set and compared between the female and male models.

## Final performance

Held-out test set; calibrated models at a threshold of 0.15. 95% confidence intervals in parentheses: Wilson intervals for proportions, and percentile bootstrap intervals (1000 resamples) for AUROC and AUPRC.

| Metric | Women (n = 396) | Men (n = 501) |
|---|---|---|
| CAD prevalence | 29.3% | 29.1% |
| **NPV** | **0.96 (0.91–0.98)** | **0.92 (0.86–0.95)** |
| **Sensitivity** | **0.95 (0.89–0.98)** | **0.92 (0.86–0.95)** |
| Specificity | 0.50 (0.44–0.55) | 0.37 (0.32–0.42) |
| PPV | 0.44 (0.38–0.50) | 0.37 (0.33–0.43) |
| AUROC | 0.76 (0.70–0.81) | 0.73 (0.69–0.78) |
| AUPRC | 0.49 (0.41–0.59) | 0.50 (0.42–0.58) |
| TN / FP / FN / TP | 139 / 141 / 6 / 110 | 131 / 224 / 12 / 134 |
| Classified as low risk | 36.6% | 28.5% |
| Calibration intercept / slope | 0.28 / 1.26 | 0.03 / 1.12 |
| Brier score | 0.172 | 0.178 |

Classifying every patient in the test set as low risk would give an NPV of 0.71 for both women and men. The models raise NPV to 0.96 for women and 0.92 for men while classifying about a third of women and more than a quarter of men as low risk.

## Limitations

- Discrimination is moderate (AUROC 0.73–0.76), and many features are often missing: most laboratory values are available for 20–57% of visits.

