# Enrollment Funnel Predictive Modeling Pipeline

A reproducible machine learning workflow for predicting application likelihood across enrollment funnel prospects using Python and R.

This project combines modern data preprocessing, feature engineering, and gradient boosting models to identify prospective students with the highest probability of applying for a future enrollment term.

Built for higher education enrollment management and admissions analytics workflows, the pipeline emphasizes:

- Reproducibility
- Interpretability
- Scalable preprocessing
- Modern machine learning practices
- CRM-ready prediction outputs

---

## Project Overview

The workflow predicts the probability that a prospect or inquiry record will convert into an application.

The pipeline:

1. Cleans and standardizes CRM export data
2. Engineers behavioral and temporal features
3. Trains gradient boosting / XGBoost models
4. Validates model performance using ROC-AUC
5. Scores future-term inquiries
6. Outputs ranked prediction files for recruitment strategy

---

## Key Features

### Reproducible ML Workflow

Built using:

#### Python Stack

- `pandas`
- `numpy`
- `scikit-learn`
- `joblib`

#### R Stack

- `tidymodels`
- `xgboost`
- `tidyverse`
- `recipes`
- `workflows`

The project supports both Python-native and R-native machine learning workflows.

---

### Enrollment Funnel Prediction

The model predicts likelihood of application based on variables including:

| Feature | Purpose |
|---|---|
| Entry Year | Enrollment cycle segmentation |
| Major | Program interest |
| Recruitment Source | Attribution analysis |
| Engagement Score | Inquiry quality proxy |
| Inquiry Age | Recency and timing behavior |
| Active Term | Enrollment intent |

---

### Robust Feature Engineering

The preprocessing pipeline includes:

- CRM column normalization
- ISO timestamp parsing
- Temporal feature generation
- Missing value handling
- One-hot encoding
- Numeric standardization
- Zero variance feature removal

This allows the workflow to handle inconsistent CRM exports with minimal manual cleanup.

---

## Modeling Approaches

### Python Workflow

The Python implementation uses:

```python
GradientBoostingClassifier()
```

with:

- train/validation splitting
- ROC-AUC evaluation
- feature alignment for prediction datasets
- serialized model persistence using `joblib`

---

### R Tidymodels Workflow

The R implementation uses:

```r
boost_tree() %>%
  set_engine("xgboost")
```

with:

- recipes-based preprocessing
- cross-validation
- hyperparameter tuning
- workflow abstraction
- reproducible modeling pipelines

---

## Validation & Evaluation

Model quality is evaluated using:

- ROC-AUC
- Cross-validation
- Holdout validation sets

The workflow is designed to prioritize ranking and prioritization quality rather than binary classification alone.

This supports practical enrollment use cases such as:

- counselor outreach prioritization
- inquiry segmentation
- communication targeting
- recruitment campaign optimization

---

## Output Structure

Prediction outputs include:

| Field | Description |
|---|---|
| element_id | CRM identifier |
| major | Program of interest |
| recruitment_source | Lead source |
| engagement_score | Engagement metric |
| inquiry_age | Days since inquiry |
| prob_apply | Predicted application probability |

Results are sorted by highest predicted probability to support recruitment prioritization workflows.

---

## Repository Notes

This repository intentionally excludes:

- Raw CRM exports
- Student data
- Personally identifiable information (PII)
- Institutional prediction outputs
- Production recruitment files

The repository focuses on:

- Modeling architecture
- Reproducible preprocessing workflows
- Machine learning methodology
- Enrollment analytics infrastructure

Selected screenshots or example outputs may be included for demonstration purposes.

---

## Repository Structure

```text
├── preprocess.R
├── train_model.R
├── predict_2026.R
├── ml_apply_model.py
├── README.md
└── apply_model.pkl
```

---

## Workflow Stages

### 1. Data Preprocessing

The preprocessing stage:

- cleans CRM field names
- parses timestamps
- engineers temporal variables
- separates training vs prediction cohorts

```r
clean_names()
ymd_hms()
parse_number()
```

---

### 2. Model Training

The training workflow includes:

- train/test splitting
- feature preprocessing
- dummy variable encoding
- normalization
- cross-validation
- XGBoost tuning

```r
vfold_cv()
tune_grid()
select_best()
```

---

### 3. Prediction Scoring

Future-term inquiries are scored using the trained model:

```r
predict(model, df, type = "prob")
```

Predictions are exported as ranked CSV outputs for downstream recruitment workflows.

---

## Design Philosophy

This project follows several guiding principles:

- Reproducibility over manual scoring
- Transparent preprocessing over opaque automation
- Practical predictive analytics for enrollment management
- Scalable workflows over spreadsheet modeling
- Interpretable outputs for operational decision-making

The intent is not merely to build a machine learning model, but to create a sustainable enrollment analytics framework.

---

## Potential Future Enhancements

Possible future directions include:

- Shiny dashboard deployment
- Probability threshold optimization
- Lead scoring dashboards
- Feature importance visualization
- Automated retraining workflows
- CRM API integration
- Calibration analysis
- Ensemble modeling
- Funnel conversion forecasting

---

## Running the Workflow

### Python

```bash
python predict_pipeline.py
```

---

### R Workflow

#### Step 1 — Preprocess

```r
source("preprocess_data.R")
```

#### Step 2 — Train Model

```r
source("train_model.R")
```

#### Step 3 — Score Predictions

```r
source("score_predictions.R")
```

---

## Packages Used

### Python

```python
pandas
numpy
scikit-learn
joblib
```

### R

```r
tidyverse
tidymodels
xgboost
recipes
workflows
readr
janitor
lubridate
```

---

## Why This Matters

Many enrollment operations still rely on static lead lists and intuition-based outreach prioritization.

This project demonstrates a more modern enrollment analytics approach:

- reproducible
- data-informed
- scalable
- version controlled
- operationally actionable

It provides a foundation for integrating predictive analytics into day-to-day admissions strategy while maintaining transparency and reproducibility.

---

## Author

Todd Cooley  
Enrollment Analytics & Graduate Admissions  
Focused on reproducible reporting, predictive enrollment analytics, and accessible data workflows.
