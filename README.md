# Medical Risk Prediction Using Machine Learning

A synthetic medical machine-learning project that generates patient data, introduces realistic data-quality problems, cleans the data, engineers features, trains multiple classification models, optimizes their hyperparameters, and combines the optimized models into a soft-voting ensemble for final high-risk prediction.

> **Important:** This project uses synthetic data for educational and software-development purposes. It is not a clinical decision-support system and must not be used for diagnosis, treatment, or real patient care.

## Project Pipeline

```text
Synthetic Data Generation
        ↓
Add Missing Values, Duplicates, Outliers and Entry Errors
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Train/Test Split + Standardization
        ↓
Train 5 Classification Models
        ↓
Hyperparameter Optimization
        ↓
Soft-Voting Ensemble
        ↓
Evaluation and Visualizations
```

## Models

The project trains and optimizes:

- Random Forest
- Gradient Boosting
- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

The final ensemble averages the predicted probability of the positive class from all five optimized models and classifies a patient as high risk when the ensemble probability is at least 0.5.

## Synthetic Dataset

The relationship-based synthetic generator creates 1,000 patient records with variables including age, gender, height, weight, BMI, blood pressure, heart rate, temperature, blood sugar, cholesterol, hemoglobin, smoking, alcohol consumption, exercise, diabetes, hypertension, heart disease, hospital visits, insurance type, and risk score.

The generator uses synthetic probabilistic rules so that disease and risk outcomes are related to selected patient characteristics rather than being purely random.

## Data Quality Simulation

The impurity stage introduces:

- Missing values in selected numerical columns
- Duplicate-like records
- Numerical outliers
- Negative weight values
- Inconsistent gender entries

The cleaning stage handles missing values, duplicate medical records, numerical outliers, categorical normalization, negative values, and BMI recalculation.

## Feature Engineering

`High_Risk` is created from whether `Risk_Score` is above its median. Categorical variables are label encoded and the selected numerical and encoded variables are used for classification.

### Important modeling note

The current synthetic design includes diabetes, hypertension, and heart disease in both the risk-score construction and the model features. This can create target leakage/circularity and may produce unusually strong performance. Results should therefore be interpreted as a demonstration of the pipeline, not as evidence of clinical predictive performance.

## Repository Structure

```text
medical-risk-prediction-ml/
│
├── README.md
├── requirements.txt
├── .gitignore
├── run_pipeline.py
│
├── synthetic_data_gen.py
├── add_impurities.py
├── data_preparation.py
├── feature_engineering.py
├── train_test_split.py
├── models_training.py
├── optimization_model.py
├── ensemble_model.py
├── evaluate_models.py
│
├── data/
│   └── README.md
├── models/
│   └── README.md
└── results/
    └── README.md
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Complete Project

From the repository root:

```bash
python run_pipeline.py
```

The pipeline runs the stages in order and generates the intermediate datasets, trained models, optimized models, predictions, metrics, and plots locally.

## Run Individual Stages

```bash
python synthetic_data_gen.py
python add_impurities.py
python data_preparation.py
python feature_engineering.py
python train_test_split.py
python models_training.py
python optimization_model.py
python ensemble_model.py
```

`evaluate_models.py` evaluates the baseline models saved by `models_training.py`.

## Main Generated Outputs

| Output | Purpose |
|---|---|
| `clean_patient_data.csv` | Initial synthetic dataset |
| `dirty_patient_data.csv` | Dataset after simulated impurities |
| `final_cleaned_data.csv` | Cleaned dataset |
| `features.csv` | Model features |
| `target.csv` | `High_Risk` target |
| `X_train.csv`, `X_test.csv` | Split feature data |
| `y_train.csv`, `y_test.csv` | Split target data |
| `optimized_model_results.csv` | Optimized model metrics |
| `optimized_models.pkl` | All optimized models |
| `best_optimized_model.pkl` | Best optimized model by cross-validation accuracy |
| `ensemble_predictions.csv` | Final ensemble predictions and probabilities |
| `model_vs_ensemble_results.csv` | Individual vs ensemble metrics |
| `ensemble_model.pkl` | Saved ensemble configuration |
| `optimized_model_comparison.png` | Optimized model comparison |
| `model_vs_ensemble.png` | Individual models vs ensemble |

Generated datasets, model artifacts, and plots are excluded from Git by `.gitignore` so the repository remains lightweight and reproducible.

## Evaluation Metrics

The project reports:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC
- Classification report
- Confusion matrix

Hyperparameter selection is based on 5-fold cross-validation on the training set. The held-out test set is used for final performance reporting.

## Reproducibility

Random seeds are set to `42` in the synthetic-data generation, train/test split, and model configuration where applicable.

## Limitations

This is an educational synthetic-data project. The synthetic relationships do not represent validated clinical relationships, the dataset is not representative of a real patient population, and the model should not be interpreted as medically validated.

## License

You may add a license appropriate to your intended use. For a personal portfolio project, the MIT License is a common permissive choice.
