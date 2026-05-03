# Credit Risk Model Backtesting (Basel Framework)

## Project Overview

This project implements an end-to-end quantitative credit risk modeling pipeline for evaluating:

* PD — Probability of Default
* LGD — Loss Given Default
* EAD — Exposure at Default

The framework simulates a credit portfolio, trains multiple machine learning models, and evaluates model performance using statistical metrics, stress testing, Monte Carlo simulation, and bootstrap confidence intervals.

The methodology aligns with concepts used in Basel regulatory risk frameworks applied in banking risk management.

The system automatically:

* Generates a credit dataset (~50,000 records)
* Trains multiple PD models
* Selects the best model based on ROC-AUC
* Estimates portfolio expected loss
* Performs stress testing
* Computes portfolio Value-at-Risk (VaR)
* Generates visualizations and reports

---

# Dataset

The dataset is synthetically generated to simulate realistic credit portfolio data.

## Dataset Size

50,000 credit observations

## Features

| Feature           | Description              |
| ----------------- | ------------------------ |
| age               | Borrower age             |
| income            | Annual borrower income   |
| loan_amount       | Loan amount              |
| loan_term         | Loan duration (months)   |
| credit_cards      | Number of credit cards   |
| previous_defaults | Historical default count |
| credit_score      | Borrower credit score    |

## Target Variables

| Variable | Meaning                |
| -------- | ---------------------- |
| PD       | Probability of Default |
| LGD      | Loss Given Default     |
| EAD      | Exposure at Default    |

---

# Project Pipeline

The pipeline follows these stages:

1. Data Generation
2. Train–Test Split
3. Class Imbalance Handling using SMOTE
4. Training Multiple PD Models
5. Best Model Selection
6. LGD Model Training
7. Portfolio Expected Loss Calculation
8. Stress Scenario Simulation
9. Monte Carlo Portfolio Risk Simulation
10. Bootstrap Confidence Intervals
11. Visualization & Reporting

---

# Machine Learning Models

The project evaluates several ML models for Probability of Default prediction:

| Model                   | Purpose                         |
| ----------------------- | ------------------------------- |
| Logistic Regression     | Baseline interpretable PD model |
| Random Forest           | Non-linear ensemble PD model    |
| Gradient Boosting       | Boosted ensemble PD model       |
| Random Forest Regressor | LGD estimation                  |

The pipeline automatically selects the best model based on ROC-AUC.

---

# Quantification of Model Performance

Quantitative metrics calculated by the pipeline:

| Metric              | Meaning                          |
| ------------------- | -------------------------------- |
| ROC-AUC             | Model discrimination power       |
| Brier Score         | Probability calibration accuracy |
| Hit Rate            | Default classification accuracy  |
| Expected Loss       | PD × LGD × EAD                   |
| Stress Loss         | Portfolio loss under stressed PD |
| Portfolio VaR       | 99% Value-at-Risk                |
| Confidence Interval | Uncertainty of expected loss     |

These metrics provide **quantitative validation of model performance and portfolio risk exposure**.

---

# Model Results (Actual Program Output)

## PD Model Training Results

```
Training PD Models...

LogisticRegression ROC-AUC: 0.6162
RandomForest ROC-AUC: 0.5643
GradientBoosting ROC-AUC: 0.5589

Best PD Model: LogisticRegression | ROC-AUC: 0.6162
```

The Logistic Regression model performed best, achieving the highest ROC-AUC among the tested models.

---

# Quantitative Results

| Metric              | Value              |
| ------------------- | ------------------ |
| Best Model          | LogisticRegression |
| ROC-AUC             | 0.6162             |
| Brier Score         | 0.2376             |
| Hit Rate            | 0.5872             |
| Expected Loss       | 1,478              |
| Stress Loss         | 17,502,970         |
| Portfolio VaR (99%) | 17,845,814         |
| CI Lower            | 1,466              |
| CI Upper            | 1,490              |

---

# Quantitative Analysis of Results

## Model Discrimination

ROC-AUC = 0.616

Indicates moderate predictive power — the model performs better than random classification but still has room for improvement.

## Probability Calibration

Brier Score = 0.2376

Shows reasonable probability estimates; lower values indicate better calibration.

## Default Prediction Accuracy

Hit Rate = 58.7%

Percentage of correct classifications using the default threshold.

## Portfolio Risk Estimation

Expected Portfolio Loss = 1,478

Represents the average predicted loss across the portfolio:

**Expected Loss = PD × LGD × EAD**

## Stress Testing

Stress Portfolio Loss = 17,502,970

Highlights how portfolio risk escalates under adverse default scenarios.

## Portfolio Value-at-Risk

Portfolio VaR (99%) = 17,845,814

Indicates that 99% of simulated portfolio losses fall below this level.

## Confidence Interval for Expected Loss

95% CI = [1,466 , 1,490]

Quantifies uncertainty around the expected loss estimate, useful for capital allocation.

---

# Key Insights

### Best PD Model Selection

Logistic Regression achieved the highest ROC-AUC (0.616), showing moderate predictive power.

### Probability Calibration & Hit Rate

Brier Score (0.2376) and Hit Rate (58.7%) indicate reasonable probability prediction accuracy.

### Portfolio Risk Exposure

The large difference between expected and stressed losses (1,478 vs 17,502,970) demonstrates how adverse scenarios can dramatically increase portfolio risk.

### Uncertainty Estimation

The 95% confidence interval [1,466 , 1,490] quantifies the reliability of expected loss predictions.

### Feature Insights

Logistic Regression feature importance highlights which borrower characteristics most influence default probabilities, aiding risk mitigation.

Overall, the pipeline provides Basel-compliant quantitative evaluation, supporting risk management, stress testing, and portfolio loss prediction.

---

# Generated Outputs

## Figures (`figures/`)

* PD probability histogram (`pd_histogram.png`)
* Predicted vs actual LGD (`lgd_scatter.png`)
* Bootstrap expected loss distribution (`bootstrap_ci.png`)
* Logistic regression feature importance (`logreg_feature_importance.png`)

## Models (`models/`)

* `best_pd_model.pkl`
* `lgd_model.pkl`

## Results (`results/`)

* `model_metrics.csv`
* `model_report.txt`

---

# Repository Structure

```
credit-risk-model-backtesting-basel/

data/
    credit_data.csv

models/
    best_pd_model.pkl
    lgd_model.pkl

figures/
    pd_histogram.png
    lgd_scatter.png
    bootstrap_ci.png
    logreg_feature_importance.png

results/
    model_metrics.csv
    model_report.txt

notebooks/
    credit_risk_pipeline.ipynb

scripts/
    data_generation.py
    portfolio_risk.py
    model_training.py
    visualisation.py

requirements.txt
README.md
LICENSE
```

---

# Installation

Clone the repository:

```
git clone https://github.com/yourusername/credit-risk-model-backtesting-basel.git
cd credit-risk-model-backtesting-basel
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Running the Project

Run the full pipeline:

```
python scripts/credit_risk_pipeline.py
```

The program automatically:

* Generates data
* Trains models
* Evaluates performance
* Computes risk metrics
* Produces figures
* Saves results

---

# Technologies Used

* Python 3.12
* NumPy
* Pandas
* SciPy
* Scikit-Learn
* Imbalanced-Learn (SMOTE)
* SHAP
* Matplotlib
* Joblib

---

# Research & Industry Relevance

Credit risk modeling is critical in modern banking risk management.

This project demonstrates:

* Probability of Default modeling
* Loss estimation
* Monte Carlo risk simulation
* Stress testing
* Model validation
* Portfolio Value-at-Risk estimation

The framework reflects methodologies commonly used in **Basel regulatory capital modeling and financial risk management**.
