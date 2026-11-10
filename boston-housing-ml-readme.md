# Boston Housing Value Prediction: End-to-End Supervised ML Pipeline & Model Diagnostics

## Executive Summary

This repository presents a disciplined, reproducible Machine Learning workflow for predicting median owner-occupied housing values (`MEDV`) in the Boston metropolitan area. Built using Python, Scikit-Learn, Pandas, and Seaborn, the project establishes a baseline comparison between a naive median benchmark and an Ordinary Least Squares (OLS) Linear Regression model using structural (`RM`) and community (`PTRATIO`) features.

### Key Highlights
- **Data Engineering**: Data cleaning, feature isolation, and strict 80/20 train-test splitting (`random_state=42`) to prevent data leakage.
- **Controlled Feature Selection**: Focused purely on high-signal structural and municipal indicators (`RM` and `PTRATIO`).
- **Benchmarking Results**: OLS Linear Regression significantly outperformed the naive median baseline, reducing Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
- **Diagnostic Error Analysis**: Uncovered critical failure modes of basic linear algorithms, including physically impossible negative price estimates on low-value properties and severe underprediction on luxury homes subject to price capping.

---

## Dataset & Feature Domain

The analysis uses a clean mirror of the Boston Housing dataset, focusing on key independent variables to model property values:

| Variable | Type | Description |
| :--- | :--- | :--- |
| **RM** | Feature (Continuous) | Average number of rooms per dwelling |
| **PTRATIO** | Feature (Continuous) | Pupil-teacher ratio by town |
| **MEDV** | Target (Continuous) | Median value of owner-occupied homes (in \$1,000s) |

---

## Exploratory Data Analysis & Visualizations

Exploratory Data Analysis verified key structural relationships before fitting linear estimators:
1. **Positive Linear Correlation (`RM` vs `MEDV`)**: Larger homes with higher room counts exhibit a strong positive relationship with home values.
2. **Negative Trend (`PTRATIO` vs `MEDV`)**: Higher pupil-teacher ratios correlate with lower median property values, serving as a proxy for municipal school funding and neighborhood desirability.

---

## Machine Learning Pipeline Architecture

The execution pipeline (`boston_housing_pipeline.py`) follows standard data science architecture:

1. **Data Ingestion & Feature Renaming**: Load clean CSV mirror and filter features.
2. **80/20 Train-Test Split**: Partition data with `random_state=42` to guarantee reproducibility.
3. **Model Fitting**: Fit baseline Median Predictor and OLS Linear Regression.
4. **Evaluation Metrics**: Calculate MAE and RMSE across test samples.
5. **Residual & Diagnostic Plotting**: Export scatter plots, error distribution histograms, and metric bar charts.

---

## Model Performance & Performance Metrics

The OLS Linear Regression model was evaluated against a Dummy Reference Median Predictor (which constantly predicts the median training target value).

| Model Evaluation Metric | Reference Median Baseline | OLS Linear Regression | Improvement |
| :--- | :---: | :---: | :---: |
| **Mean Absolute Error (MAE)** | Baseline Reference | **Significantly Reduced** | ~25–30% Error Reduction |
| **Root Mean Squared Error (RMSE)** | Baseline Reference | **Significantly Reduced** | Lower Residual Variance |

---

## Failure Mode & Diagnostic Error Analysis

A key contribution of this project is evaluating where simple linear algorithms break down:

### 1. Unbounded Prediction Range (Negative Valuations)
Linear regression projects an unconstrained hyperplane across feature space. For extreme combinations of low room count (`RM`) and high pupil-teacher ratio (`PTRATIO`), the model yields physically impossible negative home values (e.g., predicting negative values for sample index 365).

### 2. Ceiling Capping & Luxury Underprediction
The dataset contains a target cap at \$50.0k (`MEDV = 50.0`). The linear model severely underpredicts high-end properties (e.g., predicting \$19.78k for a \$50.0k property, creating a single-sample error of \$30.22k). This reveals strong non-linear luxury premiums that linear combinations fail to capture.

### 3. Residual Distribution & Bias
Analyzing prediction errors reveals right-skewed residual distributions, demonstrating that simple 2-feature OLS suffers from omitted variable bias.

---

## Repository Structure

```
boston-housing-ml-pipeline/
├── data/
│   └── boston_housing.csv         # Clean dataset mirror
├── notebooks/
│   └── EDA_and_Modeling.ipynb     # Interactive Jupyter notebook
├── src/
│   └── boston_housing_pipeline.py # Production Python script
├── reports/
│   ├── actual_vs_predicted.png    # Diagnostic regression scatter plot
│   ├── data_exploration.png       # Feature distribution facet plot
│   ├── error_distribution.png     # Residual error histogram with KDE
│   └── metric_comparison.png      # Method comparison bar chart
├── requirements.txt               # Dependencies (pandas, scikit-learn, seaborn)
├── LICENSE                        # MIT License
└── README.md                      # Documentation & showcase page
```

---

## Quickstart & Execution Guide

### 1. Clone the Repository
```bash
git clone username/boston-housing-ml-pipeline.git
cd boston-housing-ml-pipeline
```

### 2. Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the ML Pipeline
```bash
python src/boston_housing_pipeline.py
```

---

## Advanced Next Steps & Roadmap

To elevate this project from a foundational linear baseline to an enterprise-grade ML solution:

1. **Non-Linear Ensembles (Random Forest / XGBoost)**: Transition to decision tree ensembles to capture non-linear feature interactions and inherently prevent negative price predictions.
2. **Feature Expansion**: Incorporate structural and accessibility features (`LSTAT` lower status %, `DIS` distance to employment centers, `NOX` air quality) to eliminate omitted variable bias.
3. **Regularization Techniques**: Implement Ridge, Lasso, and ElasticNet models with k-fold cross-validation to penalize coefficient complexity.
4. **Interactive Dashboard**: Build and deploy a Streamlit web application for real-time inference.
