Boston Housing Price Prediction & Diagnostic ML Pipeline
An end-to-end machine learning pipeline evaluating Ordinary Least Squares (OLS) Linear Regression against baseline predictors on Boston Housing data, featuring residual error diagnostics and non-linear failure mode analysis.

📌 Summary
This project implements a rigorous, end-to-end Supervised Machine Learning workflow in Python to predict median housing prices (MEDV) in the Boston metropolitan area. Using a clean subset of structural (RM — average rooms) and community (PTRATIO — pupil-teacher ratio) features, the pipeline contrasts an Ordinary Least Squares (OLS) Linear Regression model against a Median Reference Predictor across an 80/20 train-test split.
While OLS achieves substantial error reduction over baseline benchmarks, diagnostic residual analysis reveals critical model limitations—specifically negative price predictions at low feature extremes and underprediction of luxury property ceilings.

🎯 Key Findings & Diagnostic Insights

Baseline vs. OLS Error Reduction: OLS regression reduces Mean Absolute Error (MAE) from $4.45k (Median Baseline) to $3.88k, and Root Mean Squared Error (RMSE) from $6.44k to $5.66k.
Negative Price Anomaly: At Index 365, the linear model outputs -$1.68k, illustrating how linear extrapolation fails at feature boundaries without non-negativity constraints.
Luxury Ceiling Saturation: For high-end properties capped at $50.0k, OLS underpredicts by $30.22k (predicting $19.78k), proving that linear models cannot capture luxury market premiums without non-linear terms.


📊 Visual Diagnostic Dashboard
1. Data Exploration: Predictor Relationships
--Image of: --Feature vs Target Exploration
Linear regression trends showing strong positive correlation with RM and negative correlation with PTRATIO.
2. Model Evaluation: Actual vs. Predicted Prices
--Image of: --Actual vs Predicted Values
Scatter plot comparing test predictions against ideal ground truth parity (red dashed line). Note the residual gap at $50k actual value.
3. Residual Error Distribution
--Image of: --Residual Error Histogram
Distribution of prediction errors (Actual - Predicted). The mean error centers near zero, but positive skew highlights extreme underprediction outliers.
4. Method Performance Comparison
--Image of: --MAE vs RMSE Model Comparison
Side-by-side performance metrics showing MAE and RMSE improvements of OLS over Baseline.

📈 Performance Summary Table
Evaluation ModelStrategy / AlgorithmTraining StrategyTest MAE ($1,000s)Test RMSE ($1,000s)Performance GainsReference BaselineTraining Set Median ($21.20k)Constant Predictor$4.45$6.44Benchmark BaselineFitted ModelOrdinary Least Squares (OLS)80/20 Split (seed=42)$3.88$5.6612.8% MAE Reduction

🛠️ Data Pipeline Architecture
boston-housing-ml-pipeline/
├── data/
│   └── BostonHousing.csv            # Clean dataset mirror
├── notebooks/
│   └── EDA_and_Modeling.ipynb       # Interactive analysis & visualizations
├── reports/
│   ├── data_exploration.png         # Feature distribution plots
│   ├── actual_vs_predicted.png      # OLS parity scatter plot
│   ├── error_distribution.png       # Residual error histogram & KDE
│   └── metric_comparison.png        # Bar chart metric comparison
├── src/
│   └── boston_housing_pipeline.py   # Full executable Python script
├── requirements.txt                 # Project dependencies
├── LICENSE                          # MIT License
└── README.md                        # Documentation


📐 Mathematical Model Formulation
Linear Regression Equation
$$\text{MEDV} = \beta_0 + \beta_1 (\text{RM}) + \beta_2 (\text{PTRATIO}) + \epsilon$$
Fitted Model Coefficients

Intercept ($\beta_0$): $+7.28$
Room Count ($\beta_1$): $+5.52$ ($+\$5,520$ per additional room)
Pupil-Teacher Ratio ($\beta_2$): $-0.59$ ($-\$590$ per unit increase in class size)

$$\widehat{\text{MEDV}} = 7.28 + 5.52(\text{RM}) - 0.59(\text{PTRATIO})$$

🚀 Quickstart & Setup Guide
1. Clone Repository
git clone git@github.com:your-username/boston-housing-ml-pipeline.git
cd boston-housing-ml-pipeline

2. Install Dependencies
pip install -r requirements.txt

3. Run Pipeline Script
python src/boston_housing_pipeline.py


🛡️ License
Distributed under the MIT License. See LICENSE for details.
