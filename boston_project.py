# -*- coding: utf-8 -*-
"""
Spyder Editor

Boston Project - Complete Pipeline
"""
import sys
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Record library versions for your report/notebook
print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Scikit-Learn version: {sklearn.__version__}")

# Set random seed for reproducibility
RANDOM_SEED = 42

# 2. Load data from the clean archive mirror (bypassing OpenML bug)
# Load data from the clean archive mirror
df = pd.read_csv(
    "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
)

# Print columns to see exact names if needed, and map them to uppercase
print("Original columns:", df.columns.tolist())

# Rename to match your brief requirement ('RM', 'PTRATIO', 'MEDV')
df = df.rename(columns={
    'rm': 'RM', 
    'ptratio': 'PTRATIO', 
    'medv': 'MEDV'
})

# Keep only the required columns
df = df[['RM', 'PTRATIO', 'MEDV']]

print("\n--- Data Summary ---")
print("Data loaded successfully! Shape:", df.shape)
print(df.isnull().sum())
df = df.dropna()
print(df.head())

# 3. Inspect and Split the Data (80/20 Train/Test split)
X = df[['RM', 'PTRATIO']]
y = df['MEDV']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_SEED
)

print(f"\nTraining set shape: {X_train.shape}, Target mean: {y_train.mean():.2f}")
print(f"Test set shape: {X_test.shape}, Target mean: {y_test.mean():.2f}")

# 4. Implement Methods (Reference vs. Fitted Comparison Method)
# Reference Method: Training Median Predictor
median_train_val = y_train.median()
y_pred_ref = np.full_like(y_test, fill_value=median_train_val, dtype=float)

# Comparison Method: Ordinary Least Squares Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("\n--- Model Parameters ---")
print(f"Reference Median Value: {median_train_val:.2f}")
print("Linear Regression Coefficients:", lr_model.coef_)
print("Linear Regression Intercept:", lr_model.intercept_)

# 5. Evaluate Metrics (MAE and RMSE)
mae_ref = mean_absolute_error(y_test, y_pred_ref)
rmse_ref = np.sqrt(mean_squared_error(y_test, y_pred_ref))

mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

print("\n--- Evaluation Results ---")
print(f"Reference Model   -> MAE: {mae_ref:.2f}, RMSE: {rmse_ref:.2f}")
print(f"Linear Regression -> MAE: {mae_lr:.2f}, RMSE: {rmse_lr:.2f}")

# 6. Plot Actual vs. Predicted Values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.7, color='b', label='Linear Regression Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Fit')
plt.xlabel("Actual MEDV ($1,000s)")
plt.ylabel("Predicted MEDV ($1,000s)")
plt.title("Actual vs. Predicted Home Values (Linear Regression)")
plt.legend()
plt.grid(True)
plt.savefig("actual_vs_predicted.png", dpi=300, bbox_inches='tight')
plt.show()

# 7. Inspect Two Largest Absolute Errors for Discussion
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred_lr,
    'Absolute_Error': np.abs(y_test - y_pred_lr)
})
largest_errors = results_df.sort_values(by='Absolute_Error', ascending=False).head(2)
print("\n--- Two Largest Absolute Errors ---")
print(largest_errors)

#------------------EXTRAS--------------------
import seaborn as sns #Seaborn for aesthetic plotting

# Ensure we are using the set of predictions from Linear Regression 
# that generated results_df from the previous step.
# results_df already contains 'Actual' and 'Predicted' values.

# set plot style
sns.set_style("whitegrid")
context_font_scale = 1.2 # Scale fonts for better readability in presentation
sns.set_context("notebook", font_scale=context_font_scale)

# ==============================================================================
# Visualization 1: Feature vs. Target Distributions (Data Overview)
# Goal: To understand how the input features relate to price.
# ==============================================================================
# Prepare long-form data for faceted plotting
melted_X = X.melt(var_name='Feature', value_name='Value')
melted_X['MEDV'] = np.tile(y, X.shape[1]) # Add target column back for comparison

# Create the plot
g = sns.lmplot(
    data=melted_X, 
    x='Value', y='MEDV', 
    col='Feature', 
    facet_kws={'sharex': False}, # Updated to avoid the deprecation warning
    height=5, aspect=1.1,
    scatter_kws={'alpha':0.5, 'color':'teal'},
    line_kws={'color': 'darkred', 'lw': 3}
)
g.set_titles("{col_name} vs. MEDV")
g.set_axis_labels("Feature Value", "Median Value (MEDV, $1,000s)")
plt.subplots_adjust(top=0.85) # Make room for main title
g.fig.suptitle('Relationship Between Input Predictors and Home Value')
g.fig.savefig('data_exploration.png', dpi=300, bbox_inches='tight')
plt.show()


# ==============================================================================
# Visualization 2: Distribution of Prediction Errors (Residuals Histogram)
# Goal: To see if our errors are centered around zero or biased.
# ==============================================================================
plt.figure(figsize=(10, 6))

# Calculate the error (residuals)
residuals = results_df['Actual'] - results_df['Predicted']

# Plot residual histogram and KDE
sns.histplot(residuals, kde=True, bins=25, color='darkblue', edgecolor='white')

# Add context lines
plt.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Zero Error Line')
plt.axvline(x=np.mean(residuals), color='darkred', linestyle='-', linewidth=2, label=f'Mean Error ({np.mean(residuals):.2f})')

plt.title('Distribution of Prediction Errors (Linear Regression)', fontsize=16)
plt.xlabel('Error: Actual MEDV - Predicted MEDV ($1,000s)', fontsize=14)
plt.ylabel('Count (Number of Tows)', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig('error_distribution.png', dpi=300, bbox_inches='tight')
plt.show()


# ==============================================================================
# Visualization 3: Compact Comparison of Methods (Bar Chart)
# Goal: Provide the primary evidence for the report/presentation in one image.
# ==============================================================================
# Create a dedicated comparison dataframe
metric_comparison = pd.DataFrame({
    'Metric': ['MAE (Primary)', 'RMSE (Secondary)'] * 2,
    'Model': ['Reference Median'] * 2 + ['Linear Regression'] * 2,
    'Value': [mae_ref, rmse_ref, mae_lr, rmse_lr]
})

plt.figure(figsize=(10, 6))

# Use seaborn barplot for complex categorical grouping
chart = sns.barplot(
    data=metric_comparison, 
    x='Metric', y='Value', 
    hue='Model', 
    palette='muted',
    edgecolor='gray'
)

# Add value labels on top of the bars for clarity
for container in chart.containers:
    chart.bar_label(container, fmt='%.2f', fontsize=12, padding=3)

plt.title('Model Performance Comparison (MAE vs. RMSE)', fontsize=16)
plt.ylabel('Error Value ($1,000s) - Lower is Better', fontsize=14)
plt.xlabel('', fontsize=14)
plt.legend(title='Method', fontsize=12)
plt.tight_layout()
plt.savefig('metric_comparison.png', dpi=300, bbox_inches='tight')
plt.show()