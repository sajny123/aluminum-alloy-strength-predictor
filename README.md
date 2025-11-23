# Aluminum Alloy Strength Predictor

This project predicts the tensile strength of aluminum alloys using machine learning.

## Files

-   Al_alloys.csv, Al_alloys_cleaned.csv: Datasets (ignored by git)
-   cleaning.py, preprocessing.py: Data cleaning and preprocessing scripts
-   model.py: Model training and prediction
-   visualization.py: Data visualization
-   poly_features.pkl, scaler_y.pkl, processed_data.npz: Model artifacts

## Methodology

1. Preprocessing Pipeline

Data Cleaning: Removed redundant identifiers (e.g., 'Grade') to prevent multicollinearity.

Feature Engineering: Applied PolynomialFeatures(degree=2) to capture non-linear relationships (e.g., the quadratic decay of strength vs. temperature) and chemical interactions (e.g., $Mg \times Si$).

Scaling: Applied StandardScaler to both Inputs ($X$) and Target ($y$) to ensure stability for distance-based models (SVR/GPR).

2. Models Benchmarked

Multiple Linear Regression (Baseline)

Lasso Regression (L1 Regularization)

Support Vector Regression (SVR - RBF Kernel)

Gaussian Process Regression (GPR - Matern/RBF Kernel)
