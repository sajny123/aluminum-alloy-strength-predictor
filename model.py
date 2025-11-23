import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = np.load('processed_data.npz')
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test'] 

scaler_y = joblib.load('scaler_y.pkl')

def evaluate_model(model, X_test, y_test_scaled, scaler):
    y_pred_scaled = model.predict(X_test)
    
    y_pred_real = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_test_real = scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()
    
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    r2 = r2_score(y_test_real, y_pred_real)
    
    print(f"R2 Score: {r2:.4f}")
    
    residuals = y_test_real - y_pred_real
    
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred_real, residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted Tensile Strength (MPa)")
    plt.ylabel("Residuals (Error in MPa)")
    plt.title("Residual Plot: Look for Random Scatter")
    plt.show()

# Apply linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

print('Linear Regression Evaluation')
evaluate_model(model, X_test, y_test, scaler_y)

# Apply LASSO model
model = Lasso(alpha=0.01, max_iter=50000, random_state=42)
model.fit(X_train, y_train)

print('LASSO evaluation')
evaluate_model(model, X_test, y_test, scaler_y)

# poly = joblib.load('poly_features.pkl')
# feature_names = poly.get_feature_names_out(X_train.columns)
# coeffs = model.coef_
# df_coeffs = pd.DataFrame({
#     'Feature': feature_names, 
#     'Coefficients': coeffs
# })

# df_active = df_coeffs[df_coeffs['Coefficient'] != 0].copy()
# df_active['Abs_Coeff'] = df_active['Coefficient'].abs()
# df_active = df_active.sort_values(by='Abs_Coeff', ascending=False)

# plt.figure(figsize=(10, 6))
# plt.barh(df_active['Feature'].head(10), df_active['Coefficient'].head(10))
# plt.xlabel("Impact on Strength (Scaled Coefficient)")
# plt.title("What is actually driving your Model?")
# plt.axvline(0, color='black', linewidth=0.8)
# plt.grid(axis='x', linestyle='--', alpha=0.7)
# plt.gca().invert_yaxis()
# plt.show()


# Apply Gaussian Process Regression 
kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)

model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42, normalize_y=False)
model.fit(X_train, y_train)

print('GPR Evaluation')
evaluate_model(model, X_test, y_test, scaler_y)

# Apply Support Vector Regression
model = SVR(kernel='rbf', C=25, epsilon=0.1)
model.fit(X_train, y_train)

print('SVR Evaluation')
evaluate_model(model, X_test, y_test, scaler_y)

# Validate models are not overfitting
print(f"{'Model':<20} | {'Train R2':<10} | {'Test R2':<10} | {'Gap':<10}")
print("-" * 55)

models = {'Linear Regression': LinearRegression(), 'Lasso': Lasso(alpha=0.01, max_iter=50000, random_state=42), 'GPR': GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42, normalize_y=False), 'SVR': SVR(kernel='rbf', C=25, epsilon=0.1)}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_train_pred_scaled = model.predict(X_train)
    y_test_pred_scaled = model.predict(X_test)

    y_train_real_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()
    y_test_real_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).ravel()
    
    y_train_real = scaler_y.inverse_transform(y_train.reshape(-1, 1)).ravel()
    y_test_real = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

    train_r2 = r2_score(y_train_real, y_train_real_pred)
    test_r2 = r2_score(y_test_real, y_test_real_pred)
    gap = train_r2 - test_r2

    print(f"{name:<20} | {train_r2:.4f}     | {test_r2:.4f}     | {gap:.4f}")