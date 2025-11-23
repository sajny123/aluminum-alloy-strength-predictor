import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import joblib

df = pd.read_csv('Al_alloys_cleaned.csv')

# Separate into predictors and target
X = df.drop(columns=['TensileStrength'])
y = df['TensileStrength']

# Split data into testing and training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate square and interaction of every feature column
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Scale data for Lasso/SVR
scaler_X = StandardScaler()
X_train_final = scaler_X.fit_transform(X_train_poly)
X_test_final = scaler_X.transform(X_test_poly)

# Scale target for GPR/SVR
y_train_temp = y_train.values.reshape(-1, 1)
y_test_temp = y_test.values.reshape(-1, 1)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train_temp)
y_test_scaled = scaler_y.transform(y_test_temp)

y_train_final = y_train_scaled.ravel()
y_test_final = y_test_scaled.ravel()

print(f"X_train shape: {X_train_final.shape}")
print(f"y_train shape: {y_train_final.shape}")
print(f"X_test shape: {X_test_final.shape}")
print(f"y_test shape: {y_test_final.shape}")

# Save processed data
np.savez_compressed(
    'processed_data.npz', 
    X_train=X_train_final, 
    y_train=y_train_final, 
    X_test=X_test_final, 
    y_test=y_test_final
)

# Save y_scaler
joblib.dump(scaler_y, 'scaler_y.pkl')

# Save feautures
joblib.dump(poly, 'poly_features.pkl')
