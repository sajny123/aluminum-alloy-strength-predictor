import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv("Al_alloys_cleaned.csv")

# Create correlation matrix
correlation_matrix = df.corr()

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(method='spearman'), annot=True,  cmap='coolwarm', center=0)
plt.show()


X = df.drop(columns=['TensileStrength'])
y = df['TensileStrength']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 1. Setup Data (Use your scaled/encoded X and y)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 2. Get Importance
importances = model.feature_importances_
feature_names = X_train.columns

# 3. Create a DataFrame for plotting
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# 4. Plot
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'][:10], feature_importance_df['Importance'][:10])
plt.gca().invert_yaxis()  # Best feature at top
plt.title("Top Features (According to Random Forest)")
plt.show()

# Create the plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['ServiceTemp'], y=df['TensileStrength'], s=60, alpha=0.6)

# Add labels
plt.title("Dominant Feature Check: Service Temp vs. Strength", fontsize=14)
plt.xlabel("Service Temperature", fontsize=12)
plt.ylabel("Tensile Strength", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Show it
plt.show()