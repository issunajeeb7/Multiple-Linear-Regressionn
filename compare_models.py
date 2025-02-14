import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess data
dataset = pd.read_csv('cars.csv')

# Drop duplicates and handle missing values
dataset = dataset.drop_duplicates()
dataset = dataset.dropna()

# Select features
features = [
    'Engine Information.Engine Statistics.Torque',
    'Engine Information.Engine Statistics.Horsepower',
    'Dimensions.Height',
    'Dimensions.Length',
    'Fuel Information.City mpg'
]

df = dataset[features]

# Function to remove outliers using IQR method
def remove_outliers(df):
    df_clean = df.copy()
    for column in df_clean.columns:
        Q1 = df_clean[column].quantile(0.25)
        Q3 = df_clean[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)]
    return df_clean

# Prepare data with outliers
y_with = df['Fuel Information.City mpg']
X_with = df.drop(columns=['Fuel Information.City mpg'])

# Prepare data without outliers
df_no_outliers = remove_outliers(df)
y_without = df_no_outliers['Fuel Information.City mpg']
X_without = df_no_outliers.drop(columns=['Fuel Information.City mpg'])

# Scale the data
scaler_with = StandardScaler()
scaler_without = StandardScaler()

X_with_scaled = scaler_with.fit_transform(X_with)
X_without_scaled = scaler_without.fit_transform(X_without)

y_with = y_with.values.reshape(-1, 1)
y_without = y_without.values.reshape(-1, 1)

# Split the data
X_train_with, X_test_with, y_train_with, y_test_with = train_test_split(
    X_with_scaled, y_with, test_size=0.2, random_state=42
)

X_train_without, X_test_without, y_train_without, y_test_without = train_test_split(
    X_without_scaled, y_without, test_size=0.2, random_state=42
)

# Train models
model_with = LinearRegression()
model_without = LinearRegression()

model_with.fit(X_train_with, y_train_with)
model_without.fit(X_train_without, y_train_without)

# Make predictions
y_pred_train_with = model_with.predict(X_train_with)
y_pred_test_with = model_with.predict(X_test_with)
y_pred_train_without = model_without.predict(X_train_without)
y_pred_test_without = model_without.predict(X_test_without)

# Calculate metrics
metrics = {
    'With Outliers': {
        'Train R2': r2_score(y_train_with, y_pred_train_with),
        'Test R2': r2_score(y_test_with, y_pred_test_with),
        'Train RMSE': np.sqrt(mean_squared_error(y_train_with, y_pred_train_with)),
        'Test RMSE': np.sqrt(mean_squared_error(y_test_with, y_pred_test_with))
    },
    'Without Outliers': {
        'Train R2': r2_score(y_train_without, y_pred_train_without),
        'Test R2': r2_score(y_test_without, y_pred_test_without),
        'Train RMSE': np.sqrt(mean_squared_error(y_train_without, y_pred_train_without)),
        'Test RMSE': np.sqrt(mean_squared_error(y_test_without, y_pred_test_without))
    }
}

# Print results
print("\nModel Comparison:")
print("=" * 50)
for model_type, scores in metrics.items():
    print(f"\n{model_type}:")
    print(f"Training R² Score: {scores['Train R2']:.4f}")
    print(f"Testing R² Score: {scores['Test R2']:.4f}")
    print(f"Training RMSE: {scores['Train RMSE']:.4f}")
    print(f"Testing RMSE: {scores['Test RMSE']:.4f}")

# Compare feature importance
feature_importance_with = pd.DataFrame({
    'Feature': X_with.columns,
    'Coefficient': model_with.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)

feature_importance_without = pd.DataFrame({
    'Feature': X_without.columns,
    'Coefficient': model_without.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)

print("\nFeature Importance Comparison:")
print("=" * 50)
print("\nWith Outliers:")
print(feature_importance_with)
print("\nWithout Outliers:")
print(feature_importance_without)

# Create visualization
plt.figure(figsize=(12, 6))

# Actual vs Predicted plot for both models
plt.subplot(1, 2, 1)
plt.scatter(y_test_with, y_pred_test_with, alpha=0.5, label='With Outliers')
plt.plot([y_test_with.min(), y_test_with.max()], [y_test_with.min(), y_test_with.max()], 'r--', lw=2)
plt.xlabel('Actual MPG')
plt.ylabel('Predicted MPG')
plt.title('With Outliers\nActual vs Predicted')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_test_without, y_pred_test_without, alpha=0.5, label='Without Outliers')
plt.plot([y_test_without.min(), y_test_without.max()], [y_test_without.min(), y_test_without.max()], 'r--', lw=2)
plt.xlabel('Actual MPG')
plt.ylabel('Predicted MPG')
plt.title('Without Outliers\nActual vs Predicted')
plt.legend()

plt.tight_layout()
plt.show()
