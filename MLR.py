import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Set style for better visualizations
plt.style.use('seaborn')
sns.set_palette("husl")

# Load the dataset
df = pd.read_csv('cars.csv')

# 1. Data Preprocessing
# Handle missing values
df = df.dropna()

# 2. Handle Categorical Variables
le = LabelEncoder()
categorical_columns = ['Engine Information.Driveline', 'Engine Information.Engine Type', 
                      'Engine Information.Transmission', 'Engine Information.Hybrid']
for col in categorical_columns:
    df[col + '_encoded'] = le.fit_transform(df[col])

# 3. Feature Selection
numerical_features = ['Dimensions.Height', 'Dimensions.Length', 'Dimensions.Width',
                     'Engine Information.Number of Forward Gears',
                     'Engine Information.Driveline_encoded',
                     'Engine Information.Transmission_encoded',
                     'Engine Information.Hybrid_encoded']

# 4. Create scatter plots for each feature
fig, axes = plt.subplots(3, 3, figsize=(20, 20))
axes = axes.ravel()

for idx, feature in enumerate(numerical_features):
    sns.regplot(data=df, x=feature, y='Fuel Information.City mpg', 
                ax=axes[idx], scatter_kws={'alpha':0.5}, line_kws={'color': 'red'})
    axes[idx].set_title(f'{feature} vs City MPG')

# Remove empty subplot if any
if len(numerical_features) < 9:
    for idx in range(len(numerical_features), 9):
        fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig('feature_scatter_plots.png')
plt.close()

# 5. Correlation Analysis
plt.figure(figsize=(12, 8))
correlation_matrix = df[numerical_features + ['Fuel Information.City mpg']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# 6. Distribution of Target Variable
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Fuel Information.City mpg', bins=30, kde=True)
plt.title('Distribution of City MPG')
plt.xlabel('City MPG')
plt.ylabel('Count')
plt.savefig('target_distribution.png')
plt.close()

# 7. Outlier Detection using Z-score
def detect_outliers(df, column):
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    return z_scores > 3

# Check outliers in numerical columns
outliers_summary = {}
for column in numerical_features:
    outliers = detect_outliers(df, column)
    outliers_summary[column] = sum(outliers)
print("\nOutliers Summary:")
for column, count in outliers_summary.items():
    print(f"{column}: {count} outliers")

# 8. Calculate VIF
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

# Prepare features for VIF calculation
X = df[numerical_features]
vif_results = calculate_vif(X)
print("\nVariance Inflation Factors:")
print(vif_results)

# 9. Multiple Linear Regression
X = df[numerical_features]
y = df['Fuel Information.City mpg']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Model Evaluation
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print("\nModel Performance:")
print(f"R-squared Score (Training): {train_score:.4f}")
print(f"R-squared Score (Testing): {test_score:.4f}")

# 10. Actual vs Predicted Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_train, y_train_pred, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('Actual City MPG')
plt.ylabel('Predicted City MPG')
plt.title('Actual vs Predicted City MPG (Training Data)')
plt.tight_layout()
plt.savefig('training_regression_plot.png')
plt.close()

# Actual vs Predicted Plot for Test Data
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual City MPG')
plt.ylabel('Predicted City MPG')
plt.title('Actual vs Predicted City MPG (Test Data)')
plt.tight_layout()
plt.savefig('testing_regression_plot.png')
plt.close()

# 11. Residual Plot
residuals = y_test - y_test_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_test_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted City MPG')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.tight_layout()
plt.savefig('residual_plot.png')
plt.close()

# Feature Importance
feature_importance = pd.DataFrame({
    'Feature': numerical_features,
    'Coefficient': model.coef_
})
print("\nFeature Importance:")
print(feature_importance.sort_values(by='Coefficient', key=abs, ascending=False))

# Plot Feature Importance
plt.figure(figsize=(12, 6))
feature_importance_sorted = feature_importance.sort_values(by='Coefficient', key=abs, ascending=True)
plt.barh(feature_importance_sorted['Feature'], feature_importance_sorted['Coefficient'])
plt.title('Feature Importance')
plt.xlabel('Coefficient Value')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()