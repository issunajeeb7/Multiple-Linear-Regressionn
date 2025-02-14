# Car Fuel Efficiency Prediction

## Overview
This Jupyter notebook implements a Multiple Linear Regression model to predict city MPG (Miles Per Gallon) for cars using various vehicle specifications.

## Dependencies
```python
pandas
seaborn
matplotlib
scikit-learn
```

## Data Processing Pipeline

### 1. Data Loading
- Loads car dataset from `cars.csv`
- Contains vehicle specifications and fuel efficiency metrics

### 2. Feature Selection
#### Numerical Features:
- Vehicle Dimensions (Height, Length, Width)
- Number of Forward Gears
- Engine Statistics (Horsepower, Torque)

#### Categorical Features:
- Engine Driveline
- Transmission Type
- Fuel Type

### 3. Data Preprocessing
- Correlation analysis with visualization
- One-hot encoding for categorical variables
- Feature matrix preparation

### 4. Model Development
- Train-test split (80-20 ratio)
- Model: Multiple Linear Regression
- Target Variable: City MPG

## Code Structure

```python
# Data Loading and Preprocessing
file_path = "cars.csv"
df = pd.read_csv(file_path)

# Feature Engineering
numerical_features = [
    "Dimensions.Height", "Dimensions.Length", "Dimensions.Width",
    "Engine Information.Number of Forward Gears",
    "Engine Information.Engine Statistics.Horsepower",
    "Engine Information.Engine Statistics.Torque"
]

categorical_features = [
    "Engine Information.Driveline",
    "Engine Information.Transmission",
    "Fuel Information.Fuel Type"
]

# Model Training and Evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
```

## Visualization
- Correlation Matrix Heatmap
- Feature importance analysis
- Data distribution plots

## Model Evaluation
- Performance metric: Mean Squared Error (MSE)
- Test set predictions

## Usage
1. Ensure all dependencies are installed
2. Place `cars.csv` in the working directory
3. Run all cells in the notebook sequentially

## Future Improvements
1. Feature scaling implementation
2. Cross-validation
3. Hyperparameter tuning
4. Additional evaluation metrics (R², MAE)
5. Residual analysis

## License
Open source - Free to use and modify
