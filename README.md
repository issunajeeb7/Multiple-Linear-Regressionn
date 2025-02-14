Car Fuel Efficiency Prediction Using Multiple Linear Regression
This project implements a Multiple Linear Regression model to predict city MPG (Miles Per Gallon) for cars based on various features like dimensions, engine specifications, and other characteristics.

Dataset
The model uses the cars.csv dataset which contains various car specifications including:

Dimensional features (height, length, width)
Engine specifications (horsepower, torque, number of gears)
Categorical information (driveline, transmission type, fuel type)
Target variable: City MPG
Analysis Pipeline
The project consists of several analysis steps:

Data Preprocessing

Loading and cleaning the dataset
Feature selection based on correlation analysis
One-hot encoding of categorical variables
Feature Engineering

Numerical features selection
Categorical features encoding
Correlation analysis with heatmap visualization
Model Development

Train-test split (80-20)
Multiple Linear Regression implementation
Model evaluation using MSE (Mean Squared Error)
Visualization

Correlation matrix (correlation_matrix.png)
Feature scatter plots (feature_scatter_plots.png)
Feature importance analysis (feature_importance.png)
Residual analysis (residual_plot.png)
Target distribution (target_distribution.png)
Files Description
MLR.py: Main implementation of Multiple Linear Regression
MultipleLinearRegression.ipynb: Jupyter notebook with step-by-step analysis
compare_models.py: Script to compare different model configurations
