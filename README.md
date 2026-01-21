# Healthcare Insurance Cost Prediction using Neural Networks

## Overview
This project predicts individual healthcare insurance expenses using a feed-forward neural network.  
The model learns from demographic and lifestyle factors such as age, BMI, smoking status, and region.

The project demonstrates a complete applied machine learning workflow:
- data preprocessing
- feature engineering
- neural network modeling
- regression evaluation
- visualization of predictions


## Problem Statement
Healthcare insurance costs vary widely across individuals. Accurately estimating medical expenses helps:
- insurance companies assess risk
- policymakers understand cost drivers
- organizations design fair pricing models

This project formulates the problem as a **regression task**, predicting continuous insurance expenses.

## Dataset

The dataset is stored locally in:
`
data/insuarance.csv
`
Features:
- `age`
- `sex`
- `bmi`
- `children`
- `smoker`
- `region`

Target:
- `expenses` (annual medical cost)


## Methodology

### 1. Data Preprocessing
- Numerical features are standardized using `StandardScaler`
- Categorical features are encoded using `OneHotEncoder`
- `ColumnTransformer` ensures a clean and reproducible pipeline

### 2. Model Architecture

A fully connected neural network:
- Input layer based on processed feature size
- Two hidden layers (64 neurons, ReLU activation)
- Output layer for continuous regression

### 3. Training

- Optimizer: Adam
- Loss function: Mean Absolute Error (MAE)
- Epochs: 100
- Validation split: 20%

### 4. Evaluation

Model performance is evaluated using MAE on a held-out test set.


## Results

The scatter plot of **Actual vs Predicted Expenses** shows how closely the model approximates real insurance costs.

Lower dispersion around the diagonal indicates better predictive performance.


## How to Run

## 1. Install dependencies
```bash`
pip install -r requirements.txt
`
## 2. Run the model
python insurance_model.py

## 3. Run tests
python test.py

### Technologies Used

Python

TensorFlow / Keras

Pandas

NumPy

Scikit-learn

Matplotlib

### Notes

The project is designed to run identically on local machines, cloud platforms, and CI systems.

Model hyperparameters and visualization behavior are intentionally unchanged to preserve reproducibility with my original Colab output.

### Author

Muhammad Ibrahim Gimba
Machine Learning | Data Science | Business Analysis | Project Management | Remote-Ready
