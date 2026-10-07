# Credit Card Fraud Detection using KNN

## Project Overview

This project implements a K-Nearest Neighbors (KNN) classification model to detect fraudulent credit card transactions.

Credit card fraud detection is a binary classification problem:

- `0` — Normal Transaction
- `1` — Fraudulent Transaction

The dataset is highly imbalanced, with fraudulent transactions representing a very small proportion of the total transactions. SMOTE (Synthetic Minority Over-sampling Technique) was applied to the training data to address class imbalance.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- Joblib
- Jupyter Notebook / Google Colab

## Machine Learning Workflow

1. Load the dataset
2. Perform data inspection and exploratory data analysis
3. Check for missing values and duplicate records
4. Separate features and target variable
5. Perform stratified train-test split
6. Scale features using StandardScaler
7. Handle class imbalance using SMOTE
8. Train the KNN classifier
9. Generate predictions
10. Evaluate model performance

## Model

The project uses the K-Nearest Neighbors classifier:

```python
KNeighborsClassifier
