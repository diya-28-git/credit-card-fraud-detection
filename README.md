# 💳 Credit Card Fraud Detection Using Machine Learning

## Overview

Developed a machine learning-based fraud detection system to identify fraudulent credit card transactions using multiple classification and anomaly detection techniques.

## Technologies Used

* Python
* Scikit-learn
* XGBoost
* Optuna
* Pandas
* NumPy
* Matplotlib
* Seaborn

## Features

* Data Preprocessing and Feature Scaling
* Class Imbalance Handling using SMOTETomek
* Hyperparameter Tuning using Optuna
* Threshold Optimization using Precision-Recall Curves
* Model Evaluation using ROC-AUC, Precision, Recall and F1-Score
* Comparative Analysis of Multiple Models

## Models Implemented

* Logistic Regression
* Random Forest
* XGBoost
* Isolation Forest

## Results

* Compared model performance using ROC-AUC, Average Precision and F1-Score.
* Optimized classification thresholds to improve fraud detection performance.
* Generated confusion matrices, ROC curves, Precision-Recall curves and threshold analysis visualizations.

## Project Structure

Credit-Card-Fraud-Detection-ML/

├── fraud_detection_v2.py

├── predict_v2.py

├── requirements.txt

├── README.md

└── outputs/

  ├── confusion_matrices.png

  ├── roc_curves.png

  ├── pr_curves.png

  ├── model_comparison.png

  └── threshold_curves.png

## How to Run

pip install -r requirements.txt

python fraud_detection_v2.py

python predict_v2.py



