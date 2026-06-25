""" This file generates two data files for training client models 
and evaluating the accuracy of the global model."""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split with stratify to preserve class balance across all splits
X_clients, X_evalmetrics_te, y_clients, y_evalmetrics_te = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

feature_cols = [f"feature_{i}" for i in range(X_scaled.shape[1])]

""" Generate the two csv: one for local training, 
the other for evaluation of copy accuracy.
"""
for name, X, y in [("data_clients", X_clients, y_clients),
                    ("data_evalmetrics_te", X_evalmetrics_te, y_evalmetrics_te)]:
    df = pd.DataFrame(X, columns=feature_cols)
    df["label"] = y
    df.to_csv(f"{name}.csv", index=False)
    print(f"{name}.csv: {len(df)} rows, class balance: {y.mean()*100:.1f}% class 1")

