""" This file plots the decision boundaries of the global and clients models."""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob
import joblib
from task import (
    BinaryClassifier, 
    SmallNN, 
    DecisionTree
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Get the paths that we need
base_dir = os.path.dirname(os.path.abspath(__file__))

# Points in 2D space that will be used to build the image
x1 = np.linspace(-10, 10, 300)
x2 = np.linspace(-10, 10, 300)
xx1, xx2 = np.meshgrid(x1, x2)
grid = torch.tensor(np.c_[xx1.ravel(), xx2.ravel()], dtype=torch.float32)

def load_any_client_model(path):
    """Load and return the client model stored in 'path'. """
    if path.endswith('.pkl'):
        return joblib.load(path)
    checkpoint = torch.load(path, map_location='cpu')
    model_type = checkpoint['model_type']
    if model_type == 'BinaryClassifier':
        model = BinaryClassifier()
    elif model_type == 'SmallNN':
        model = SmallNN()
    model.load_state_dict(checkpoint['model'])
    return model

def get_predictions(model):
    "Get predictions over the points in the grid"
    if type(model).__name__ == 'DecisionTree':
        probs = model.predict_proba(grid)
        return probs.numpy().reshape(xx1.shape)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(grid)).numpy().reshape(xx1.shape)

# FIRST PLOT: global model
global_model = SmallNN()
checkpoint = torch.load(os.path.join(base_dir, "global_model.pt"), map_location='cpu')
global_model.load_state_dict(checkpoint["model"])

fig, ax = plt.subplots(figsize=(7, 5))
preds = get_predictions(global_model)
ax.contourf(xx1, xx2, preds, levels=50, cmap="RdBu", alpha=0.8)
ax.contour(xx1, xx2, preds, levels=[0.5], colors="black", linewidths=2)
ax.plot([-10, 10], [-10, 10], "k--", linewidth=1, label="True boundary (x2=x1)")
ax.set_title("Global Model")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.legend()
plt.tight_layout()
plt.show()

# SECOND PLOT: client models
client_model_files = glob.glob(os.path.join(base_dir, "client_models", "*.pt")) + \
                     glob.glob(os.path.join(base_dir, "client_models", "*.pkl"))
n = len(client_model_files)

if n == 0:
    print("No client models found.")
else:
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2))
    axes = axes.flatten() if n > 1 else [axes]

    for i, path in enumerate(sorted(client_model_files)):
        model = load_any_client_model(path)
        model_type = type(model).__name__
        model = load_any_client_model(path)

        preds = get_predictions(model)
        axes[i].contourf(xx1, xx2, preds, levels=50, cmap="RdBu", alpha=0.8)
        axes[i].contour(xx1, xx2, preds, levels=[0.5], colors="black", linewidths=2)
        axes[i].plot([-10, 10], [-10, 10], "k--", linewidth=1)
        if model_type == 'BinaryClassifier':
            axes[i].set_title(f"Client {i} ['LogisticRegressor']", fontsize=7)
        else:
            axes[i].set_title(f"Client {i} [{model_type}]", fontsize=7)
        axes[i].set_xlabel("x1")
        axes[i].set_ylabel("x2")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Client Models Decision Boundaries", fontsize=14)
    plt.tight_layout()
    plt.show()