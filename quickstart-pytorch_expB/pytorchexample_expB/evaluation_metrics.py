"""Script for evaluation analysis: 
   Includes: Fidelity histogram
   Test error histogram"""
import os
import sys
import glob
import torch
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import joblib
from task import (
    BinaryClassifier, 
    SmallNN, 
    DecisionTree, 
    predict
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Get current file path
base_dir = os.path.dirname(os.path.abspath(__file__))

# Get the paths with global and client models
models_dir = os.path.join(base_dir, 'client_models')
global_model_path = os.path.join(base_dir, 'global_model.pt')

def get_data(data_file):
    """This function gets the data from the csv."""

    df = pd.read_csv(data_file)
    x = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
    labels = torch.tensor(df['label'].values, dtype=torch.float32)
    return x, labels

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

def get_global_model():
    """Load and return the global model."""
    global_model = SmallNN()
    checkpoint = torch.load(global_model_path, map_location='cpu')
    global_model.load_state_dict(checkpoint['model'])
    return global_model

def get_preds(model, data_batch):
    """Get binary predictions (hard labels) from any model type."""

    if type(model).__name__ == 'DecisionTree':
        probs = model.predict_proba(data_batch)
        return (probs >= 0.5).float()
    else:
        preds, _ = predict(model, data_batch)
        return preds.squeeze(1)


def get_fidelities(data_batch):
    """Get the global model - client fidelity for each client."""

    global_model = get_global_model()
    global_preds, _ = predict(global_model, data_batch)
    global_preds = global_preds.squeeze(1).numpy()

    client_paths = glob.glob(os.path.join(models_dir, 'model_*.pt')) + \
                   glob.glob(os.path.join(models_dir, 'model_*.pkl'))
    fidelities = []
    for path in client_paths:
        client_model = load_any_client_model(path)
        client_preds = get_preds(client_model, data_batch).numpy()
        fidelity = (client_preds.round() == global_preds).mean()
        fidelities.append(fidelity)
    return np.array(fidelities)

def get_global_fidelity(data_batch):
    """Get average fidelity of the global model."""

    global_model = get_global_model()
    global_preds, _ = predict(global_model, data_batch)
    global_preds = global_preds.squeeze(1).numpy()

    client_paths = glob.glob(os.path.join(models_dir, 'model_*.pt')) + \
                   glob.glob(os.path.join(models_dir, 'model_*.pkl'))
    fidelities_per_client = []
    for path in client_paths:
        client_model = load_any_client_model(path)
        client_preds = get_preds(client_model, data_batch).numpy()
        agreement = (client_preds.round() == global_preds).mean()
        fidelities_per_client.append(agreement)
    return np.mean(fidelities_per_client)

def get_test_errors(data_batch, true_labels):
    """Get the test error of each client with respect to the true labels."""

    client_paths = glob.glob(os.path.join(models_dir, 'model_*.pt')) + \
                   glob.glob(os.path.join(models_dir, 'model_*.pkl'))
    test_errors = []
    for path in client_paths:
        client_model = load_any_client_model(path)
        client_preds = get_preds(client_model, data_batch)
        error = (client_preds != true_labels).float().mean().item()
        test_errors.append(error)
    return np.array(test_errors)

def get_global_test_error(data_batch, true_labels):
    """Get the test error of the global model against true labels."""

    global_model = get_global_model()
    global_preds, _ = predict(global_model, data_batch)
    global_preds = global_preds.squeeze(1)
    error = (global_preds != true_labels).float().mean().item()
    return error

def fidelity_histogram(array_fidelities, global_fidelity):
    """It creates the fidelity histogram with a vertical line for the 
    global model fidelity.
    """

    plt.figure(figsize=(5, 4))
    bins = np.linspace(0.45, 1, 12)
    plt.rcParams['axes.axisbelow'] = True
    plt.hist(array_fidelities, bins=bins, edgecolor='black', color='steelblue')
    plt.axvline(x=global_fidelity, color='red', linestyle='--', linewidth=2,
                label=f'Global model fidelity: {global_fidelity:.3f}')
    plt.xticks(np.arange(0.45, 1.01, 0.05))
    plt.yticks(np.arange(0.00, 8, 1))
    plt.xlabel("Fidelity")
    plt.ylabel("Count")
    plt.title("Client-Global Model Fidelity")
    plt.legend()
    plt.grid(True, linewidth=0.5)
    plt.tight_layout()
    plt.show()

def test_error_histogram(array_test_errors, global_test_error):
    """It creates the test error histogram with a vertical line for the global
     model test error.
     """

    plt.figure(figsize=(5, 4))
    bins = np.linspace(0, 0.55, 12)
    plt.rcParams['axes.axisbelow'] = True
    plt.hist(array_test_errors, bins=bins, edgecolor='black', color='salmon')
    plt.axvline(x=global_test_error, color='red', linestyle='--', linewidth=2,
                label=f'Global model test error: {global_test_error:.3f}')
    plt.xticks(np.arange(0.00, 0.56, 0.05))
    plt.yticks(np.arange(0.00, 8, 1))
    plt.xlabel("Test Error")
    plt.ylabel("Count")
    plt.title("Client Test Error")
    plt.legend()
    plt.grid(True, linewidth=0.5)
    plt.tight_layout()
    plt.show()


# Code that generated the histograms
if __name__ == "__main__":
    x_fi, labels_fi = get_data(os.path.join(base_dir, 'data_evalmetrics_fi.csv'))
    x_te, labels_te = get_data(os.path.join(base_dir, 'data_evalmetrics_te.csv'))


    fidelities = get_fidelities(x_fi)
    global_fidelity = get_global_fidelity(x_fi)
    print(f"Fidelities: {fidelities}")
    print(f"Mean client fidelity: {fidelities.mean():.3f}")
    print(f"Global model fidelity: {global_fidelity:.3f}")
    fidelity_histogram(fidelities, global_fidelity)

    test_errors = get_test_errors(x_te, labels_te)
    global_test_error = get_global_test_error(x_te, labels_te)
    print(f"Test errors: {test_errors}")
    print(f"Mean client test error: {test_errors.mean():.3f}")
    print(f"Global model test error: {global_test_error:.3f}")
    test_error_histogram(test_errors, global_test_error)


# Uncomment to show the fidelity and test error together with each model type
"""
client_paths = glob.glob(os.path.join(models_dir, 'model_*.pt')) + \
               glob.glob(os.path.join(models_dir, 'model_*.pkl'))

global_model = get_global_model()
global_preds, _ = predict(global_model, x_fi)
global_preds = global_preds.squeeze(1).numpy()

for path in client_paths:
    model = load_any_client_model(path)
    model_type = type(model).__name__
    
    client_preds_fi = get_preds(model, x_fi).numpy()
    fidelity = (client_preds_fi.round() == global_preds).mean()
    
    client_preds_te = get_preds(model, x_te)
    test_error = (client_preds_te != labels_te).float().mean().item()
    
    print(f"{os.path.basename(path)} [{model_type}]: fidelity = {fidelity:.3f}, test_error = {test_error:.3f}")"""
