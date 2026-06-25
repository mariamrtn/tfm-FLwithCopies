""" Some functions used in files:
- client_app.py
- custom_strategy.py
- server_app.py
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib

class BinaryClassifier(nn.Module):
    """Create the model Logistic Regressor (1 layer nn)"""
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.linear = nn.Linear(30, 1)
    
    def forward(self, x):
        return self.linear(x)

class SmallNN(nn.Module):
    """ Create a Small NN model"""
    def __init__(self):
        super(SmallNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(30, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1))

    def forward(self, x):
        return self.net(x)


class DecisionTree:
    """ Create a Decision Tree class"""
    def __init__(self):
        self.model = DecisionTreeClassifier(max_depth=5)
        self.is_fitted = False
    
    def fit(self, x, y):
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        if isinstance(y, torch.Tensor):
            y = y.numpy()
        self.model.fit(x, y.astype(int))
        self.is_fitted = True
    
    def predict_proba(self, x):
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        if not self.is_fitted:
            return torch.full((len(x),), 0.5)
        proba = self.model.predict_proba(x)
        if proba.shape[1] == 1:
            # only one class seen — return 0.5 for all (uncertain)
            return torch.full((len(x),), 0.5)
        probs = proba[:, 1]
        return torch.tensor(probs, dtype=torch.float32).round()

class XYDataset(Dataset):
    """Load the data and prepare for dataloader"""
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.features = self.data.iloc[:, :-1].values
        self.labels = self.data['label'].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y

def train_one_batch(model, optimizer, batch_data, targets):
    """Function to train Logistic Regressiors and NN with one batch of data"""
    model.train()
    # Criterion
    criterion = nn.BCEWithLogitsLoss()
    x = batch_data.float()
    y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
    optimizer.zero_grad()
    preds = model(x)
    loss = criterion(preds, y)
    loss.backward()
    optimizer.step()
    return loss.item()

def predict(model, x):
    """ Function to predict the label given a model and some data"""
    model.eval()
    with torch.no_grad():
        output = model(x)
        # Soft predictions
        probs = torch.sigmoid(output)
        # Hard predictions
        preds = (probs > 0.5).float()
    return preds, probs

def get_client_partition(x_all, y_all, partition_idx, partition_size, extra_fraction=3):
    """ Get the data corresponfing to each client"""

    total = len(x_all)

    # Base partition — client's own slice
    start = partition_idx * partition_size
    end = start + partition_size
    base_indices = list(range(start, end))
    
    # Extra data — sampled from the full dataset
    extra_size = int(partition_size * extra_fraction)
    rng = np.random.RandomState(partition_idx)
    extra_indices = rng.choice(total, size=extra_size, replace=True).tolist()
    
    # Combine
    all_indices = base_indices + extra_indices
    return x_all[all_indices], y_all[all_indices]

# Get path
models_dir = os.path.join(os.path.dirname(__file__), "client_models")
os.makedirs(models_dir, exist_ok=True)

def get_client_model_path(node_id):
    """Function for getting the path of a client's model"""
    model = get_model_for_node(node_id)
    ext = '.pkl' if isinstance(model, DecisionTree) else '.pt'
    return os.path.join(models_dir, f"model_{node_id}{ext}")

def get_model_for_node(node_id):
    """ Function that gets the type of model for each client"""
    if node_id % 3 == 0:
        return BinaryClassifier()
    elif node_id % 3 == 1:
        return SmallNN()
    else:
        return DecisionTree()

def save_client_model(node_id, model, optimizer=None):
    """Save the models of the clients"""
    path = get_client_model_path(node_id)
    if isinstance(model, DecisionTree):
        joblib.dump(model, path)
    else:
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_type": type(model).__name__
        }, path)

def load_client_model(node_id):
    """Load the model for each client"""
    model = get_model_for_node(node_id)
    path = get_client_model_path(node_id)
    if isinstance(model, DecisionTree):
        if os.path.exists(path):
            model = joblib.load(path)
        # No optimizer for trees
        return model, None 
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        return model, optimizer
