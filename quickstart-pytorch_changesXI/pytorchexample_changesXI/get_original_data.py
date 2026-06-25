""" This file generates two data files for training client models 
and evaluating the accuracy of the global model. A normal distribution
is used in this case.
"""
import numpy as np
import pandas as pd
import random

# Generate training data for clients.
x = np.random.normal(loc=0, scale=3, size=64000)
x = np.clip(x, -10, 10)
y = np.random.normal(loc=0, scale=3, size=64000)
y = np.clip(y, -10, 10)

df=pd.DataFrame(list(zip(x,y)), columns=['x', 'y'])
csv=df.to_csv('data_clients.csv', index=False)

# Generate evaluation data for evaluation accuracy of the global model.
x = np.random.normal(loc=0, scale=3, size=6400)
x = np.clip(x, -10, 10)
y = np.random.normal(loc=0, scale=3, size=6400)
y = np.clip(y, -10, 10)

df=pd.DataFrame(list(zip(x,y)), columns=['x', 'y'])
csv=df.to_csv('data_evalmetrics_te.csv', index=False)