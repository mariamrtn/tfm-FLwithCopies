""" This file generates two the synthetic data files for training the global 
model and computing its fidelity. A uniform distribution is used to generate
the data.
"""

import pandas as pd
import numpy as np
import random

# Synthetic data to train the copy
x = [random.uniform(-10, 10) for _ in range (128000)]
y = [random.uniform(-10, 10) for _ in range (128000)]

df=pd.DataFrame(list(zip(x,y)), columns=['x', 'y'])
csv=df.to_csv('data_copy.csv', index=False)

# Synthetic data to evaluate fidelity
x = [random.uniform(-10, 10) for _ in range (12800)]
y = [random.uniform(-10, 10) for _ in range (12800)]
df=pd.DataFrame(list(zip(x,y)), columns=['x', 'y'])
csv=df.to_csv('data_evalmetrics_fi.csv', index=False)