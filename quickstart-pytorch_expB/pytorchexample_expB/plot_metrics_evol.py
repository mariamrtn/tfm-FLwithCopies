""" This file contains the code to generate plots showing the 
evolution of the training metrics. """

import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Get the path where metrics are stored after the FL rounds
base_dir = os.path.dirname(os.path.abspath(__file__))
metrics_path = os.path.join(base_dir, "metrics_history.json")

# Load the metrics
with open(metrics_path, "r") as f:
    metrics = json.load(f)

rounds = list(range(1, len(metrics["train_loss"]) + 1))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot training loss
ax1.plot(
    rounds, 
    metrics["train_loss"], 
    color="green", 
    marker="o", 
    label="train loss"
    )
ax1.set_xlabel("Round")
ax1.set_ylabel("BCE Loss")
ax1.set_title("Training Loss")
ax1.legend()
ax1.grid()

# Plot fidelity
ax2.plot(rounds, metrics["fidelity"], color="steelblue", marker="o", label="fidelity")
ax2.set_xlabel("Round")
ax2.set_ylabel("Value")
ax2.set_title("Fidelity")
ax2.legend()
ax2.grid()

plt.tight_layout()

# Show image
plt.show()
