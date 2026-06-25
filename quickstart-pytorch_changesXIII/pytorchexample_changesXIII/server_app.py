"""pytorchexample: A Flower / PyTorch app."""

import os
import torch
import json
import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from pytorchexample_changesXIII.task import BinaryClassifier, SmallNN
from pytorchexample_changesXIII.custom_strategy import custom_strategy #importar la nueva estrategia


""" This file corresponds to the Server. 
The main execution process starts here. """

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp. 
    The whole Federated Learning loop will start here.
    """

    # Read run config from pyproject.toml
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    batch_size = context.run_config["batch-size"]

    # Load global model
    global_model = SmallNN()
    arrays = ArrayRecord(global_model.state_dict())

    # Call our customized strategy (custom_strategy.py)
    strategy = custom_strategy(
        model=global_model,  
        lr=lr, 
        batch_size=batch_size
        )
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    # Get file to store the global model
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(base_dir, "global_model.pt")

    # Save final model in the file 'global_model.pt'
    torch.save({
        "model": strategy.global_model.state_dict(),
        "optimizer": strategy.optimizer.state_dict(),
    }, save_path) # Save final model in the file 'global_model.pt'
    print(f"Final model saved to {save_path}")

    # Save metrics evolution
    metrics_path = os.path.join(base_dir, "metrics_history.json")
    with open(metrics_path, "w") as f:
        json.dump(strategy.metrics_history, f)
    print(f"Metrics saved to {metrics_path}")