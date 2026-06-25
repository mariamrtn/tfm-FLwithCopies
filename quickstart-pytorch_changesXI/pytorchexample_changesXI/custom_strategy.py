"""Custom Flower Strategy with Copies: In every round of the federated 
learning process, clients train their models first. Then the global 
model is trained as a copy of the "aggregation" of the client's models.
"""

import random
import numpy as np
import os
import torch
from typing import Iterable
from torch.utils.data import DataLoader
from flwr.app import (
    ArrayRecord, 
    ConfigRecord, 
    Message, 
    MetricRecord, 
    RecordDict, 
    Array
)
from flwr.serverapp import Grid
from flwr.serverapp.strategy import Strategy
from pytorchexample_changesXI.task import (
    XYDataset, 
    train_one_batch, 
    predict
)

""" To modify a Flower strategy, it's necessary to build a custom strategy 
by modifying the necessary functions. For this project, __init__(), 
configure_train(), aggregate_train(), configure_evaluate(), 
aggregate_evaluate() and summary() are modified.
""" 

class custom_strategy(Strategy):     
    def __init__(self, model,  lr=0.1, batch_size=32):
        """ We create the objects that will be used in the server side 
        in the federated rounds.
        """

        torch.manual_seed(42)
        np.random.seed(42)

        # Get location of the file
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Load synthetic dataset
        self.dataset = XYDataset(os.path.join(base_dir, 'data_copy.csv'))

        # Create a dataloader
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        
        # Obtain an iterable object
        self.dataloader_iter = iter(self.dataloader)

        # Get the global model instance (created in server_app)
        self.global_model = model

        # Object to store current batch of data
        self.current_batch = None

        # Optimizer used by the global model
        self.optimizer = torch.optim.Adam(self.global_model.parameters(), lr=lr)

        # Create a dictonary to store metrics
        self.metrics_history = {"train_loss": [], "fidelity": [], "test_error": []}
        
        # Get path of the file with the parameters of the global model
        save_path = os.path.join(base_dir, "global_model.pt")

        if os.path.exists(save_path):
            # Load model
            checkpoint = torch.load(save_path)
            self.global_model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            print("Global model loaded from previous run")
        else:
            print("Global model initialized from scratch")

    def configure_train(self, server_round: int, arrays: ArrayRecord, 
        config: ConfigRecord, grid: Grid) -> Iterable[Message]:
        """ Thie function creates training messages for the clients with 
        configuration information for their training. These messages
        trigger app.train() in clients.
        """

        # List to store all messages (one per client)
        messages = []
        node_ids = sorted(grid.get_node_ids())

        # Create messages for all clients
        for partition_idx, node_id in enumerate(node_ids):
            content = RecordDict({
                "arrays": arrays,
                "config": ConfigRecord({"lr": config["lr"], "partition_idx": partition_idx})
            })
            msg = Message(
                content=content,
                # This will trigger the use of @app.train() in clients
                message_type="train",
                dst_node_id=node_id,
                group_id=str(server_round),
            )
            messages.append(msg)

        # Return all messages
        return messages 
    
    def aggregate_train(self, server_round: int, 
        replies: Iterable[Message]) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """ Function used to obtain the batch of data we will train the global model on. 
            """

        # Take the replies from clients after local training
        replies_list = list(replies)

        # In case no clients trained or something happened
        if not replies_list:
            return None, None

        # Get the batch of synthetic data for this round
        self.current_batch = next(self.dataloader_iter) 

        # Convert it into an Array object (necessary to be sent to clients)
        data = Array(self.current_batch.numpy()) 

        # Create outputs
        arrays = ArrayRecord({"batch_data": data})      
        metrics = MetricRecord({"train_loss": 0.0})
        return arrays, metrics

    def configure_evaluate(self, server_round: int, arrays: ArrayRecord, 
        config: ConfigRecord, grid: Grid) -> Iterable[Message]: 
        """This function is used to send the currect batch of synthetic data 
        from the server to each client.
        """

        # Iterable object to store all messages (one per client)
        messages = []
        for node_id in grid.get_node_ids():

            # Here we send the current batch of data to the clients
            content = RecordDict({"arrays": arrays, "config": config})

            # Create the message
            msg = Message(
                content=content,

                # This will trigger the use of @app.evaluate in clients
                message_type="evaluate",
                dst_node_id=node_id,
                group_id=str(server_round),
            )
            messages.append(msg)

        # Return all messages    
        return messages

    def aggregate_evaluate(self, server_round: int, 
        replies: Iterable[Message]) -> MetricRecord | None:
        """The function receives an iterable object of messages sent from the 
        clients to the server, which contains the predictions of each client 
        for the batch of data sent by the previous function 'configure_evaluate'.
        The function returns the training loss, the fidelity and the test error 
        of the copy."""

        # Take the predictions from the clients. 
        replies_list = list(replies)

        # We store predictions in an np.array with dimension
        all_targets = np.array([msg.content["metrics"]["predictions"] for msg in replies_list])

        # Average the predictions of each datapoint
        targets = all_targets.mean(axis=0)           
        
        # Print interesting information
        print(f"all_targets shape: {all_targets.shape}")
        print(f"targets mean: {targets.mean():.3f}  min: {targets.min():.3f}  max: {targets.max():.3f}")
        
        
        # Get the current batch of synthetic data
        batch_data = torch.tensor(self.current_batch.numpy()).float()

        # Train global model
        for i in range(20):
            loss = train_one_batch(self.global_model, self.optimizer, batch_data, targets)

        # Print training loss
        print(f"[Round {server_round}] Training loss: {loss:.4f}")
        
        # Compute the fidelity error of the copy
        global_preds, _ = predict(self.global_model, batch_data)
        global_preds = global_preds.squeeze(1).numpy()
        fidelity = (all_targets.round() == global_preds).mean(axis=1).mean()

        # Print fidelity
        print(f"Client-Server Fidelity: {fidelity*100:.1f}%")

        # Store metrics in dictionary
        self.metrics_history["train_loss"].append(loss)
        self.metrics_history["fidelity"].append(float(fidelity))

        # Return metrics 
        return MetricRecord({"train_loss": loss, "fidelity": fidelity})

    def summary(self) -> None:
        """ 
        This function prints a custom message at the beginning of the FL loop.
        """
        print("Custom Strategy: Federated Learning with copies.") 