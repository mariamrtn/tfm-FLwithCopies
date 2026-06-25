'''This script configures the Clients, 
namely: TrainingApp and EvaluationApp of the clients.
'''
import os
import torch
import numpy as np
import pandas as pd
from flwr.app import (
    Array, 
    ArrayRecord, 
    Context, 
    Message, 
    MetricRecord, 
    RecordDict
)
from flwr.clientapp import ClientApp
from pytorchexample_expB.task import (
    BinaryClassifier, 
    SmallNN, 
    DecisionTree, 
    train_one_batch, 
    get_client_partition, 
    get_client_model_path, 
    save_client_model, 
    load_client_model
)

# Client side
app = ClientApp()

# Get the path of the folder that contains the training data for clients
csv_path = os.path.join(os.path.dirname(__file__), "data_clients.csv")

# Number of batches and batch size used for local client training.
batch_size = 32
batches_per_round = 30

# Load full dataset once
df = pd.read_csv(csv_path)
x_all = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
y_all = torch.tensor(df['label'].values, dtype=torch.float32)

# Number of clients in each round, size of the partition
num_clients = 10
partition_size = len(x_all) // num_clients

# This section will be called after the configure_train function of the strategy runs and creates the training messages.
@app.train()
def train(msg: Message, context: Context):
    '''This section will be called after the configure_train function of 
    the strategy runs and creates the training messages.
    '''

    # Get client's id and server round. Load each client's model.
    node_id = context.node_id
    server_round = int(msg.metadata.group_id)
    model, optimizer = load_client_model(node_id)

     # Print if model is new or not. Just for clarification.
    if server_round == 1:
        print(f"[Node {node_id}] Round 1 — model created ({type(model).__name__})")
    else:
        print(f"[Node {node_id}] Round {server_round} — resuming ({type(model).__name__})")

    # Get the training data corresponding to each client
    partition_idx = int(msg.content["config"]["partition_idx"])
    x, y = get_client_partition(x_all, y_all, partition_idx, partition_size, extra_fraction=0.5)

    # Get total number of batches at disposal for each client
    total_batches = len(x) // batch_size

    # Local training of clients according to their models
    if isinstance(model, DecisionTree):

        # Incremental: use more data each FL round to train Decision trees
        num_rounds = 30 # should match your pyproject.toml
        total = len(x)
        chunk_size = total // num_rounds
        end = min(server_round * chunk_size, total)  # add one chunk per round
        model.fit(x[:end], y[:end])
        avg_loss = 0.0
    else:
        total_loss = 0.0
        for i in range(batches_per_round):

            # Get the corresponding batch
            batch_idx = ((server_round - 1) * batches_per_round + i) % total_batches
            start = batch_idx * batch_size
            end = start + batch_size

            # Train the model with the appropriate batch and update loss
            loss = train_one_batch(model, optimizer, x[start:end], y[start:end])
            total_loss += loss
        avg_loss = total_loss / batches_per_round

    # Print average loss after training
    print(f"[Node {node_id}] Round {server_round} — avg loss: {avg_loss:.4f}")

    # Store model parameter, so that we can continue training in next round
    save_client_model(node_id, model, optimizer)

    # Create response messages
    metrics = MetricRecord({"train_loss": avg_loss, "num-examples": len(x)})
    return Message(content=RecordDict({"metrics": metrics}), reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    ''' This section will be called after the configure_evaluate function of 
    the strategy runs and creates the evaluation messages.
    '''

    # Get each client's id and model path to load their models
    node_id = context.node_id
    path = get_client_model_path(node_id)

    # In case path is not found
    if not os.path.exists(path):
        print(f"[Node {node_id}] No model found, skipping eval")
        return Message(content=RecordDict({"metrics": MetricRecord({"predictions": []})}), reply_to=msg)

    # Load each client's model
    model, _ = load_client_model(node_id)

    # Access the batch of synthetic data sent by the server
    batch_array = msg.content["arrays"]["batch_data"]
    x_batch = torch.tensor(batch_array.numpy(), dtype=torch.float32)

    # Predict labels usign the corresponding model
    if isinstance(model, DecisionTree):
        predictions = model.predict_proba(x_batch).tolist()
    else:
        model.eval()
        with torch.no_grad():
            outputs = model(x_batch)
            
            # Hard predictions
            predictions = torch.sigmoid(outputs).squeeze(1).round().tolist()

    # Print number of predictions returned 
    print(f"[Node {node_id}] Eval done — {len(predictions)} predictions returned")

    # Create the reply message
    return Message(content=RecordDict({"metrics": MetricRecord({"predictions": predictions})}), reply_to=msg)