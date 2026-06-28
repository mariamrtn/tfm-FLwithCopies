# Federated Learning with Copies: Two Experiments
This is an implementation of a new FL approach using copies of machine learning classifiers and its application in two examples.
The code has been created by modifiying the files from Flower's PyTorch tutorial: https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html
and creating new ones. In this way, we use the already existing information flow architecture given by Flower and customize it so that it performs FL with copies.

More precisely, in each of the experiments, the files:
* `client_app.py` and `task.py` have been **completely modified** compared to the ones in the original tutorial, and now consist of code fully developed for these particular experiments.
* `server_app.py` and `pyproject.toml` have been **modified to a smaller extent** and most of the code of the original tutorial is reused.
* `custom_strategy.py`, `evaluation_metrics.py`, `get_original_data.py`, `get_synthetic_data.py`, `plot_metrics_evol.py` and `plot_models_db.py` did not exist in the original tutorial and have been **fully created** for this project.

# Runing the project. Follow step by step:
1. Clone the repo:
```bash
git clone https://github.com/mariamrtn/tfm-FLwithCopies
cd tfm-FLwithCopies
```

2. Recreate the environment:
```bash
conda env create -f environment.yaml
conda activate flwr-env
```

3. Choose an experiment:
```bash
cd quickstart-pytorch_expA
```
or

```bash
cd quickstart-pytorch_expB
```

4. Generate data files:
```bash
cd pytorchexample_expA
python get_original_data.py
python get_synthetic_data.py
```
or

```bash
cd pytorchexample_expB
python get_original_data.py
python get_synthetic_data.py
```

5. Run the experiments:
```bash
cd ..
flwr run .
```

6. Analyse results
```bash
cd pytorchexample_expA
python plot_metrics_evol.py
python plot_models_db.py
python evaluation_metrics.py
```
or

```bash
cd pytorchexample_expB
python plot_metrics_evol.py
python evaluation_metrics.py
```
## Repetition of experiments
Make sure to delete the files and folders generated in the previous run: client_models, global_model.pt, model_params.pth. Datafiles, however, can be kept to skip step 4. Then just run the experiments from the corresponding 'quickstart-pytorch_expA' or 'quickstart-pytorch_expB' folders and analyse the results as explained before. 

# Project Structure 
The repository contains two experiments:

- `quickstart-pytorch_expA/`: Implementation of Experiment A.
- `quickstart-pytorch_expB/`: Implementation of Experiment B (same structure as Experiment A with the corresponding implementation changes).

Each experiment contains:
- `client_app.py` – Defines the ClientApp. It is where clients' local training happens and where predictions for the synthetic data used to train the global model (copy) are obtaiend.
- `server_app.py` – Defines the ServerApp. Here the global model is initialized and stored and it is where the FL rounds are started according to our new custom strategy.
- `custom_strategy.py` – Defines the custom FL strategy with copies that we have created.
- `task.py` – Contains functions used in other files.
- `evaluation_metrics.py` – Computes two histograms to evaluate the clients and global model after it has been trained: fidelity histogram and copy error histogram.
- `get_original_data.py` – Generates the original datasets used for local training of clients and evaluation of the global model's accuracy.
- `get_synthetic_data.py` – Generates the synthetic datasets used for training the global model and fidelity evaluation.
- `plot_metrics_evol.py` – Plots the evolution of training loss and fidelity across the FL rounds.
- `plot_models_db.py` – Visualizes the decision boundaries of the clients and the global model in Experiment A
- `pyproject.toml` - Defines the project configuration, dependencies, and application entry points

Also, the repository contains:
- `.gitignore` – Specifies files and directories ignored by Git.
- `environment.yaml` – Lists the Conda environment and required dependencies.
- `README.md` – Provides an overview of the project, instructions to set it up, and general information.
