# Federated Learning with Copies: Two Experiments
This is an implementation of a new FL approach usign copies of machine learning classifiers and its application in two examples.
The code has been created by modifiying the files from Flower's PyTorch tutorial: https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html
and creating new ones. In this way, we use the already existing information flow architecture given by Flower and customize it so that it performs FL with copies.


# For the first time running the code follow step by step:
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
# Repetition of experiments
Make sure to delete the files and folders generated in the previous run: client_models, global_model.pt, model_params.pth. Datafiles, however, can be kept to skip step 4. Then just run the experiments from the corresponding 'quickstart-pytorch_expA' or 'quickstart-pytorch_expB' folders and analyse the results as explained before. 
