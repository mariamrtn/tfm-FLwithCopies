1. Clone the repo:
```bash
git clone https://github.com/mariamrtn/tfm-FLwithCopies
cd tfm-FLwithCopies
```

2. Recreate the environment:
```bash
conda env create -f environment.yml
conda activate flower-env
```

3. Choose an experiment:
```bash
cd quickstart-pytorch_expA

# or

cd quickstart-pytorch_expB
```

4. Generate data files:
```bash
cd pytorchexample_expA
python get_original_data.py
python get_synthetic_data.py

# or

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

# or

cd pytorchexample_expB
python plot_metrics_evol.py
python evaluation_metrics.py
```

7. Repetition of experiments
Make sure to delete the files and folders generated in the previous run: client_models, global_model.pt, model_params.pth
Datafiles can be kept to skip step 4. Then just run and analyse the results as explained before. 
