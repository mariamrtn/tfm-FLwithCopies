"""This file generates two synthetics dataset:
one to train the copy, another one to evaluate its fidelity."""

import pandas as pd
import numpy as np
import copy
from pandas.api.types import (
    is_bool_dtype, 
    is_numeric_dtype
)
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Function obtained from https://github.com/mozilla/PRESC/blob/master/presc/copies/sampling.py
def is_discrete(s):
    return is_bool_dtype(s) or not is_numeric_dtype(s)

# Function obtained from https://github.com/mozilla/PRESC/blob/master/presc/copies/sampling.py
def dynamical_range(df, verbose=False):
    range_dict = {}
    for feature in df:
        range_dict[feature] = {
            "min": df[feature].min(),
            "max": df[feature].max(),
            "mean": df[feature].mean(),
            "sigma": df[feature].std(),
        }

        if verbose:
            print(
                f"\n{feature}"
                f"\n  min: {range_dict[feature]['min']:.4f}"
                f"\n  max: {range_dict[feature]['max']:.4f}"
                f"\n    (interval: "
                f"{range_dict[feature]['max'] - range_dict[feature]['min']:.4f})"
                f"\n  mean: {range_dict[feature]['mean']:.4f}"
                f"\n  sigma: {range_dict[feature]['sigma']:.4f}"
            )

    return range_dict

# Function obtained from https://github.com/mozilla/PRESC/blob/master/presc/copies/sampling.py
def find_categories(df, add_nans=False):
    categories_dict = {}
    for feature in df:
        if is_discrete(df[feature]):
            # Remove NaN values from selection
            df_no_nans = df[df[feature].notnull()]

            # Log fraction of NaN values if required
            if add_nans:
                nan_fraction = df[feature].isnull().sum() / len(df)
                total_length = len(df)
            else:
                nan_fraction = 0
                total_length = len(df_no_nans)

            categories_dict[feature] = {
                "categories": {
                    key: None for key in df_no_nans[feature].unique().tolist()
                }
            }
            for category in categories_dict[feature]["categories"].keys():
                categories_dict[feature]["categories"][category] = (
                    df_no_nans[feature].value_counts()[category] / total_length
                )
            if add_nans and nan_fraction != 0:
                categories_dict[feature]["categories"]["NaNs"] = nan_fraction

    return categories_dict
    
# Function obtained from https://github.com/mozilla/PRESC/blob/master/presc/copies/sampling.py
def mixed_data_features(df, add_nans=False):
    features_dict = {}
    for feature in df:
        df_feature = df[[feature]]
        if is_discrete(df[feature]):
            single_feature_parameters = find_categories(df_feature, add_nans=add_nans)
        else:
            single_feature_parameters = dynamical_range(df_feature)
        features_dict[feature] = single_feature_parameters[feature]

    return features_dict

# Function obtained from https://github.com/mozilla/PRESC/blob/master/presc/copies/sampling.py
def normal_sampling(feature_parameters,nsamples=500,random_state=None,):
    if random_state is not None:
        np.random.seed(seed=random_state)

    # Compute number of features
    nfeatures = len(feature_parameters)

    # Rename columns
    feature_names = []
    mus = []
    sigmas = []
    for key in feature_parameters:
        feature_names.append(key)
        mus.append(feature_parameters[key]["mean"])
        sigmas.append(feature_parameters[key]["sigma"])

    mus = np.array(mus)
    covariate_matrix = np.eye(nfeatures, nfeatures) * (np.array(sigmas)) ** 2

    # Generate normal distribution data
    X_generated = pd.DataFrame(
        np.random.multivariate_normal(mus, covariate_matrix, size=nsamples)
    )

    # Rename columns
    X_generated.columns = feature_names

    return X_generated


# Load breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Generate new df with scaled features
X_df = pd.DataFrame(X_scaled, columns=data.feature_names)
feature_parameters=mixed_data_features(X_df)

# Generate synthetic data
synthetic_data=normal_sampling(feature_parameters, nsamples=12000, random_state=None)

# Split into two sets of synthetic data
X_data_copy, X_evalmetrics_fi= train_test_split(synthetic_data, test_size=0.2, random_state=42)

feature_cols = [f"feature_{i}" for i in range(synthetic_data.shape[1])]

# Generate the two datasets
for name, X in [("data_copy", X_data_copy),
                ("data_evalmetrics_fi", X_evalmetrics_fi)]:
    df = X.copy()

    # Rename columns
    df.columns = feature_cols

    # Dummy variable
    df["label"] = 0
    df.to_csv(f"{name}.csv", index=False)
    print(f"{name}.csv: {len(df)} rows")

