# Predictive Model Package

## Overview

This folder contains the code and released checkpoints for the molecular
property prediction models used in this work.

Four regression tasks are included:

- detonation velocity (D, m/s)
- thermal decomposition temperature (Td, deg C)
- density (Ro, g/cm^3)
- natural logarithm of impact sensitivityln (ln(IS), dimensionless) 

The implementation is based on a Directed Message Passing Neural Network
(DMPNN) using Chemprop.

This package supports:

- train/test dataset splitting
- 10-fold cross-validation training
- property prediction for new molecules
- metric evaluation using `MAE`, `RMSE`, and `R2`

## Directory Layout

- `Checkpoints/`
  - Released model packages for the final four predictors.
- `data_core/`
  - Utilities for reading CSV files, generating RDKit-based features, and
    creating K-fold splits.
- `model_core/`
  - Local wrapper around the DMPNN model used by both training and inference.
- `training_core/`
  - Training loop, validation loop, and optimizer/scheduler helpers.
- `split_dataset_randomly.py`
  - Creates random train/test splits.
- `train_model.py`
  - Main entry point for model training.
- `predict_properties.py`
  - Main entry point for prediction on external molecules.
- `evaluate_model_metrics.py`
  - Evaluates a trained model on labeled datasets.
- `training_config.yaml`
  - Example training configuration file.
- `requirements.txt`
  - Python dependency list.

## Requirements

Install the dependencies with:

```bash
pip install -r requirements.txt
```

The required packages include:

- `chemprop==1.6.1`
- `torch`
- `rdkit`
- `pandas`
- `numpy`
- `scikit-learn`
- `pyyaml`
- `tqdm`
