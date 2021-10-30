import json
from random import random, randint
import yaml
import joblib
import mlflow
import argparse
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, confusion_matrix, classification_report

def read_params(config_path):
    """
    read parameters from the params.yaml file
    input: params.yaml location
    output: parameters as dictionary
    """
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def test_mlflow(config_path):
    config = read_params(config_path)
    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]
    print(f"Remote server uri: {remote_server_uri}")

    print("Establecemos el Server Tracking")
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_config["experiment_name"])

    print("Realizamos una ejecución y hemos establecido el experimento")
    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:
        mlflow.log_param("param1", randint(0, 100))
        mlflow.log_param("param2", randint(0, 100))


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    test_mlflow(config_path=parsed_args.config)