import numpy as np
import pandas as pd
import pickle
import json
import mlflow
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from google.cloud import storage
from mlflow import log_metric, log_param, log_artifact
import mlflow.sklearn
import dagshub
from mlflow.models import infer_signature

# Initialize DagsHub for experiment tracking
dagshub.init(repo_owner='juna-99', repo_name='Water-Portability-Prediction-MLOps', mlflow=True)

# Set the experiment name in MLflow
mlflow.set_experiment("DVC PIPELINE")

# Set the tracking URI for MLflow to log the experiment in DagsHub
mlflow.set_tracking_uri("https://dagshub.com/juna-99/Water-Portability-Prediction-MLOps.mlflow")

# Function to download data from GCS
def download_from_gcs(bucket_name: str, source_blob_name: str, destination_file_name: str) -> None:
    """Download a file from GCS to a local path."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        print(f"Downloaded {source_blob_name} from GCS to {destination_file_name}.")
    except Exception as e:
        raise Exception(f"Error downloading from GCS: {e}")

# Function to load data from a local file
def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath}: {e}")

# Function to prepare data for evaluation
def prepare_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    try:
        X = data.drop(columns=['Potability'], axis=1)
        y = data['Potability']
        return X, y
    except Exception as e:
        raise Exception(f"Error preparing data: {e}")

# Function to load the trained model
def load_model(filepath: str):
    try:
        with open(filepath, "rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        raise Exception(f"Error loading model from {filepath}: {e}")

# Function to evaluate the model
def evaluation_model(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> dict:
    try:
        params = yaml.safe_load(open("params.yaml", "r"))
        test_size = params["data_collection"]["test_size"]
        n_estimators = params["model_building"]["n_estimators"]

        y_pred = model.predict(X_test)

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log parameters and metrics
        mlflow.log_param("Test_size", test_size)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix for {model_name}")
        cm_dir = "reports"
        os.makedirs(cm_dir, exist_ok=True)  # Ensure the reports directory exists
        cm_path = os.path.join(cm_dir, f"confusion_matrix_{model_name.replace(' ', '_')}.png")

        plt.savefig(cm_path)

        # Log confusion matrix artifact
        mlflow.log_artifact(cm_path)

        metrics_dict = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        return metrics_dict
    except Exception as e:
        raise Exception(f"Error evaluating model: {e}")

# Function to save metrics to a JSON file
def save_metrics(metrics: dict, metrics_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w') as file:
            json.dump(metrics, file, indent=4)
    except Exception as e:
        raise Exception(f"Error saving metrics to {metrics_path}: {e}")

# Main function
def main():
    try:
        # GCS bucket and file details
        bucket_name = "my-dvc"  # Replace with your GCS bucket name
        test_source_blob_name = "Processed/test_processed.csv"  # Replace with your GCS path
        test_local_path = "/data/processed/test_processed.csv"  # Local path to save the downloaded file

        # Ensure the local directory exists
        os.makedirs(os.path.dirname(test_local_path), exist_ok=True)

        # Step 1: Download processed test data from GCS
        download_from_gcs(bucket_name, test_source_blob_name, test_local_path)

        # Step 2: Load data and model
        test_data = load_data(test_local_path)
        X_test, y_test = prepare_data(test_data)
        model_path = "models/model.pkl"
        model = load_model(model_path)

        # Step 3: Evaluate the model
        model_name = "Best Model"
        metrics_path = "reports/metrics.json"

        # Start MLflow run
        with mlflow.start_run() as run:
            metrics = evaluation_model(model, X_test, y_test, model_name)
            save_metrics(metrics, metrics_path)

            # Log artifacts
            mlflow.log_artifact(model_path)
            mlflow.log_artifact(metrics_path)
            mlflow.log_artifact(__file__)

            # Log the model
            signature = infer_signature(X_test, model.predict(X_test))
            mlflow.sklearn.log_model(model, "Best Model", signature=signature)

            # Save run ID and model info to JSON file
            run_info = {'run_id': run.info.run_id, 'model_name': "Best Model"}
            reports_path = "reports/run_info.json"
            with open(reports_path, 'w') as file:
                json.dump(run_info, file, indent=4)

    except Exception as e:
        raise Exception(f"An error occurred: {e}")

if __name__ == "__main__":
    main()