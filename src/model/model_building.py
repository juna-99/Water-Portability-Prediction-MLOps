import pandas as pd
import yaml
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple
from google.cloud import storage

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

# Function to load parameters from a YAML file
def load_params(params_path: str) -> int:
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        return params["model_building"]["n_estimators"]
    except Exception as e:
        raise Exception(f"Error loading parameters from {params_path}: {e}")

# Function to load data from a local file
def load_data(data_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(data_path)
    except Exception as e:
        raise Exception(f"Error loading data from {data_path}: {e}")

# Function to prepare data for training
def prepare_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    try:
        X = data.drop(columns=['Potability'], axis=1)
        y = data['Potability']
        return X, y
    except Exception as e:
        raise Exception(f"Error preparing data: {e}")

# Function to train the model
def train_model(X: pd.DataFrame, y: pd.Series, n_estimators: int) -> RandomForestClassifier:
    try:
        clf = RandomForestClassifier(n_estimators=n_estimators)
        clf.fit(X, y)
        return clf
    except Exception as e:
        raise Exception(f"Error training model: {e}")

# Function to save the trained model
def save_model(model: RandomForestClassifier, model_name: str) -> None:
    try:
        os.makedirs(os.path.dirname(model_name), exist_ok=True)
        with open(model_name, "wb") as file:
            pickle.dump(model, file)
    except Exception as e:
        raise Exception(f"Error saving model to {model_name}: {e}")

# Main function
def main():
    try:
        # GCS bucket and file details
        bucket_name = "my-dvc"  # Replace with your GCS bucket name
        train_source_blob_name = "Processed/train_processed.csv"  # Replace with your GCS path
        train_local_path = r"data\processed\train_processed.csv"  # Local path to save the downloaded file

        # Ensure the local directory exists
        os.makedirs(os.path.dirname(train_local_path), exist_ok=True)

        # Step 1: Download processed training data from GCS
        download_from_gcs(bucket_name, train_source_blob_name, train_local_path)

        # Step 2: Load parameters and data
        params_path = "params.yaml"
        n_estimators = load_params(params_path)
        train_data = load_data(train_local_path)

        # Step 3: Prepare and train the model
        X_train, y_train = prepare_data(train_data)
        model = train_model(X_train, y_train, n_estimators)

        # Step 4: Save the trained model locally
        model_name = "models/model.pkl"
        save_model(model, model_name)

        print("Model trained and saved successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()