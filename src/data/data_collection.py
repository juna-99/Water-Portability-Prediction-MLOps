import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from typing import Tuple
import yaml
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

# Function to upload data to GCS
def upload_to_gcs(bucket_name: str, source_file_name: str, destination_blob_name: str) -> None:
    """Upload a file from a local path to GCS."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        print(f"Uploaded {source_file_name} to GCS as {destination_blob_name}.")
    except Exception as e:
        raise Exception(f"Error uploading to GCS: {e}")

# Function to load parameters from a YAML file
def load_params(filepath: str) -> float:
    try:
        with open(filepath, "r") as file:
            params = yaml.safe_load(file)
        return params["data_collection"]["test_size"]
    except Exception as e:
        raise Exception(f"Error loading parameters from {filepath}: {e}")

# Function to load data from a local file
def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath}: {e}")

# Function to split data into training and testing sets
def split_data(data: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        return train_test_split(data, test_size=test_size, random_state=42)
    except Exception as e:
        raise Exception(f"Error splitting data: {e}")

# Function to save data to a local file
def save_data(df: pd.DataFrame, filepath: str) -> None:
    try:
        df.to_csv(filepath, index=False)
    except Exception as e:
        raise Exception(f"Error saving data to {filepath}: {e}")

# Main function
def main():
    # GCS bucket and file details
    bucket_name = "my-dvc"  
    source_blob_name = "Raw /water_potability.csv"  
    local_raw_data_path = os.path.join("data", "raw", "water_potability.csv")  # Local path to save the downloaded file
    local_split_data_path = os.path.join("data", "raw")  # Local path to save split data

    # Ensure the local directories exist
    os.makedirs(os.path.dirname(local_raw_data_path), exist_ok=True)
    os.makedirs(local_split_data_path, exist_ok=True)

    # Step 1: Download data from GCS
    download_from_gcs(bucket_name, source_blob_name, local_raw_data_path)

    # Step 2: Load parameters
    params_filepath = "params.yaml"
    test_size = load_params(params_filepath)

    # Step 3: Load and process the data
    data = load_data(local_raw_data_path)
    train_data, test_data = split_data(data, test_size)

    # Step 4: Save processed data locally
    train_filepath = os.path.join(local_split_data_path, "train.csv")
    test_filepath = os.path.join(local_split_data_path, "test.csv")
    save_data(train_data, train_filepath)
    save_data(test_data, test_filepath)

    # Step 5: Upload processed data to GCS
    upload_to_gcs(bucket_name, train_filepath, "Raw /train.csv")
    upload_to_gcs(bucket_name, test_filepath, "Raw /test.csv")

if __name__ == "__main__":
    main()