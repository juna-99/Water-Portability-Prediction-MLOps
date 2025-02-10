import pandas as pd
import numpy as np
import os
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

# Function to load data from a local file
def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath}: {e}")

# Function to fill missing values with the mean
def fill_missing_with_mean(df: pd.DataFrame) -> pd.DataFrame:
    try:
        for column in df.columns:
            if df[column].isnull().any():
                mean_value = df[column].mean()
                df[column].fillna(mean_value, inplace=True)
        return df
    except Exception as e:
        raise Exception(f"Error filling missing values with mean: {e}")

# Function to save data to a local file
def save_data(df: pd.DataFrame, filepath: str) -> None:
    try:
        df.to_csv(filepath, index=False)
    except Exception as e:
        raise Exception(f"Error saving data to {filepath}: {e}")

# Main function
def main():
    # GCS bucket and file details
    bucket_name = "my-dvc"  # Replace with your GCS bucket name
    raw_data_path = r"data\raw"  # Local path to raw data
    processed_data_path = r"data\processed"  # Local path to processed data

    # Ensure the local directories exist
    os.makedirs(raw_data_path, exist_ok=True)
    os.makedirs(processed_data_path, exist_ok=True)

    # Step 1: Download raw data from GCS
    train_source_blob_name = "Raw /train.csv"  # Replace with your GCS path
    test_source_blob_name = "Raw /test.csv"  # Replace with your GCS path

    train_local_path = os.path.join(raw_data_path, "train.csv")
    test_local_path = os.path.join(raw_data_path, "test.csv")

    download_from_gcs(bucket_name, train_source_blob_name, train_local_path)
    download_from_gcs(bucket_name, test_source_blob_name, test_local_path)

    # Step 2: Load and preprocess the data
    train_data = load_data(train_local_path)
    test_data = load_data(test_local_path)

    train_processed_data = fill_missing_with_mean(train_data)
    test_processed_data = fill_missing_with_mean(test_data)

    # Step 3: Save processed data locally
    train_processed_path = os.path.join(processed_data_path, "train_processed.csv")
    test_processed_path = os.path.join(processed_data_path, "test_processed.csv")

    save_data(train_processed_data, train_processed_path)
    save_data(test_processed_data, test_processed_path)

    print(f"Saved processed data to {train_processed_path} and {test_processed_path}")

# Step 4: Upload processed data to GCS
    train_destination_blob_name = "Processed/train_processed.csv"  
    test_destination_blob_name = "Processed/test_processed.csv"

    upload_to_gcs(bucket_name, train_processed_path, train_destination_blob_name)
    upload_to_gcs(bucket_name, test_processed_path, test_destination_blob_name)

if __name__ == "__main__":
    main()