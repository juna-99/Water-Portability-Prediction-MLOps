import json
from mlflow.tracking import MlflowClient
import mlflow

import dagshub

dagshub.init(repo_owner='juna-99', repo_name='Water-Portability-Prediction-MLOps', mlflow=True)

# Set the experiment name in MLflow
mlflow.set_experiment("Final_Model")

# Set the tracking URI for MLflow to log the experiment in DagsHub
mlflow.set_tracking_uri("https://dagshub.com/juna-99/Water-Portability-Prediction-MLOps.mlflow")

# Load the run ID and model name from the saved JSON file
reports_path = "reports/run_info.json"
with open(reports_path, 'r') as file:
    run_info = json.load(file)

run_id = run_info['run_id'] # Fetch run id from the JSON file
model_name = run_info['model_name']  # Fetch model name from the JSON file

# Create an MLflow client
client = MlflowClient()

# Create the model URI
model_uri = f"runs:/{run_id}/artifacts/{model_name}"

# Register the model
reg = mlflow.register_model(model_uri, model_name)

# Get the model version
model_version = reg.version  # Get the registered model version

# Define alias instead of deprecated stages
alias = "Production"  # Equivalent to 'Staging' stage

# Fetch the latest registered model version
latest_version = client.get_latest_versions(model_name)[0].version

# Assign alias to the latest model version
client.set_registered_model_alias(model_name, alias, latest_version)

# Optionally, add a tag to indicate validation status
client.set_model_version_tag(model_name, latest_version, "validation_status", "pending")

print(f"Model {model_name} version {latest_version} assigned alias '{alias}' and tagged for validation.")
