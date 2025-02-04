import mlflow
import dagshub

mlflow.set_tracking_uri("https://dagshub.com/juna-99/Water-Portability-Prediction-MLOps.mlflow")

dagshub.init(repo_owner='juna-99', repo_name='Water-Portability-Prediction-MLOps', mlflow=True)


with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)