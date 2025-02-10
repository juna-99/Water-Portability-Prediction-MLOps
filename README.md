# 🌊 Water Potability Prediction with MLOps

This project is an end-to-end machine learning pipeline that predicts whether water is potable or not based on various water quality parameters. The project incorporates **MLOps practices** for automation, reproducibility, and scalability. It also includes a **Tkinter-based GUI** for user interaction, allowing users to input water quality information and get predictions in real-time.

## 📈 Project Overview

The goal of this project is to build a scalable and automated machine learning pipeline that:

1. Collects and preprocesses water quality data.
2. Trains a machine learning model to predict water potability.
3. Deploys the model for real-time predictions using a Tkinter GUI.

The pipeline is built using **DVC (Data Version Control)** for data and model versioning, **MLflow** for experiment tracking, and **Google Cloud Storage (GCS)** for remote data storage. The project demonstrates **MLOps best practices**, including automation, reproducibility, and monitoring.

---

## 🔄 Project Workflow

1. **Experiment Setup**: Use a pre-configured Cookiecutter template and initialize Git for version control.
2. **MLflow Tracking**: Log experiments and model metrics on DagsHub using MLflow.
3. **DVC Pipeline**: Set up data versioning with DVC and build a robust ML pipeline.
4. **Cloud Integration**: Data storage and versioning using Google Cloud Storage (GCS).
5. **Model Registration**: Register the best model in MLflow’s registry using aliases for easy deployment.
6. **Desktop Application**: Create a Tkinter app that fetches the latest model from MLflow and performs predictions.

![Alt text](https://github.com/juna-99/Water-Portability-Prediction-MLOps/blob/32034c756697c4f2e4cd371ba63491c60c97762e/blob/Flowchart%20(1).jpg)

---

## 📂 Project Structure

This project follows a structured workflow to streamline the MLOps process:

### Setup
- Install project structure with Cookiecutter.
- Initialize **Git** and push to **GitHub**.

### Experiment Tracking
1. **DagsHub + MLflow**:
   - Log experiments on DagsHub.
   - Track model metrics, parameters, and artifacts.
   
2. **Experiment Execution**:
   - **Experiment 1**: Baseline model with Random Forest.
   - **Experiment 2**: Multiple models (e.g., Logistic Regression, XGBoost).
   - **Experiment 3**: Test mean vs. median imputation for missing values.
   - **Experiment 4**: Hyperparameter tuning on Random Forest.

### DVC Pipeline
1. **Data Versioning**:
   - Set up DVC for versioning data on Google Cloud Storage.

2. **Pipeline Stages**:
   - **Data Collection**: Gather and structure data.
   - **Data Preprocessing**: Handle missing values (mean imputation).
   - **Model Building**: Train a Random Forest model.
   - **Model Evaluation**: Track performance metrics with MLflow.

### Model Registration
- **MLflow Registry**:
  - Register the best model with optimal parameters and metadata.
  - Assign an alias (e.g., `Production`) instead of using deprecated model stages.
  - Deploy the model using **FastAPI** or **Streamlit** for predictions.

### Tkinter Desktop Application 🖥️
- **Tkinter App**:
   - A simple, user-friendly desktop app built with Tkinter.
   - Automatically fetches the latest model from the MLflow model registry using the alias `Production`.
   - Allows users to input data and receive potability predictions.

---

## 📦 Results and Analysis
- **Best Model**: Random Forest with mean imputation.
- **Optimal Hyperparameters**: `n_estimators=1000`, `max_depth=None`.
- **Performance Metrics**:
  - Accuracy: 68%
  - F1-Score: 0.448
  - Precision: 0.641
  - Recall: 0.344
  
---

## 👤 Credits

- **DataThinkers**: Original project framework.
- **[Your Name]**: Added **Google Cloud Storage (GCS) integration**, updated **MLflow model aliasing**, and improved **Tkinter UI for real-time predictions**.

---

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

