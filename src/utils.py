!pip install mlflow
import os
import shutil
import mlflow
import mlflow.sklearn

def init_mlflow():
    """
    Инициализирует среду MLFlow и создаёт каталог для экспериментов.
    """
    tracking_uri = "file:/./mlruns/"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Credit Risk Prediction")

def log_experiment(model_name, model, metrics, artifacts=[]):
    """
    Логирует эксперимент в MLFlow.
    :param model_name: Название модели.
    :param model: Сама модель.
    :param metrics: Словарь метрик.
    :param artifacts: Дополнительные артефакты (графики и т.д.).
    """
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        for artifact in artifacts:
            mlflow.log_artifact(artifact)
        mlflow.sklearn.log_model(model, artifact_path="model")