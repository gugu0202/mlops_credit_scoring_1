import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from scipy.stats import uniform, randint
from src.config import DATA_PATH, EXPERIMENT_NAME, TRACKING_URI
from src.utils import init_mlflow, log_experiment
from src.preprocessing import load_data, clean_data, feature_engineering, validate_data

def main():
    # Инициализация MLFlow
    init_mlflow()

    # Загрузка и предварительная обработка данных
    df = load_data(DATA_PATH)
    y = df.pop("default_payment_next_month")
    X = df.copy()
    cleaned_X = clean_data(X)
    enriched_X = feature_engineering(cleaned_X)
    validate_data(enriched_X)

    # Разделение на тренировочный и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(enriched_X, y, test_size=0.2, random_state=42)

    # Логистика регрессия
    lr_pipeline = make_pipeline(make_column_transformer((StandardScaler(), ['LIMIT_BAL']), remainder='passthrough'), LogisticRegression(max_iter=1000))
    param_dist_lr = {
        'logisticregression__C': uniform(loc=0, scale=4),
        'logisticregression__solver': ['liblinear'],
        'logisticregression__penalty': ['l1', 'l2']
    }
    grid_lr = RandomizedSearchCV(lr_pipeline, param_dist_lr, n_iter=10, cv=5, scoring='roc_auc', verbose=1, random_state=42)
    grid_lr.fit(X_train, y_train)

    # Метрики LR
    y_pred_proba_lr = grid_lr.predict_proba(X_test)[:, 1]
    y_pred_lr = grid_lr.predict(X_test)
    report = classification_report(y_test, y_pred_lr, output_dict=True)
    metrics_lr = {
        "auc": roc_auc_score(y_test, y_pred_proba_lr),
        "accuracy": grid_lr.score(X_test, y_test),
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"]
    }

    # График ROC-кривой
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
    roc_auc_lr = auc(fpr_lr, tpr_lr)
    fig_lr, ax_lr = plt.subplots()
    ax_lr.plot(fpr_lr, tpr_lr, color='darkorange', lw=2, label=f'LR ROC curve (AUC = {roc_auc_lr:.2f})')
    ax_lr.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax_lr.set_xlabel('False Positive Rate')
    ax_lr.set_ylabel('True Positive Rate')
    ax_lr.set_title('Logistic Regression ROC Curve')
    ax_lr.legend(loc="lower right")
    fig_lr.savefig("lr_roc.png")

    # Логируем эксперимент
    log_experiment("Logistic Regression", grid_lr.best_estimator_, metrics_lr, ["lr_roc.png"])

    # Аналогично проводим для другого алгоритма (например, Gradient Boosting)
    gb_pipeline = make_pipeline(make_column_transformer((StandardScaler(), ['LIMIT_BAL']), remainder='passthrough'), GradientBoostingClassifier())
    param_dist_gb = {
        'gradientboostingclassifier__learning_rate': uniform(loc=0.01, scale=0.5),
        'gradientboostingclassifier__max_depth': randint(low=1, high=10),
        'gradientboostingclassifier__n_estimators': randint(low=50, high=200)
    }
    grid_gb = RandomizedSearchCV(gb_pipeline, param_dist_gb, n_iter=10, cv=5, scoring='roc_auc', verbose=1, random_state=42)
    grid_gb.fit(X_train, y_train)

    # Метрики GB
    y_pred_proba_gb = grid_gb.predict_proba(X_test)[:, 1]
    y_pred_gb = grid_gb.predict(X_test)
    report = classification_report(y_test, y_pred_gb, output_dict=True)
    metrics_gb = {
        "auc": roc_auc_score(y_test, y_pred_proba_gb),
        "accuracy": grid_gb.score(X_test, y_test),
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"]
    }

    # График ROC-кривой
    fpr_gb, tpr_gb, _ = roc_curve(y_test, y_pred_proba_gb)
    roc_auc_gb = auc(fpr_gb, tpr_gb)
    fig_gb, ax_gb = plt.subplots()
    ax_gb.plot(fpr_gb, tpr_gb, color='green', lw=2, label=f'GB ROC curve (AUC = {roc_auc_gb:.2f})')
    ax_gb.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax_gb.set_xlabel('False Positive Rate')
    ax_gb.set_ylabel('True Positive Rate')
    ax_gb.set_title('Gradient Boosting ROC Curve')
    ax_gb.legend(loc="lower right")
    fig_gb.savefig("gb_roc.png")

    # Логируем эксперимент
    log_experiment("Gradient Boosting", grid_gb.best_estimator_, metrics_gb, ["gb_roc.png"])

if __name__ == "__main__":
    main()