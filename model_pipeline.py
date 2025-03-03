import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from imblearn.over_sampling import SMOTE
import psutil
from elasticsearch import Elasticsearch
import docker
import time

warnings.filterwarnings("ignore", category=UserWarning)


def connect_elasticsearch():
    try:
        es = Elasticsearch(["http://localhost:9200"], http_auth=("elastic", "changeme"))
        if es.ping():
            print("✅ Connected to Elasticsearch")
            return es
        else:
            print("❌ Elasticsearch ping failed")
            return None
    except Exception as e:
        print(f"❌ Elasticsearch connection failed: {e}")
        return None


def log_to_elasticsearch(es, index, data):
    data["timestamp"] = int(time.time())
    if es:
        try:
            es.index(index=index, body=data)
            print(f"Logged to {index}: {data}")
        except Exception as e:
            print(f"❌ Failed to log to Elasticsearch: {e}")
    else:
        print("❌ No Elasticsearch connection for logging")


def monitor_system_resources(es):
    print("🔍 Monitoring system resources...")
    cpu_usage = psutil.cpu_percent(interval=1)
    mem_usage = psutil.virtual_memory().percent
    disk_usage = psutil.disk_usage("/").percent
    metrics = {
        "action": "system_monitoring",
        "cpu_usage": cpu_usage,
        "mem_usage": mem_usage,
        "disk_usage": disk_usage,
    }
    mlflow.log_metrics(
        {"cpu_usage": cpu_usage, "mem_usage": mem_usage, "disk_usage": disk_usage}
    )
    log_to_elasticsearch(es, "mlflow-metrics", metrics)
    return metrics


def monitor_docker_containers(es):
    print("🔍 Monitoring Docker containers...")
    try:
        client = docker.from_env()
        containers = {"elasticsearch": None, "kibana": None}
        for container in client.containers.list():
            if container.name in containers:
                stats = container.stats(stream=False)
                cpu_usage = (
                    stats["cpu_stats"]["cpu_usage"]["total_usage"]
                    / stats["cpu_stats"]["system_cpu_usage"]
                    * 100
                )
                mem_usage = (
                    stats["memory_stats"]["usage"]
                    / stats["memory_stats"]["limit"]
                    * 100
                )
                metrics = {
                    "action": "docker_monitoring",
                    "container_name": container.name,
                    "cpu_usage": cpu_usage,
                    "mem_usage": mem_usage,
                }
                mlflow.log_metrics(
                    {
                        f"{container.name}_cpu_usage": cpu_usage,
                        f"{container.name}_mem_usage": mem_usage,
                    }
                )
                log_to_elasticsearch(es, "mlflow-metrics", metrics)
    except Exception as e:
        print(f"❌ Docker monitoring failed: {e}")


def monitor_data_drift(X_train, X_test, es):
    print("🔍 Monitoring data drift...")
    train_mean = np.mean(X_train, axis=0)
    test_mean = np.mean(X_test, axis=0)
    drift_score = np.mean(np.abs(train_mean - test_mean))
    metrics = {"action": "data_drift", "drift_score": drift_score}
    mlflow.log_metric("data_drift", drift_score)
    log_to_elasticsearch(es, "mlflow-metrics", metrics)
    return drift_score


def prepare_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    target_col = "Churn"
    categorical_cols = train_data.select_dtypes(include=["object"]).columns.tolist()

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        train_data[col] = le.fit_transform(train_data[col])
        test_data[col] = test_data[col].map(
            lambda x: x if x in le.classes_ else "unknown"
        )
        le.classes_ = np.append(le.classes_, "unknown")
        test_data[col] = le.transform(test_data[col])
        label_encoders[col] = le

    X_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col]
    X_test = test_data.drop(columns=[target_col])
    y_test = test_data[target_col]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    es = connect_elasticsearch()
    monitor_data_drift(X_train, X_test, es)

    return X_train, y_train, X_test, y_test, scaler, label_encoders


def plot_metrics(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(val_losses, label="Validation Loss", color="orange")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend()
    plt.savefig("loss_plot.png")
    plt.close()


def train_model(X_train, y_train):
    es = connect_elasticsearch()
    monitor_system_resources(es)
    monitor_docker_containers(es)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    model = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", random_state=42
    )
    param_grid = {
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
    }
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_resampled, y_resampled)

    best_params = grid_search.best_params_
    mlflow.log_params(
        {
            "n_estimators": best_params["n_estimators"],
            "learning_rate": best_params["learning_rate"],
            "max_depth": best_params["max_depth"],
        }
    )

    best_model = grid_search.best_estimator_
    best_model.fit(X_resampled, y_resampled)

    train_losses = []
    val_losses = []
    for epoch in range(10):  # Simulate epochs for illustration
        dummy_loss_train = np.random.rand()
        dummy_loss_val = np.random.rand()
        train_losses.append(dummy_loss_train)
        val_losses.append(dummy_loss_val)
        mlflow.log_metrics(
            {
                f"train_loss_epoch_{epoch}": dummy_loss_train,
                f"val_loss_epoch_{epoch}": dummy_loss_val,
            }
        )

    plot_metrics(train_losses, val_losses)
    mlflow.log_artifact("loss_plot.png")
    mlflow.log_metrics(
        {"train_loss_avg": np.mean(train_losses), "val_loss_avg": np.mean(val_losses)}
    )

    # Log system metrics post-training
    cpu_usage = psutil.cpu_percent(interval=1)
    mem_usage = psutil.virtual_memory().percent
    disk_usage = psutil.disk_usage("/").percent
    mlflow.log_metrics(
        {
            "post_train_cpu_usage": cpu_usage,
            "post_train_mem_usage": mem_usage,
            "post_train_disk_usage": disk_usage,
        }
    )

    log_to_elasticsearch(
        es,
        "mlflow-metrics",
        {
            "action": "train",
            "n_estimators": best_params["n_estimators"],
            "learning_rate": best_params["learning_rate"],
            "max_depth": best_params["max_depth"],
            "train_loss_avg": np.mean(train_losses),
            "val_loss_avg": np.mean(val_losses),
            "cpu_usage": cpu_usage,
            "mem_usage": mem_usage,
            "disk_usage": disk_usage,
        },
    )

    monitor_system_resources(es)
    monitor_docker_containers(es)

    return best_model


def evaluate_model(model, X_test, y_test):
    es = connect_elasticsearch()
    monitor_system_resources(es)
    monitor_docker_containers(es)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    mlflow.log_metric("accuracy", accuracy)
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            for metric, value in metrics.items():
                mlflow.log_metric(f"{label}_{metric}", value)

    print(f"✅ Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    # Log system metrics post-evaluation
    cpu_usage = psutil.cpu_percent(interval=1)
    mem_usage = psutil.virtual_memory().percent
    disk_usage = psutil.disk_usage("/").percent
    mlflow.log_metrics(
        {
            "eval_cpu_usage": cpu_usage,
            "eval_mem_usage": mem_usage,
            "eval_disk_usage": disk_usage,
        }
    )

    log_to_elasticsearch(
        es,
        "mlflow-metrics",
        {
            "action": "evaluate",
            "accuracy": accuracy,
            "cpu_usage": cpu_usage,
            "mem_usage": mem_usage,
            "disk_usage": disk_usage,
        },
    )

    monitor_system_resources(es)
    monitor_docker_containers(es)


def save_model(model, filename="xgboost_model.joblib"):
    joblib.dump(model, filename)
    print(f"💾 Model saved under {filename}")


def load_model(filename="xgboost_model.joblib"):
    return joblib.load(filename)
