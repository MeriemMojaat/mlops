import argparse
import model_pipeline as mp
import mlflow
import mlflow.sklearn
import os
import psutil
import time
import json


def log_trace(message):
    """Log a trace message with timestamp to MLflow."""
    timestamp = time.time()
    trace_line = f"{time.ctime(timestamp)}: {message}"
    with open("traces.txt", "a") as f:
        f.write(trace_line + "\n")
    if mlflow.active_run():
        mlflow.log_metric("trace_timestamp", timestamp)
    print(trace_line)


def log_system_metrics(step_name):
    """Log system metrics with a prefix and as a JSON artifact in the traces folder."""
    cpu_usage = psutil.cpu_percent(interval=1)
    mem_usage = psutil.virtual_memory().percent
    disk_usage = psutil.disk_usage("/").percent
    metrics = {
        f"system_{step_name}_cpu_usage": cpu_usage,
        f"system_{step_name}_mem_usage": mem_usage,
        f"system_{step_name}_disk_usage": disk_usage,
    }
    if mlflow.active_run():
        mlflow.log_metrics(metrics)

    system_metrics = {
        "step": step_name,
        "cpu_usage": cpu_usage,
        "memory_usage": mem_usage,
        "disk_usage": disk_usage,
        "timestamp": time.ctime(),
    }
    with open(f"system_metrics_{step_name}.json", "w") as f:
        json.dump(system_metrics, f)
    if mlflow.active_run():
        mlflow.log_artifact(f"system_metrics_{step_name}.json", "traces")


def clear_active_run():
    """Clear any active MLflow run if present."""
    if mlflow.active_run():
        active_run = mlflow.active_run()
        log_trace(f"Clearing active run: {active_run.info.run_id}")
        mlflow.end_run()
    log_trace("No active runs remain after cleanup")


def main():
    parser = argparse.ArgumentParser(description="Machine Learning Pipeline")
    parser.add_argument("--train_data", required=True, help="Path to training data CSV")
    parser.add_argument("--test_data", required=True, help="Path to test data CSV")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument(
        "--save", action="store_true", help="Save the model (requires --train)"
    )
    parser.add_argument("--load", action="store_true", help="Load a pre-trained model")
    parser.add_argument("--prepare_data", action="store_true", help="Prepare the data")
    args = parser.parse_args()

    # Clear old traces file
    if os.path.exists("traces.txt"):
        os.remove("traces.txt")

    # Validate input files
    if not os.path.exists(args.train_data):
        log_trace(f"Error: Train data file {args.train_data} does not exist.")
        print(f"Error: Train data file {args.train_data} does not exist.")
        return
    if not os.path.exists(args.test_data):
        log_trace(f"Error: Test data file {args.test_data} does not exist.")
        print(f"Error: Test data file {args.test_data} does not exist.")
        return

    clear_active_run()  # Ensure clean slate at start

    if args.prepare_data:
        with mlflow.start_run(run_name="Prepare Data"):
            log_trace("Starting data preparation")
            log_system_metrics("prepare_start")
            X_train, y_train, X_test, y_test, scaler, label_encoders = mp.prepare_data(
                args.train_data, args.test_data
            )
            log_trace("Data prepared successfully")
            log_system_metrics("prepare_end")
            mlflow.log_artifact("traces.txt", "traces")
        return

    log_trace("Preparing data")
    with mlflow.start_run(run_name="DataPrep") as prep_run:
        X_train, y_train, X_test, y_test, scaler, label_encoders = mp.prepare_data(
            args.train_data, args.test_data
        )
        mlflow.log_artifact("traces.txt", "traces")

    mlflow.set_experiment("Churn Prediction")
    log_trace("Attempting to start Churn Pipeline run")
    with mlflow.start_run(run_name="Churn Pipeline") as run:
        log_trace("Started Churn Pipeline")
        log_system_metrics("pipeline_start")

        if args.train:
            log_trace("Starting model training")
            model = mp.train_model(X_train, y_train)
            input_example = X_train[:1]
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                input_example=input_example,
                registered_model_name=f"ChurnModel_{run.info.run_id}",
            )
            log_trace("Model trained and logged in MLflow")
            log_system_metrics("train_end")
            if args.save:
                model_path = "xgboost_model.joblib"  # Save to root
                mp.save_model(model, model_path)
                mlflow.log_artifact(model_path, "saved_models")
                log_trace(f"Model saved as {model_path}")

        elif args.save:
            log_trace("Cannot save model without training")
            print("❌ Cannot save model without training. Use --train with --save.")
            return

        if args.evaluate:
            model_path = "xgboost_model.joblib"
            if not os.path.exists(model_path):
                log_trace(f"Error: Model file '{model_path}' does not exist")
                print(f"Error: Model file '{model_path}' does not exist.")
                return
            model = mp.load_model(model_path)
            log_trace("Starting model evaluation")
            mp.evaluate_model(model, X_test, y_test)
            log_system_metrics("evaluate_end")

        if args.load:
            model_path = "xgboost_model.joblib"
            if not os.path.exists(model_path):
                log_trace(f"Error: Model file '{model_path}' does not exist")
                print(f"Error: Model file '{model_path}' does not exist.")
                return
            model = mp.load_model(model_path)
            log_trace("Model loaded successfully")

        # Log loss plot and traces
        if os.path.exists("loss_plot.png"):
            mlflow.log_artifact("loss_plot.png", "traces")
        mlflow.log_artifact("traces.txt", "traces")
        log_system_metrics("pipeline_end")


if __name__ == "__main__":
    main()