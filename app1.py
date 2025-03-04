from flask import Flask, render_template, request, jsonify, redirect, url_for
import requests
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.sklearn

# Initialize Flask instance
app = Flask(__name__)

# Configure MLflow
mlflow.set_experiment("Churn_Prediction")

# Load the trained model
MODEL_PATH = "xgboost_model.joblib"
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print(f"Model not found at {MODEL_PATH}. Please ensure it exists.")
    model = None


@app.route("/")
def home():
    return render_template("main.html")  # Updated to use main.html


@app.route("/predict_form")
def predict_form():
    return render_template("index.html")  # Using index.html (formerly predict.html)


@app.route("/retrain_form")
def retrain_form():
    return render_template("retrain.html")


@app.route("/predict", methods=["POST"])
def predict():
    input_data = {
        "State": request.form["State"],
        "Account_length": int(request.form["Account_length"]),
        "Area_code": int(request.form["Area_code"]),
        "International_plan": request.form["International_plan"],
        "Voice_mail_plan": request.form["Voice_mail_plan"],
        "Number_vmail_messages": int(request.form["Number_vmail_messages"]),
        "Total_day_minutes": float(request.form["Total_day_minutes"]),
        "Total_day_calls": int(request.form["Total_day_calls"]),
        "Total_day_charge": float(request.form["Total_day_charge"]),
        "Total_eve_minutes": float(request.form["Total_eve_minutes"]),
        "Total_eve_calls": int(request.form["Total_eve_calls"]),
        "Total_eve_charge": float(request.form["Total_eve_charge"]),
        "Total_night_minutes": float(request.form["Total_night_minutes"]),
        "Total_night_calls": int(request.form["Total_night_calls"]),
        "Total_night_charge": float(request.form["Total_night_charge"]),
        "Total_intl_minutes": float(request.form["Total_intl_minutes"]),
        "Total_intl_calls": int(request.form["Total_intl_calls"]),
        "Total_intl_charge": float(request.form["Total_intl_charge"]),
        "Customer_service_calls": int(request.form["Customer_service_calls"]),
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict", json=input_data, timeout=5
        )
        response.raise_for_status()
        prediction = response.json().get("prediction", "No prediction received.")
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e), "prediction": None}), 500

    return redirect(url_for("result", prediction=prediction))


@app.route("/result")
def result():
    prediction = request.args.get("prediction", "No prediction found")
    return render_template("result.html", prediction=prediction)


@app.route("/retrain", methods=["POST"])
def retrain():
    try:
        params = request.get_json()
        required_params = [
            "learning_rate",
            "n_estimators",
            "max_depth",
            "min_child_weight",
            "gamma",
            "subsample",
            "colsample_bytree",
        ]

        for param in required_params:
            if param not in params:
                return jsonify({"error": f"Missing parameter: {param}"}), 400

        train_data = pd.read_csv("churn-bigml-80.csv")
        test_data = pd.read_csv("churn-bigml-20.csv")

        X_train = train_data.drop(columns=["Churn"])
        y_train = train_data["Churn"]
        X_test = test_data.drop(columns=["Churn"])
        y_test = test_data["Churn"]

        categorical_cols = ["State", "International plan", "Voice mail plan"]
        for col in categorical_cols:
            X_train[col] = X_train[col].astype("category").cat.codes
            X_test[col] = X_test[col].astype("category").cat.codes

        with mlflow.start_run():
            mlflow.log_param("learning_rate", params["learning_rate"])
            mlflow.log_param("n_estimators", params["n_estimators"])
            mlflow.log_param("max_depth", params["max_depth"])
            mlflow.log_param("min_child_weight", params["min_child_weight"])
            mlflow.log_param("gamma", params["gamma"])
            mlflow.log_param("subsample", params["subsample"])
            mlflow.log_param("colsample_bytree", params["colsample_bytree"])

            new_model = xgb.XGBClassifier(
                learning_rate=params["learning_rate"],
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_child_weight=params["min_child_weight"],
                gamma=params["gamma"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                use_label_encoder=False,
                eval_metric="logloss",
            )

            new_model.fit(X_train, y_train)
            predictions = new_model.predict(X_test)
            accuracy = np.mean(predictions == y_test)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(new_model, "model")

        global model
        model = new_model
        joblib.dump(new_model, MODEL_PATH)

        return jsonify(
            {"message": "Modèle réentraîné avec succès", "accuracy": float(accuracy)}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
