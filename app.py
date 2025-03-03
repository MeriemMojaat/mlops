from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.sklearn

# Configuration de MLflow o ay haja tetsajel fel mlflow tahet l esem Churn_Prediction
mlflow.set_experiment("Churn_Prediction")

# Load the trained model
MODEL_PATH = "xgboost_model.joblib"
model = joblib.load(MODEL_PATH)


# Définition du schéma des entrées data kifeh lezem todkhel o tkoun
class PredictionInput(BaseModel):
    State: str
    Account_length: int
    Area_code: int
    International_plan: str
    Voice_mail_plan: str
    Number_vmail_messages: int
    Total_day_minutes: float
    Total_day_calls: int
    Total_day_charge: float
    Total_eve_minutes: float
    Total_eve_calls: int
    Total_eve_charge: float
    Total_night_minutes: float
    Total_night_calls: int
    Total_night_charge: float
    Total_intl_minutes: float
    Total_intl_calls: int
    Total_intl_charge: float
    Customer_service_calls: int


# create instance to initialize web service
app = FastAPI()


@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        input_dict = input_data.dict()

        # Renommer les colonnes pour correspondre au modèle
        input_dict_renamed = {
            "State": input_dict["State"],
            "Account length": input_dict["Account_length"],
            "Area code": input_dict["Area_code"],
            "International plan": input_dict["International_plan"],
            "Voice mail plan": input_dict["Voice_mail_plan"],
            "Number vmail messages": input_dict["Number_vmail_messages"],
            "Total day minutes": input_dict["Total_day_minutes"],
            "Total day calls": input_dict["Total_day_calls"],
            "Total day charge": input_dict["Total_day_charge"],
            "Total eve minutes": input_dict["Total_eve_minutes"],
            "Total eve calls": input_dict["Total_eve_calls"],
            "Total eve charge": input_dict["Total_eve_charge"],
            "Total night minutes": input_dict["Total_night_minutes"],
            "Total night calls": input_dict["Total_night_calls"],
            "Total night charge": input_dict["Total_night_charge"],
            "Total intl minutes": input_dict["Total_intl_minutes"],
            "Total intl calls": input_dict["Total_intl_calls"],
            "Total intl charge": input_dict["Total_intl_charge"],
            "Customer service calls": input_dict["Customer_service_calls"],
        }

        input_df = pd.DataFrame([input_dict_renamed])

        # Encodage des variables catégoriques
        categorical_cols = ["State", "International plan", "Voice mail plan"]
        for col in categorical_cols:
            input_df[col] = input_df[col].astype("category").cat.codes

        feature_order = [
            "State",
            "Account length",
            "Area code",
            "International plan",
            "Voice mail plan",
            "Number vmail messages",
            "Total day minutes",
            "Total day calls",
            "Total day charge",
            "Total eve minutes",
            "Total eve calls",
            "Total eve charge",
            "Total night minutes",
            "Total night calls",
            "Total night charge",
            "Total intl minutes",
            "Total intl calls",
            "Total intl charge",
            "Customer service calls",
        ]
        input_df = input_df[feature_order]

        # Faire la prédiction
        prediction = model.predict(input_df)

        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Définition du schéma pour les hyperparamètres du réentraînement
class RetrainParams(BaseModel):
    learning_rate: float
    n_estimators: int
    max_depth: int
    min_child_weight: float
    gamma: float
    subsample: float
    colsample_bytree: float


@app.post("/retrain")
def retrain(params: RetrainParams):
    try:
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

        # Commencer une nouvelle exécution MLflow
        with mlflow.start_run():
            mlflow.log_param("learning_rate", params.learning_rate)
            mlflow.log_param("n_estimators", params.n_estimators)
            mlflow.log_param("max_depth", params.max_depth)
            mlflow.log_param("min_child_weight", params.min_child_weight)
            mlflow.log_param("gamma", params.gamma)
            mlflow.log_param("subsample", params.subsample)
            mlflow.log_param("colsample_bytree", params.colsample_bytree)

            # Configurer le modèle
            new_model = xgb.XGBClassifier(
                learning_rate=params.learning_rate,
                n_estimators=params.n_estimators,
                max_depth=params.max_depth,
                min_child_weight=params.min_child_weight,
                gamma=params.gamma,
                subsample=params.subsample,
                colsample_bytree=params.colsample_bytree,
                use_label_encoder=False,
                eval_metric="logloss",
            )

            # Entraînement du modèle
            new_model.fit(X_train, y_train)
            predictions = new_model.predict(X_test)
            accuracy = np.mean(predictions == y_test)

            # Log des métriques dans MLflow
            mlflow.log_metric("accuracy", accuracy)

            # Sauvegarder le modèle dans MLflow
            mlflow.sklearn.log_model(new_model, "model")

        # Sauvegarde locale du modèle
        joblib.dump(new_model, MODEL_PATH)

        return {"message": "Modèle réentraîné avec succès", "accuracy": accuracy}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/")
def read_root():
    return {"message": "Welcome to the Prediction API!"}
