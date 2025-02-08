import mlflow
import os

def register_model(model_name: str, run_id: str):
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri, model_name)
    print(f"Model {model_name} registered from run {run_id}")

if __name__ == "__main__":
    model_name = os.getenv("MODEL_NAME", "insurance_model")
    run_id = os.getenv("RUN_ID")
    if run_id:
        register_model(model_name, run_id)
