import mlflow
from mlflow.tracking import MlflowClient
import os

def rollback_model(model_name: str, version: str):
    client = MlflowClient()
    
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production"
    )
    print(f"Model {model_name} version {version} rolled back to Production")

if __name__ == "__main__":
    model_name = os.getenv("MODEL_NAME", "insurance_model")
    version = os.getenv("VERSION")
    if version:
        rollback_model(model_name, version)
