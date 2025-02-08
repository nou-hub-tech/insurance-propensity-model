import mlflow
from mlflow.tracking import MlflowClient
import os

def promote_model(model_name: str, stage: str):
    client = MlflowClient()
    
    model_version_infos = client.search_model_versions(name=model_name)
    latest_version = model_version_infos[-1].version
    
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage=stage
    )
    print(f"Model {model_name} version {latest_version} promoted to {stage}")

if __name__ == "__main__":
    model_name = os.getenv("MODEL_NAME", "insurance_model")
    stage = os.getenv("STAGE", "Staging")
    promote_model(model_name, stage)
