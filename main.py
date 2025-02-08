import logging
import yaml
import os
import mlflow
import mlflow.sklearn
from src.insurance_mlops.pipeline.ingest import Ingestion
from src.insurance_mlops.pipeline.clean import Cleaner
from src.insurance_mlops.pipeline.train import Trainer
from src.insurance_mlops.pipeline.predict import Predictor
from src.insurance_mlops.validation.validator import DataValidator
from src.insurance_mlops.utils.logging import setup_logger
from src.insurance_mlops.utils.config import load_config
from sklearn.metrics import classification_report

logger = setup_logger(__name__)

def main():
    config = load_config()
    
    ingestion = Ingestion()
    train, test = ingestion.load_data()
    logger.info("Data ingestion completed successfully")

    validator = DataValidator()
    validator.validate(train, raise_on_error=True)
    validator.validate(test, raise_on_error=True)
    logger.info("Data validation completed successfully")

    cleaner = Cleaner()
    train_data = cleaner.clean_data(train)
    test_data = cleaner.clean_data(test)
    logger.info("Data cleaning completed successfully")

    trainer = Trainer()
    X_train, y_train = trainer.feature_target_separator(train_data)
    trainer.train_model(X_train, y_train)
    trainer.save_model()
    logger.info("Model training completed successfully")

    predictor = Predictor()
    X_test, y_test = predictor.feature_target_separator(test_data)
    accuracy, class_report, roc_auc_score = predictor.evaluate_model(X_test, y_test)
    logger.info("Model evaluation completed successfully")
    
    print("\n============= Model Evaluation Results ==============")
    print(f"Model: {trainer.model_name}")
    print(f"Accuracy Score: {accuracy:.4f}, ROC AUC Score: {roc_auc_score:.4f}")
    print(f"\n{class_report}")
    print("=====================================================\n")


def train_with_mlflow():
    config = load_config()
    logger_config = config.get('logging', {})
    logger = setup_logger(__name__, logger_config.get('level', 'INFO'), logger_config.get('format'))

    mlflow_config = config.get('mlflow', {})
    mlflow.set_experiment(mlflow_config.get('experiment_name', 'Model Training Experiment'))
    
    with mlflow.start_run() as run:
        ingestion = Ingestion()
        train, test = ingestion.load_data()
        logger.info("Data ingestion completed successfully")

        validator = DataValidator()
        validator.validate(train, raise_on_error=True)
        validator.validate(test, raise_on_error=True)
        logger.info("Data validation completed successfully")

        cleaner = Cleaner()
        train_data = cleaner.clean_data(train)
        test_data = cleaner.clean_data(test)
        logger.info("Data cleaning completed successfully")

        trainer = Trainer()
        X_train, y_train = trainer.feature_target_separator(train_data)
        trainer.train_model(X_train, y_train)
        trainer.save_model()
        logger.info("Model training completed successfully")
        
        predictor = Predictor()
        X_test, y_test = predictor.feature_target_separator(test_data)
        accuracy, class_report, roc_auc_score = predictor.evaluate_model(X_test, y_test)
        report = classification_report(y_test, trainer.pipeline.predict(X_test), output_dict=True)
        logger.info("Model evaluation completed successfully")
        
        mlflow.set_tag('Model developer', 'nou-hub-tech')
        mlflow.set_tag('preprocessing', 'OneHotEncoder, Standard Scaler, and MinMax Scaler')
        
        model_params = config['model']['params']
        mlflow.log_params(model_params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc", roc_auc_score)
        mlflow.log_metric('precision', report['weighted avg']['precision'])
        mlflow.log_metric('recall', report['weighted avg']['recall'])
        mlflow.sklearn.log_model(trainer.pipeline, "model")
                
        model_name = mlflow_config.get('model_name', 'insurance_model')
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, model_name)

        logger.info("MLflow tracking completed successfully")

        print("\n============= Model Evaluation Results ==============")
        print(f"Model: {trainer.model_name}")
        print(f"Accuracy Score: {accuracy:.4f}, ROC AUC Score: {roc_auc_score:.4f}")
        print(f"\n{class_report}")
        print("=====================================================\n")
        
if __name__ == "__main__":
    train_with_mlflow()
