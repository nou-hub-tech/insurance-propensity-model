import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from evidently.report import Report
from evidently.metric_preset import ClassificationPreset
import os

class PerformanceMonitor:
    def __init__(self, model_path: str, test_data_path: str):
        self.model = joblib.load(model_path)
        self.test_data = pd.read_csv(test_data_path)
    
    def evaluate_performance(self):
        X = self.test_data.iloc[:, :-1]
        y = self.test_data.iloc[:, -1]
        
        y_pred = self.model.predict(X)
        
        accuracy = accuracy_score(y, y_pred)
        class_report = classification_report(y, y_pred, output_dict=True)
        roc_auc = roc_auc_score(y, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'precision': class_report['weighted avg']['precision'],
            'recall': class_report['weighted avg']['recall'],
            'f1': class_report['weighted avg']['f1-score'],
            'roc_auc': roc_auc
        }
        
        return metrics
    
    def generate_report(self, output_path: str):
        X = self.test_data.iloc[:, :-1]
        y = self.test_data.iloc[:, -1]
        
        y_pred = self.model.predict(X)
        
        report = Report(metrics=[ClassificationPreset()])
        report.run(reference_data=self.test_data, current_data=self.test_data.assign(prediction=y_pred))
        report.save_html(output_path)
