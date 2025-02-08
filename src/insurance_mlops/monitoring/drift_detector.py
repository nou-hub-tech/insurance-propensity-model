import pandas as pd
import joblib
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.ui.workspace import Workspace
import os

class DriftDetector:
    def __init__(self, reference_data_path: str, production_data_path: str):
        self.reference_data = pd.read_csv(reference_data_path)
        self.production_data = pd.read_csv(production_data_path)
        self.workspace = Workspace('mlruns')
    
    def detect_drift(self):
        target_column = self.reference_data.columns[-1]
        
        reference_data = self.reference_data.drop(columns=[target_column])
        production_data = self.production_data.drop(columns=[target_column])
        
        data_drift_report = Report(metrics=[DataDriftPreset()])
        data_drift_report.run(reference_data=reference_data, current_data=production_data)
        
        return data_drift_report
    
    def save_report(self, report, output_path: str):
        report.save_html(output_path)
    
    def get_drift_score(self):
        report = self.detect_drift()
        return report.as_dict()
