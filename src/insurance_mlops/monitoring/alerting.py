import os
import requests
from typing import Dict, Any

class AlertManager:
    def __init__(self):
        self.slack_webhook_url = os.getenv('SLACK_WEBHOOK_URL')
    
    def send_slack_alert(self, message: str):
        if not self.slack_webhook_url:
            print(f"Slack alert (not sent): {message}")
            return
        
        payload = {'text': message}
        
        try:
            response = requests.post(self.slack_webhook_url, json=payload)
            response.raise_for_status()
        except Exception as e:
            print(f"Failed to send Slack alert: {e}")
    
    def alert_on_drift(self, drift_score: float, threshold: float = 0.5):
        if drift_score > threshold:
            message = f"🚨 Data drift detected! Score: {drift_score:.2f} (threshold: {threshold})"
            self.send_slack_alert(message)
    
    def alert_on_performance_degradation(self, current_metric: float, baseline_metric: float, metric_name: str, threshold: float = 0.1):
        degradation = (baseline_metric - current_metric) / baseline_metric
        
        if degradation > threshold:
            message = f"⚠️ Performance degradation detected for {metric_name}! Current: {current_metric:.4f}, Baseline: {baseline_metric:.4f}, Degradation: {degradation:.2%}"
            self.send_slack_alert(message)
