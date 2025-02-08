import os
import joblib
import yaml
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

class Predictor:
    def __init__(self, config_path=None):
        self.config = self.load_config(config_path)
        self.model_path = self.config['model']['store_path']
        self.pipeline = self.load_model()

    def load_config(self, config_path=None):
        if config_path is None:
            env = os.getenv('ENV', 'dev')
            config_path = f'config/{env}.yml'
            if not os.path.exists(config_path):
                config_path = 'config/base.yml'
        
        with open(config_path, 'r') as config_file:
            return yaml.safe_load(config_file)
        
    def load_model(self):
        model_file_path = os.path.join(self.model_path, 'model.pkl')
        return joblib.load(model_file_path)

    def feature_target_separator(self, data):
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        return X, y

    def evaluate_model(self, X_test, y_test):
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        return accuracy, class_report, roc_auc
