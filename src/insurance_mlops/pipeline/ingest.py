import pandas as pd
import yaml
import os

class Ingestion:
    def __init__(self, config_path=None):
        self.config = self.load_config(config_path)

    def load_config(self, config_path=None):
        if config_path is None:
            env = os.getenv('ENV', 'dev')
            config_path = f'config/{env}.yml'
            if not os.path.exists(config_path):
                config_path = 'config/base.yml'
        
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def load_data(self):
        train_data_path = self.config['data']['train_path']
        test_data_path = self.config['data']['test_path']
        train_data = pd.read_csv(train_data_path)
        test_data = pd.read_csv(test_data_path)
        return train_data, test_data
