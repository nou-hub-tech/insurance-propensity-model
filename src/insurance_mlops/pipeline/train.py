import os
import joblib
import yaml
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

class Trainer:
    def __init__(self, config_path=None):
        self.config = self.load_config(config_path)
        self.model_name = self.config['model']['name']
        self.model_params = self.config['model']['params']
        self.model_path = self.config['model']['store_path']
        self.pipeline = self.create_pipeline()

    def load_config(self, config_path=None):
        if config_path is None:
            env = os.getenv('ENV', 'dev')
            config_path = f'config/{env}.yml'
            if not os.path.exists(config_path):
                config_path = 'config/base.yml'
        
        with open(config_path, 'r') as config_file:
            return yaml.safe_load(config_file)
        
    def create_pipeline(self):
        preprocessor = ColumnTransformer(transformers=[
            ('minmax', MinMaxScaler(), ['AnnualPremium']),
            ('standardize', StandardScaler(), ['Age','RegionID']),
            ('onehot', OneHotEncoder(handle_unknown='ignore'), ['Gender', 'PastAccident']),
        ])
        
        smote = SMOTE(sampling_strategy=1.0)
        
        model_map = {
            'RandomForestClassifier': RandomForestClassifier,
            'DecisionTreeClassifier': DecisionTreeClassifier,
            'GradientBoostingClassifier': GradientBoostingClassifier
        }
    
        model_class = model_map[self.model_name]
        model = model_class(**self.model_params)

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('smote', smote),
            ('model', model)
        ])

        return pipeline

    def feature_target_separator(self, data):
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        return X, y

    def train_model(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)

    def save_model(self):
        model_file_path = os.path.join(self.model_path, 'model.pkl')
        joblib.dump(self.pipeline, model_file_path)
