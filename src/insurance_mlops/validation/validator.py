import pandas as pd
from .schema import validate_schema

class DataValidator:
    def __init__(self):
        self.validation_results = None
    
    def validate(self, data: pd.DataFrame, raise_on_error: bool = True) -> bool:
        self.validation_results = validate_schema(data)
        
        if not self.validation_results['is_valid'] and raise_on_error:
            raise ValueError(
                f"Data validation failed: {self.validation_results['errors']}"
            )
        
        return self.validation_results['is_valid']
    
    def get_validation_report(self) -> dict:
        if self.validation_results is None:
            return {}
        
        report = {
            'status': 'PASS' if self.validation_results['is_valid'] else 'FAIL',
            'errors': self.validation_results['errors'],
            'warnings': self.validation_results['warnings']
        }
        
        return report
