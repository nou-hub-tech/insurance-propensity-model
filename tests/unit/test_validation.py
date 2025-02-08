import pytest
import pandas as pd
from src.insurance_mlops.validation.validator import DataValidator

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'Gender': ['Male', 'Female', 'Male', 'Female'],
        'Age': [30, 25, 40, 35],
        'HasDrivingLicense': [1, 1, 1, 1],
        'RegionID': [1.0, 2.0, 3.0, 4.0],
        'Switch': [0, 1, 0, 1],
        'PastAccident': ['Yes', 'No', 'Yes', 'No'],
        'AnnualPremium': [1200.0, 2500.0, 3000.0, 5000.0],
        'target': [0, 1, 0, 1]
    })

def test_data_validation(sample_data):
    validator = DataValidator()
    result = validator.validate(sample_data, raise_on_error=False)
    
    assert result['is_valid'] == True
    assert len(result['errors']) == 0
