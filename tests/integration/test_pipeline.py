import pytest
import pandas as pd
from src.insurance_mlops.pipeline.ingest import Ingestion
from src.insurance_mlops.pipeline.clean import Cleaner
from src.insurance_mlops.validation.validator import DataValidator

def test_end_to_end_pipeline():
    validator = DataValidator()
    cleaner = Cleaner()
    
    sample_data = pd.DataFrame({
        'id': [1, 2, 3, 4],
        'SalesChannelID': [1, 1, 1, 1],
        'VehicleAge': [1, 1, 1, 1],
        'DaysSinceCreated': [1, 1, 1, 1],
        'Gender': ['Male', 'Female', 'Male', 'Female'],
        'Age': [30, 25, 40, 35],
        'HasDrivingLicense': [1, 1, 1, 1],
        'RegionID': [1.0, 2.0, 3.0, 4.0],
        'Switch': [0, 1, 0, 1],
        'PastAccident': ['Yes', 'No', 'Yes', 'No'],
        'AnnualPremium': ['£1,200', '£2,500', '£3,000', '£5,000'],
        'target': [0, 1, 0, 1]
    })
    
    validation_result = validator.validate(sample_data, raise_on_error=False)
    assert validation_result['is_valid'] == True
    
    cleaned_data = cleaner.clean_data(sample_data)
    assert 'id' not in cleaned_data.columns
    assert cleaned_data['AnnualPremium'].dtype == float
