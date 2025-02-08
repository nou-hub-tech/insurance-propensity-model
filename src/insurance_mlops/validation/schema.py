import pandas as pd
from typing import Dict, List, Any

class DataSchema:
    REQUIRED_COLUMNS = [
        'Gender', 'Age', 'HasDrivingLicense', 'RegionID', 
        'Switch', 'PastAccident', 'AnnualPremium'
    ]
    
    COLUMN_DTYPES = {
        'Gender': 'object',
        'Age': 'float64',
        'HasDrivingLicense': 'float64',
        'RegionID': 'float64',
        'Switch': 'float64',
        'PastAccident': 'object',
        'AnnualPremium': 'float64'
    }
    
    VALUE_RANGES = {
        'Age': (18, 100),
        'HasDrivingLicense': (0, 1),
        'Switch': (-1, 1),
        'AnnualPremium': (0, 100000)
    }
    
    CATEGORICAL_VALUES = {
        'Gender': ['Male', 'Female'],
        'PastAccident': ['Yes', 'No', 'Unknown']
    }
    
    MAX_MISSING_PERCENTAGE = 0.3

def validate_schema(data: pd.DataFrame) -> Dict[str, Any]:
    results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    missing_cols = set(DataSchema.REQUIRED_COLUMNS) - set(data.columns)
    if missing_cols:
        results['is_valid'] = False
        results['errors'].append(f"Missing required columns: {missing_cols}")
    
    for col, expected_dtype in DataSchema.COLUMN_DTYPES.items():
        if col in data.columns:
            actual_dtype = str(data[col].dtype)
            if actual_dtype != expected_dtype:
                results['warnings'].append(
                    f"Column '{col}' has dtype '{actual_dtype}', expected '{expected_dtype}'"
                )
    
    for col, (min_val, max_val) in DataSchema.VALUE_RANGES.items():
        if col in data.columns:
            out_of_range = data[(data[col] < min_val) | (data[col] > max_val)]
            if len(out_of_range) > 0:
                results['warnings'].append(
                    f"Column '{col}' has {len(out_of_range)} values outside range [{min_val}, {max_val}]"
                )
    
    for col, valid_values in DataSchema.CATEGORICAL_VALUES.items():
        if col in data.columns:
            invalid_values = set(data[col].dropna().unique()) - set(valid_values)
            if invalid_values:
                results['warnings'].append(
                    f"Column '{col}' has invalid values: {invalid_values}"
                )
    
    missing_percentage = data.isnull().sum() / len(data)
    high_missing_cols = missing_percentage[missing_percentage > DataSchema.MAX_MISSING_PERCENTAGE]
    if not high_missing_cols.empty:
        results['warnings'].append(
            f"Columns with high missing values: {high_missing_cols.to_dict()}"
        )
    
    return results
