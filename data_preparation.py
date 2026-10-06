import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def clean_data(dirty_data):
    cleaned_data = dirty_data.copy()

    # Handle missing values
    numerical_cols_with_missing = ['Height_cm', 'Weight_kg', 'Blood_Sugar',
                                 'Cholesterol', 'Exercise_Hours_Week']
    for col in numerical_cols_with_missing:
        if col in cleaned_data.columns and cleaned_data[col].isnull().any():
            median_val = cleaned_data[col].median()
            cleaned_data[col].fillna(median_val, inplace=True)

    categorical_cols = cleaned_data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if cleaned_data[col].isnull().any():
            mode_val = cleaned_data[col].mode()[0]
            cleaned_data[col].fillna(mode_val, inplace=True)

    # Remove duplicates
    medical_columns = [col for col in cleaned_data.columns if col != 'Patient_ID']
    cleaned_data = cleaned_data.drop_duplicates(subset=medical_columns, keep='first')

    # Handle outliers using Z-score
    numerical_columns = cleaned_data.select_dtypes(include=[np.number]).columns.tolist()
    numerical_columns = [col for col in numerical_columns if col != 'Patient_ID']

    #IQR-> Interquartile Range
    z_threshold = 3
    for col in numerical_columns:
        z_scores = np.abs(stats.zscore(cleaned_data[col]))
        outliers_mask = z_scores > z_threshold

        if outliers_mask.sum() > 0:
            mean_val = cleaned_data[col].mean()
            std_val = cleaned_data[col].std()
            upper_threshold = mean_val + (z_threshold * std_val)
            lower_threshold = mean_val - (z_threshold * std_val)
            cleaned_data[col] = np.clip(cleaned_data[col], lower_threshold, upper_threshold)

    # Standardize categorical values
    if 'Gender' in cleaned_data.columns:
        cleaned_data['Gender'] = cleaned_data['Gender'].str.upper()
        cleaned_data['Gender'] = cleaned_data['Gender'].replace({'M': 'MALE', 'F': 'FEMALE'})

    # Fix negative values
    columns_to_check = ['Weight_kg', 'Height_cm', 'Age', 'BMI']
    for col in columns_to_check:
        if col in cleaned_data.columns:
            negative_mask = cleaned_data[col] < 0
            if negative_mask.sum() > 0:
                cleaned_data[col] = abs(cleaned_data[col])

    # Recalculate BMI
    if all(col in cleaned_data.columns for col in ['Height_cm', 'Weight_kg']):
        cleaned_data['BMI'] = cleaned_data['Weight_kg'] / ((cleaned_data['Height_cm'] / 100) ** 2)

    return cleaned_data

def generate_cleaning_report(dirty_data, cleaned_data):
    report = {
        'original_shape': dirty_data.shape,
        'cleaned_shape': cleaned_data.shape,
        'missing_values_removed': dirty_data.isnull().sum().sum(),
        'duplicates_removed': dirty_data.shape[0] - cleaned_data.shape[0],
        'outliers_handled': 'Z-score method applied',
        'data_types_standardized': 'Categorical values normalized'
    }
    return report

if __name__ == "__main__":
    print("Loading dirty data and cleaning...")
    dirty_data = pd.read_csv('dirty_patient_data.csv')

    cleaned_data = clean_data(dirty_data)
    report = generate_cleaning_report(dirty_data, cleaned_data)

    # Save cleaned data
    cleaned_data.to_csv('final_cleaned_data.csv', index=False)

    print("Data Cleaning Report:")
    print(f"Original shape: {report['original_shape']}")
    print(f"Cleaned shape: {report['cleaned_shape']}")
    print(f"Missing values handled: {report['missing_values_removed']}")
    print(f"Records removed (duplicates): {report['duplicates_removed']}")
    print(f"Final missing values: {cleaned_data.isnull().sum().sum()}")
    print(f"Final duplicates: {cleaned_data.duplicated().sum()}")
    print(f"Saved to: final_cleaned_data.csv")