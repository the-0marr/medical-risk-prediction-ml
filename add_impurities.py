import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def add_data_impurities(data):
    np.random.seed(42)
    dirty_data = data.copy()

    # Add missing values
    missing_columns = ['Height_cm', 'Weight_kg', 'Blood_Sugar', 'Cholesterol', 'Exercise_Hours_Week']
    for col in missing_columns:
        missing_indices = np.random.choice(dirty_data.index, size=int(0.05 * len(dirty_data)), replace=False)
        #print(f"These are the missing indexes: f{missing_indices}")
        dirty_data.loc[missing_indices, col] = np.nan

    # Add duplicates
    duplicate_indices = np.random.choice(dirty_data.index, size=25, replace=False)
    duplicate_rows = dirty_data.loc[duplicate_indices].copy()
    duplicate_rows['Patient_ID'] = duplicate_rows['Patient_ID'] + 10000
    dirty_data = pd.concat([dirty_data, duplicate_rows], ignore_index=True)

    # Add outliers
    outlier_modifications = {
        'Age': (5, [150, 200, 250]),
        'BMI': (10, [60, 70, 80]),
        'Systolic_BP': (8, [300, 350, 400]),
        'Heart_Rate': (7, [200, 220, 250]),
        'Blood_Sugar': (12, [500, 600, 700])
    }

    for col, (count, outlier_values) in outlier_modifications.items():
        outlier_indices = np.random.choice(dirty_data.index, size=count, replace=False)
        dirty_data.loc[outlier_indices, col] = np.random.choice(outlier_values, size=count)

    # Add data entry errors
    negative_indices = np.random.choice(dirty_data.index, size=5, replace=False)
    dirty_data.loc[negative_indices, 'Weight_kg'] = -dirty_data.loc[negative_indices, 'Weight_kg']

    inconsistent_indices = np.random.choice(dirty_data.index, size=8, replace=False)
    dirty_data.loc[inconsistent_indices, 'Gender'] = np.random.choice(['M', 'F', 'male', 'female'], size=8)

    return dirty_data

if __name__ == "__main__":
    print("Loading clean data and adding impurities...")
    clean_data = pd.read_csv('clean_patient_data.csv')

    dirty_data = add_data_impurities(clean_data)

    # Save dirty data
    dirty_data.to_csv('dirty_patient_data.csv', index=False)

    print(f"Original dataset: {clean_data.shape}")
    print(f"Dirty dataset: {dirty_data.shape}")
    print(f"Missing values: {dirty_data.isnull().sum().sum()}")
    print(f"Duplicates: {dirty_data.duplicated().sum()}")
    print(f"Saved to: dirty_patient_data.csv")