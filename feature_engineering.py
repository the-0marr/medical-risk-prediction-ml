import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def prepare_features_and_target(data):
    processed_data = data.copy()

    # Create target variable based on risk assessment
    risk_threshold = processed_data['Risk_Score'].median()
    processed_data['High_Risk'] = (processed_data['Risk_Score'] > risk_threshold).astype(int)

    # Encode categorical variables
    label_encoders = {}
    categorical_columns = ['Gender', 'Smoking', 'Alcohol_Consumption', 'Diabetes',
                          'Hypertension', 'Heart_Disease', 'Insurance_Type']

    for col in categorical_columns:
        if col in processed_data.columns:
            le = LabelEncoder()
            processed_data[col + '_encoded'] = le.fit_transform(processed_data[col])
            label_encoders[col] = le

    # Select strong/clinically relevant feature columns
    feature_columns = [
        'Age', 'Height_cm', 'Weight_kg', 'BMI',
        'Blood_Sugar', 'Cholesterol', 'Hemoglobin',
        'Exercise_Hours_Week',
        'Gender_encoded', 'Smoking_encoded', 'Alcohol_Consumption_encoded',
        'Diabetes_encoded', 'Hypertension_encoded', 'Heart_Disease_encoded'
    ]

    # Filter existing columns
    available_features = [col for col in feature_columns if col in processed_data.columns]

    X = processed_data[available_features]
    y = processed_data['High_Risk']

    # Final processed dataset (only relevant features + target)
    processed_data_final = pd.concat([X, y], axis=1)

    return X, y, label_encoders, processed_data_final

def save_feature_info(X, y, label_encoders):
    feature_info = {
        'feature_names': list(X.columns),
        'target_name': 'High_Risk',
        'n_features': X.shape[1],
        'n_samples': X.shape[0],
        'target_distribution': y.value_counts().to_dict()
    }

    import pickle
    with open('label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)

    with open('feature_info.pkl', 'wb') as f:
        pickle.dump(feature_info, f)

    return feature_info

if __name__ == "__main__":
    print("Preparing features and target variable...")
    data = pd.read_csv('final_cleaned_data.csv')

    X, y, label_encoders, processed_data_final = prepare_features_and_target(data)
    feature_info = save_feature_info(X, y, label_encoders)

    # Save features, target, and processed data
    X.to_csv('features.csv', index=False)
    y.to_csv('target.csv', index=False)
    processed_data_final.to_csv('processed_data.csv', index=False)

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target distribution: {feature_info['target_distribution']}")
    print(f"Feature columns: {len(feature_info['feature_names'])}")
    print("Saved files: features.csv, target.csv, processed_data.csv")
    print("Saved encoders: label_encoders.pkl, feature_info.pkl")
