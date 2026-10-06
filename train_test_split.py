import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

def scale_and_split_data(X, y, test_size=0.2, random_state=42):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def save_split_data(X_train, X_test, y_train, y_test, scaler):
    # Save datasets
    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)

    # Save scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    split_info = {
        'train_size': X_train.shape[0],
        'test_size': X_test.shape[0],
        'train_ratio': X_train.shape[0] / (X_train.shape[0] + X_test.shape[0]),
        'test_ratio': X_test.shape[0] / (X_train.shape[0] + X_test.shape[0]),
        'train_target_dist': y_train.value_counts().to_dict(),
        'test_target_dist': y_test.value_counts().to_dict()
    }

    with open('split_info.pkl', 'wb') as f:
        pickle.dump(split_info, f)

    return split_info

if __name__ == "__main__":
    print("Loading features and target, then scaling and splitting...")
    X = pd.read_csv('features.csv')
    y = pd.read_csv('target.csv').squeeze()

    X_train, X_test, y_train, y_test, scaler = scale_and_split_data(X, y)
    split_info = save_split_data(X_train, X_test, y_train, y_test, scaler)

    print(f"Training set: {split_info['train_size']} samples ({split_info['train_ratio']:.2%})")
    print(f"Test set: {split_info['test_size']} samples ({split_info['test_ratio']:.2%})")
    print(f"Train target distribution: {split_info['train_target_dist']}")
    print(f"Test target distribution: {split_info['test_target_dist']}")
    print("Saved files: X_train.csv, X_test.csv, y_train.csv, y_test.csv")
    print("Saved objects: scaler.pkl, split_info.pkl")