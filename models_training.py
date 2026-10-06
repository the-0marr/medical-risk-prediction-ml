import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import pickle
import warnings
warnings.filterwarnings('ignore')

def train_multiple_models(X_train, y_train):
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42, probability=True),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }

    trained_models = {}
    model_scores = {}

    for name, model in models.items():
        print(f"Training {name}...")

        # Train model
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        model_scores[name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }

        print(f"{name} CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    return trained_models, model_scores

def select_best_model(model_scores):
    best_model_name = max(model_scores, key=lambda x: model_scores[x]['cv_mean'])
    best_score = model_scores[best_model_name]['cv_mean']

    return best_model_name, best_score

def save_models(trained_models, model_scores, best_model_name):
    # Save all trained models
    with open('trained_models.pkl', 'wb') as f:
        pickle.dump(trained_models, f)

    # Save model scores
    with open('model_scores.pkl', 'wb') as f:
        pickle.dump(model_scores, f)

    # Save best model separately
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(trained_models[best_model_name], f)

    # Save model info
    model_info = {
        'best_model_name': best_model_name,
        'best_model_score': model_scores[best_model_name]['cv_mean'],
        'all_model_scores': {name: scores['cv_mean'] for name, scores in model_scores.items()}
    }

    with open('model_info.pkl', 'wb') as f:
        pickle.dump(model_info, f)

    return model_info

if __name__ == "__main__":
    print("Loading training data and training models...")
    X_train = pd.read_csv('X_train.csv')
    y_train = pd.read_csv('y_train.csv').squeeze()

    trained_models, model_scores = train_multiple_models(X_train, y_train)
    best_model_name, best_score = select_best_model(model_scores)
    model_info = save_models(trained_models, model_scores, best_model_name)

    print(f"\nModel Selection Results:")
    print(f"Best model: {best_model_name}")
    print(f"Best CV score: {best_score:.4f}")
    print(f"\nAll model scores:")
    for name, score in model_info['all_model_scores'].items():
        print(f"  {name}: {score:.4f}")

    print("\nSaved files: trained_models.pkl, model_scores.pkl, best_model.pkl, model_info.pkl")