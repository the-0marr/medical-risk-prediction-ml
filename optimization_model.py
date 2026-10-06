import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').squeeze()
y_test = pd.read_csv('y_test.csv').squeeze()

models = {
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42, n_jobs=-1),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }
    },
    'GradientBoosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [2, 3, 5],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'subsample': [0.8, 1.0]
        }
    },
    'LogisticRegression': {
        'model': LogisticRegression(random_state=42, max_iter=5000),
        'params': {
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs'],
            'penalty': ['l2']
        }
    },
    'SVM': {
        'model': SVC(random_state=42, probability=True),
        'params': [
            {'C': [0.1, 1, 10, 100], 'kernel': ['linear'], 'gamma': ['scale']},
            {'C': [0.1, 1, 10, 100], 'kernel': ['rbf'], 'gamma': ['scale', 'auto']},
            {'C': [0.1, 1, 10, 100], 'kernel': ['poly'], 'gamma': ['scale', 'auto'], 'degree': [2, 3, 4]}
        ]
    },
    'KNN': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3, 5, 7, 9, 11, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan'],
            'p': [1, 2]
        }
    }
}

best_models = {}
best_params = {}
results = []

for name, config in models.items():
    print(f'\nOptimizing {name}...')
    grid = GridSearchCV(config['model'], config['params'], cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    best_models[name] = best_model
    best_params[name] = grid.best_params_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    results.append({
        'Model': name,
        'CV_Accuracy': grid.best_score_,
        'Test_Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1_Score': f1_score(y_test, y_pred, zero_division=0),
        'AUC': roc_auc_score(y_test, y_prob)
    })
    print(f'Best parameters: {grid.best_params_}')
    print(f'Best CV accuracy: {grid.best_score_:.4f}')

results_df = pd.DataFrame(results).sort_values('CV_Accuracy', ascending=False)
results_df.to_csv('optimized_model_results.csv', index=False)

with open('optimized_models.pkl', 'wb') as f:
    pickle.dump(best_models, f)
with open('optimized_model_params.pkl', 'wb') as f:
    pickle.dump(best_params, f)

best_model_name = results_df.iloc[0]['Model']
best_model = best_models[best_model_name]
with open('best_optimized_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

model_info = {
    'best_model_name': best_model_name,
    'best_cv_accuracy': float(results_df.iloc[0]['CV_Accuracy']),
    'best_test_accuracy': float(results_df.loc[results_df['Model'] == best_model_name, 'Test_Accuracy'].iloc[0]),
    'best_parameters': best_params[best_model_name]
}
with open('optimized_model_info.pkl', 'wb') as f:
    pickle.dump(model_info, f)

print('\nOPTIMIZATION RESULTS')
print(results_df.to_string(index=False))
print(f'\nBest model by CV accuracy: {best_model_name}')
print(f'Best CV accuracy: {model_info["best_cv_accuracy"]:.4f}')
print(f'Test accuracy: {model_info["best_test_accuracy"]:.4f}')
print('\nClassification Report:')
print(classification_report(y_test, best_model.predict(X_test), zero_division=0))
print('Confusion Matrix:')
print(confusion_matrix(y_test, best_model.predict(X_test)))

plt.figure(figsize=(10, 6))
plt.bar(results_df['Model'], results_df['Test_Accuracy'])
plt.ylim(0, 1)
plt.xlabel('Models')
plt.ylabel('Test Accuracy')
plt.title('Optimized Model Accuracy Comparison')
plt.xticks(rotation=45)
for i, value in enumerate(results_df['Test_Accuracy']):
    plt.text(i, value + 0.01, f'{value:.3f}', ha='center')
plt.tight_layout()
plt.savefig('optimized_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
