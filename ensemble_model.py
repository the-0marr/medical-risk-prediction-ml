import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv').squeeze()

with open('optimized_models.pkl', 'rb') as f:
    models = pickle.load(f)

probabilities = {}
predictions = {}
for name, model in models.items():
    predictions[name] = model.predict(X_test)
    probabilities[name] = model.predict_proba(X_test)[:, 1]

probability_df = pd.DataFrame(probabilities)
ensemble_probability = probability_df.mean(axis=1)
ensemble_prediction = (ensemble_probability >= 0.5).astype(int)

metrics = {
    'Model': 'Ensemble',
    'Accuracy': accuracy_score(y_test, ensemble_prediction),
    'Precision': precision_score(y_test, ensemble_prediction, zero_division=0),
    'Recall': recall_score(y_test, ensemble_prediction, zero_division=0),
    'F1_Score': f1_score(y_test, ensemble_prediction, zero_division=0),
    'AUC': roc_auc_score(y_test, ensemble_probability)
}

rows = []
for name in models:
    rows.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, predictions[name]),
        'Precision': precision_score(y_test, predictions[name], zero_division=0),
        'Recall': recall_score(y_test, predictions[name], zero_division=0),
        'F1_Score': f1_score(y_test, predictions[name], zero_division=0),
        'AUC': roc_auc_score(y_test, probabilities[name])
    })
rows.append(metrics)
comparison = pd.DataFrame(rows)
comparison.to_csv('model_vs_ensemble_results.csv', index=False)

prediction_output = probability_df.copy()
prediction_output['Ensemble_Probability'] = ensemble_probability
prediction_output['Actual'] = y_test.to_numpy()
prediction_output['Final_Prediction'] = ensemble_prediction
prediction_output.to_csv('ensemble_predictions.csv', index=False)

with open('ensemble_model.pkl', 'wb') as f:
    pickle.dump({'models': models, 'method': 'Soft Voting', 'threshold': 0.5}, f)

print('\nENSEMBLE RESULTS')
print(comparison.to_string(index=False))
print('\nClassification Report:')
print(classification_report(y_test, ensemble_prediction, zero_division=0))
print('Confusion Matrix:')
print(confusion_matrix(y_test, ensemble_prediction))

plt.figure(figsize=(10, 6))
plt.bar(comparison['Model'], comparison['Accuracy'])
plt.ylim(0, 1)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Individual Models vs Ensemble')
plt.xticks(rotation=45)
for i, value in enumerate(comparison['Accuracy']):
    plt.text(i, value + 0.01, f'{value:.3f}', ha='center')
plt.tight_layout()
plt.savefig('model_vs_ensemble.png', dpi=300, bbox_inches='tight')
plt.close()

print(f'\nFinal ensemble accuracy: {metrics["Accuracy"]:.4f}')
print(f'Final ensemble AUC: {metrics["AUC"]:.4f}')
