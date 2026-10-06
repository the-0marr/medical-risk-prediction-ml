import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import warnings
warnings.filterwarnings('ignore')


def evaluate_all_models(trained_models, X_test, y_test):
    results = {}

    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }

    return results


def create_visualizations(results, y_test):
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 15))

    # 1. Model Comparison Bar Plot
    plt.subplot(2, 3, 1)
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    model_names = list(results.keys())

    x = np.arange(len(model_names))
    width = 0.2

    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in model_names]
        plt.bar(x + i * width, values, width, label=metric.capitalize())

    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x + width * 1.5, model_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. ROC Curves
    plt.subplot(2, 3, 2)
    for name, result in results.items():
        if result['probabilities'] is not None:
            fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
            auc_score = result['auc']
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. Best Model Confusion Matrix
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    plt.subplot(2, 3, 3)
    cm = results[best_model_name]['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    # 4. Feature Importance (if available)
    for name, trained_model in results.items():
        if hasattr(trained_model, 'feature_importances_'):
            plt.subplot(2, 3, 4)
            with open('feature_info.pkl', 'rb') as f:
                feature_info = pickle.load(f)

            feature_importance = pd.DataFrame({
                'feature': feature_info['feature_names'],
                'importance': trained_model.feature_importances_
            }).sort_values('importance', ascending=True).tail(10)

            plt.barh(feature_importance['feature'], feature_importance['importance'])
            plt.title('Top 10 Feature Importance')
            plt.xlabel('Importance')
            break

    # 5. Accuracy Comparison
    plt.subplot(2, 3, 5)
    accuracies = [results[model]['accuracy'] for model in model_names]
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
    bars = plt.bar(model_names, accuracies, color=colors)
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.xticks(rotation=45)

    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{acc:.3f}', ha='center', va='bottom')

    # 6. AUC Comparison
    plt.subplot(2, 3, 6)
    auc_scores = [results[model]['auc'] for model in model_names if results[model]['auc'] is not None]
    auc_models = [model for model in model_names if results[model]['auc'] is not None]

    if auc_scores:
        colors = plt.cm.plasma(np.linspace(0, 1, len(auc_models)))
        bars = plt.bar(auc_models, auc_scores, color=colors)
        plt.xlabel('Models')
        plt.ylabel('AUC Score')
        plt.title('Model AUC Comparison')
        plt.xticks(rotation=45)

        for bar, auc in zip(bars, auc_scores):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{auc:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('model_evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def save_evaluation_results(results):
    summary_data = []
    for model, metrics in results.items():
        summary_data.append({
            'Model': model,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1_Score': metrics['f1'],
            'AUC': metrics['auc'] if metrics['auc'] is not None else 'N/A'
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('model_evaluation_summary.csv', index=False)

    with open('evaluation_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    return summary_df

'''def save_modelmatrix_csv(results, y_test, X_test=None):
    """
    Saves a CSV file for each model containing Actual labels, Predictions, and Predicted Probabilities if available.
    """
    for model_name, metrics in results.items():
        df = pd.DataFrame({
            'Actual': y_test,
            'Prediction': metrics['predictions'],
            'Probability': metrics['probabilities'] if metrics['probabilities'] is not None else np.nan
        })
        # Uncomment below to add IDs or features from X_test if necessary
        # if X_test is not None and 'ID' in X_test.columns:
        #     df.insert(0, 'ID', X_test['ID'].values)
        df.to_csv(f"{model_name}_modelmatrix.csv", index=False)'''

if __name__ == "__main__":
    print("Loading models and test data for evaluation...")

    # Load trained models and test data
    with open('trained_models.pkl', 'rb') as f:
        trained_models = pickle.load(f)

    X_test = pd.read_csv('X_test.csv')
    y_test = pd.read_csv('y_test.csv').squeeze()

    # Evaluate all models
    results = evaluate_all_models(trained_models, X_test, y_test)

    # Create visualizations
    create_visualizations(results, y_test)

    # Save evaluation summary and detailed results
    summary_df = save_evaluation_results(results)

    # Save model matrix CSV files with Actual, Predictions, and Probabilities
    #save_modelmatrix_csv(results, y_test, X_test)

    print("\nModel Evaluation Summary:")
    print(summary_df.to_string(index=False))

    best_model = summary_df.loc[summary_df['Accuracy'].idxmax(), 'Model']
    best_accuracy = summary_df['Accuracy'].max()

    print(f"\nBest performing model: {best_model}")
    print(f"Best accuracy: {best_accuracy:.4f}")
    print("\nSaved files: model_evaluation_summary.csv, evaluation_results.pkl, model_evaluation_results.png, [modelname]_modelmatrix.csv")
