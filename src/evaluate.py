import json
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report

def evaluate():
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    model = joblib.load('models/churn_model.pkl')
    
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "precision": round(precision_score(y_test, predictions), 4),
        "recall": round(recall_score(y_test, predictions), 4),
        "roc_auc": round(roc_auc_score(y_test, probabilities), 4),
        "detailed_report": classification_report(y_test, predictions, output_dict=True)
    }
    
    with open('models/evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"✓ Model evaluated | Accuracy: {metrics['accuracy']} | ROC-AUC: {metrics['roc_auc']}")

if __name__ == "__main__":
    evaluate()
