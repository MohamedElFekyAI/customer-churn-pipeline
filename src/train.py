import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train():
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    
    model = RandomForestClassifier(
        n_estimators=150, 
        max_depth=6, 
        random_state=42
    )
    model.fit(X_train, y_train)
    
    joblib.dump(model, 'models/churn_model.pkl')
    print("✓ Model trained & saved to models/churn_model.pkl")

if __name__ == "__main__":
    train()
