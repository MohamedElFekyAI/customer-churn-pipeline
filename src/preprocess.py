import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def generate_synthetic_data(num_samples: int = 1200) -> pd.DataFrame:
    np.random.seed(42)
    tenure = np.random.randint(1, 72, size=num_samples)
    monthly_charges = np.random.uniform(20.0, 120.0, size=num_samples)
    total_charges = tenure * monthly_charges + np.random.normal(0, 50, size=num_samples)
    total_charges = np.clip(total_charges, a_min=20.0, a_max=None)
    contract_type = np.random.choice([0, 1, 2], size=num_samples, p=[0.5, 0.3, 0.2])
    
    churn_prob = 1 / (1 + np.exp(-(-0.05 * tenure + 0.02 * monthly_charges - 0.8 * contract_type)))
    churn = (np.random.rand(num_samples) < churn_prob).astype(int)
    
    return pd.DataFrame({
        'tenure': tenure,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'contract_type': contract_type,
        'churn': churn
    })

def run_pipeline():
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    df = generate_synthetic_data(1200)
    df.to_csv('data/raw/churn_raw.csv', index=False)
    
    X = df.drop(columns=['churn'])
    y = df['churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    pd.DataFrame(X_train_scaled, columns=X.columns).to_csv('data/processed/X_train.csv', index=False)
    pd.DataFrame(X_test_scaled, columns=X.columns).to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    
    joblib.dump(scaler, 'models/scaler.pkl')
    print("✓ Data preprocessed & Scaler saved.")

if __name__ == "__main__":
    run_pipeline()
