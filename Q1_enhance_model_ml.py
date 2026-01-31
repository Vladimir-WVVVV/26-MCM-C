import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------
# Helper Functions for Manual ML (No sklearn)
# ---------------------------------------------------------

class SimpleLinearRegression:
    def __init__(self, regularization=1e-5):
        self.weights = None
        self.lambda_reg = regularization
        
    def fit(self, X, y):
        # Closed form solution: w = (X^T X + lambda*I)^-1 X^T y
        n_features = X.shape[1]
        I = np.eye(n_features)
        # Don't regularize bias term (assuming first col is bias)
        I[0, 0] = 0 
        
        X_T = X.T
        try:
            self.weights = np.linalg.inv(X_T @ X + self.lambda_reg * I) @ X_T @ y
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if singular
            self.weights = np.linalg.pinv(X_T @ X + self.lambda_reg * I) @ X_T @ y
            
    def predict(self, X):
        return X @ self.weights

def prepare_features(df):
    """
    Extract features for ML model.
    """
    df = df.copy()
    
    # 1. Numerical Features & Normalization
    # Min-Max Scaling
    df['score_norm'] = df.groupby(['season', 'week'])['score'].transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6))
    df['week_norm'] = df['week'] / 10.0
    
    # 2. Categorical Features (One-Hot Encoding Manually)
    # Just take top industries for simplicity
    top_industries = ['athlete', 'actor', 'singer', 'model', 'comedian']
    for ind in top_industries:
        df[f'ind_{ind}'] = df['industry'].astype(str).apply(lambda x: 1 if ind in x.lower() else 0)
        
    # 3. Age
    df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(35)
    df['age_norm'] = (df['age'] - 20) / 50.0 # Approximate normalization
    
    # 4. Lag Features
    df = df.sort_values(['season', 'name', 'week'])
    df['prev_vote_share'] = df.groupby(['season', 'name'])['est_vote_share'].shift(1).fillna(0.1)
    
    # 5. Polynomial / Interaction Features
    df['score_sq'] = df['score_norm'] ** 2
    df['age_sq'] = df['age_norm'] ** 2
    df['score_x_age'] = df['score_norm'] * df['age_norm']
    
    # 6. Bias Term
    df['bias'] = 1.0
    
    feature_cols = ['bias', 'score_norm', 'week_norm', 'age_norm', 'prev_vote_share', 
                    'score_sq', 'age_sq', 'score_x_age'] + [f'ind_{ind}' for ind in top_industries]
                    
    return df, feature_cols

def r2_score_manual(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-8))

def main():
    print("Loading optimized estimates from Q1...")
    input_path = 'e:/美赛/Q1_estimated_fan_votes_optimized.csv'
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    df = pd.read_csv(input_path)
    
    # Prepare Data
    print("Extracting features (Manual Numpy Implementation)...")
    df_ml, feature_cols = prepare_features(df)
    
    # Clean NaNs in features
    for col in feature_cols:
        df_ml[col] = df_ml[col].fillna(0)
        
    X = df_ml[feature_cols].values
    y = df_ml['est_vote_share'].values
    
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Data shape: {X.shape}")
    
    # Train-Test Split (Manual)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split_idx = int(len(X) * 0.8)
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # Model Training (Ridge Regression)
    print("Training Ridge Regression Model...")
    model = SimpleLinearRegression(regularization=0.1)
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    r2_train = r2_score_manual(y_train, y_pred_train)
    r2_test = r2_score_manual(y_test, y_pred_test)
    
    print(f"Train R2 Score: {r2_train:.4f}")
    print(f"Test R2 Score:  {r2_test:.4f}")
    
    # Full Prediction
    y_pred_full = model.predict(X)
    df_ml['ml_pred_vote_share'] = y_pred_full
    
    # Feature Importance (Weights)
    print("\n--- Feature Importance (Regression Weights) ---")
    weights = model.weights
    sorted_idx = np.argsort(np.abs(weights))[::-1]
    for i in sorted_idx:
        print(f"{feature_cols[i]}: {weights[i]:.4f}")
        
    # Save Enhanced Results
    output_path = 'e:/美赛/Q1_enhanced_ml_estimates.csv'
    df_ml.to_csv(output_path, index=False)
    print(f"\nEnhanced estimates saved to {output_path}")

if __name__ == "__main__":
    main()
