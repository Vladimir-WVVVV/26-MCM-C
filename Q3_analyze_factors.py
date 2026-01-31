import pandas as pd
import numpy as np
import os

def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None
    return pd.read_csv(file_path)

def extract_features(df):
    # Basic cleaning
    df = df.dropna(subset=['score', 'est_vote_share']).copy()
    
    # Feature Engineering
    df['is_athlete'] = df['industry'].apply(lambda x: 1 if 'athlete' in str(x).lower() or 'football' in str(x).lower() else 0)
    df['is_actor'] = df['industry'].apply(lambda x: 1 if 'actor' in str(x).lower() or 'actress' in str(x).lower() else 0)
    df['is_singer'] = df['industry'].apply(lambda x: 1 if 'singer' in str(x).lower() or 'music' in str(x).lower() else 0)
    df['is_reality'] = df['industry'].apply(lambda x: 1 if 'reality' in str(x).lower() else 0)
    
    # Age handling
    def parse_age(x):
        try:
            return float(x)
        except:
            return 35.0 # median
    df['age_val'] = df['age'].apply(parse_age)
    df['age_sq'] = df['age_val'] ** 2 # Capture non-linearity
    
    # Normalize targets
    df['score_norm'] = df['score'] / df.groupby('season')['score'].transform('max')
    df['vote_norm'] = df['est_vote_share'] # Already share
    
    feature_cols = ['age_val', 'age_sq', 'is_athlete', 'is_actor', 'is_singer', 'is_reality', 'season']
    return df, feature_cols

class LinearRegressionNumPy:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = 0.0
        
    def fit(self, X, y):
        # Add intercept column
        X_b = np.c_[np.ones((len(X), 1)), X]
        try:
            # Normal Equation: theta = (X.T * X)^-1 * X.T * y
            # Use pinv for stability
            theta_best = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
            self.intercept_ = theta_best[0]
            self.coef_ = theta_best[1:]
        except Exception as e:
            print(f"Regression Error: {e}")
            self.coef_ = np.zeros(X.shape[1])
            
    def predict(self, X):
        X_b = np.c_[np.ones((len(X), 1)), X]
        return X_b.dot(np.r_[self.intercept_, self.coef_])

def analyze_hybrid(df, features, target_col, target_name):
    print(f"\n--- Analyzing Factors for {target_name} ---")
    
    # Standardize features for comparable coefficients (Feature Importance)
    X = df[features].values
    y = df[target_col].values
    
    # Debug
    print(f"X shape: {X.shape}")
    if np.isnan(X).any():
        print("NaNs found in X. Filling with 0.")
        X = np.nan_to_num(X)
    
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-8
    X_scaled = (X - X_mean) / X_std
    
    # Fit OLS
    lr = LinearRegressionNumPy()
    lr.fit(X_scaled, y)
    
    # Display Results
    results = []
    for idx, feat in enumerate(features):
        results.append({
            'Feature': feat,
            'Standardized_Coef (Importance)': lr.coef_[idx],
            'Direction': 'Positive' if lr.coef_[idx] > 0 else 'Negative'
        })
        
    res_df = pd.DataFrame(results).sort_values('Standardized_Coef (Importance)', key=abs, ascending=False)
    print(res_df)
    
    return res_df

def main():
    file_path = 'Q1_estimated_fan_votes_optimized.csv'
    df = load_data(file_path)
    if df is None: return
    
    df_clean, features = extract_features(df)
    
    # Analyze Judge Scores
    analyze_hybrid(df_clean, features, 'score_norm', 'Judge Scores')
    
    # Analyze Fan Votes
    analyze_hybrid(df_clean, features, 'vote_norm', 'Fan Votes')
    
    print("\n--- Insight Summary ---")
    print("1. Compare 'age_val' vs 'age_sq' to see if impact is linear or curved.")
    print("2. Compare 'is_athlete' coefficient magnitude between Judges and Fans.")

if __name__ == "__main__":
    main()
