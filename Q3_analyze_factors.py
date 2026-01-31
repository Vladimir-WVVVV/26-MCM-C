import pandas as pd
import numpy as np
import os

# ==========================================
# Regression Tool
# ==========================================

class SimpleLinearRegression:
    def __init__(self):
        self.weights = None
        self.feature_names = None
        
    def fit(self, X, y, feature_names=None):
        # Add Bias term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # w = (X^T X)^-1 X^T y
        try:
            self.weights = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        except:
            self.weights = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
            
        if feature_names:
            self.feature_names = ['Intercept'] + feature_names
        else:
            self.feature_names = ['Intercept'] + [f'x{i}' for i in range(X.shape[1])]
            
    def get_coefficients(self):
        return dict(zip(self.feature_names, self.weights))

# ==========================================
# Analysis Logic
# ==========================================

def prepare_data(df):
    """
    Prepare X (Factors) and Y (Outcomes)
    """
    df = df.copy()
    
    # 1. Clean Age
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['age'] = df['age'].fillna(df['age'].median())
    
    # 2. Industry Dummies (Top 5 + Other)
    top_inds = ['athlete', 'actor', 'singer', 'model', 'comedian']
    ind_cols = []
    for ind in top_inds:
        col_name = f'Ind_{ind}'
        df[col_name] = df['industry'].astype(str).apply(lambda x: 1 if ind in x.lower() else 0)
        ind_cols.append(col_name)
        
    # 3. Normalize numerical inputs for comparable coefficients
    df['age_std'] = (df['age'] - df['age'].mean()) / df['age'].std()
    
    feature_cols = ['age_std'] + ind_cols
    
    # Outcomes
    # Normalize outcomes to compare magnitude of impact
    df['score_std'] = (df['score'] - df['score'].mean()) / df['score'].std()
    
    # Use ML Enhanced Vote Share if available, else standard estimate
    if 'ml_pred_vote_share' in df.columns:
        vote_col = 'ml_pred_vote_share'
    else:
        vote_col = 'est_vote_share'
        
    df['vote_std'] = (df[vote_col] - df[vote_col].mean()) / df[vote_col].std()
    
    return df, feature_cols

def main():
    print("Loading Enhanced Estimates...")
    path = 'e:/美赛/Q1_enhanced_ml_estimates.csv'
    if not os.path.exists(path):
        print("Enhanced estimates not found, falling back to optimized...")
        path = 'e:/美赛/Q1_estimated_fan_votes_optimized.csv'
    
    df = pd.read_csv(path)
    
    print(f"Data Loaded: {len(df)} rows")
    
    # Prepare Data
    df_reg, features = prepare_data(df)
    
    X = df_reg[features].values
    y_score = df_reg['score_std'].values
    y_vote = df_reg['vote_std'].values
    
    # Model 1: Factors affecting Judge Score
    print("\n--- Model 1: Factors affecting Judge Score ---")
    reg_score = SimpleLinearRegression()
    reg_score.fit(X, y_score, features)
    coeffs_score = reg_score.get_coefficients()
    for k, v in coeffs_score.items():
        print(f"{k}: {v:.4f}")
        
    # Model 2: Factors affecting Fan Votes
    print("\n--- Model 2: Factors affecting Fan Votes ---")
    reg_vote = SimpleLinearRegression()
    reg_vote.fit(X, y_vote, features)
    coeffs_vote = reg_vote.get_coefficients()
    for k, v in coeffs_vote.items():
        print(f"{k}: {v:.4f}")
        
    # Comparison
    print("\n--- Comparative Analysis ---")
    print(f"{'Factor':<15} | {'Judge Impact':<12} | {'Fan Impact':<12} | {'Difference':<12}")
    print("-" * 55)
    for feat in features:
        j_imp = coeffs_score.get(feat, 0)
        f_imp = coeffs_vote.get(feat, 0)
        diff = f_imp - j_imp
        print(f"{feat:<15} | {j_imp:>.4f}       | {f_imp:>.4f}     | {diff:>.4f}")
        
    # Conclusion
    print("\n--- Key Conclusions ---")
    # Age
    age_diff = coeffs_vote['age_std'] - coeffs_score['age_std']
    if age_diff > 0.1:
        print("- Age helps Fan Votes MORE than Judge Scores.")
    elif age_diff < -0.1:
        print("- Age hurts Fan Votes MORE than Judge Scores (or helps Judges more).")
    else:
        print("- Age has similar impact on both.")
        
    # Industry
    best_ind_vote = max([(k, v) for k, v in coeffs_vote.items() if 'Ind_' in k], key=lambda x: x[1])
    print(f"- Best Industry for Fan Votes: {best_ind_vote[0]} (Coef: {best_ind_vote[1]:.4f})")
    
    best_ind_score = max([(k, v) for k, v in coeffs_score.items() if 'Ind_' in k], key=lambda x: x[1])
    print(f"- Best Industry for Judge Score: {best_ind_score[0]} (Coef: {best_ind_score[1]:.4f})")
    
    # Save results
    df_reg.to_csv('e:/美赛/Q3_factor_regression_results.csv', index=False)
    print("\nDetailed regression data saved to Q3_factor_regression_results.csv")

if __name__ == "__main__":
    main()
