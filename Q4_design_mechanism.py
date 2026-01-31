import pandas as pd
import numpy as np
import os
from pathlib import Path

# ==========================================
# Path Configuration (路径配置)
# ==========================================
Q1_OUTPUT_PATH = Path("Q1_estimated_fan_votes_optimized.csv")
# Q1_OUTPUT_PATH = Path(r"C:\_Am\mcm_outputs\Q1_estimated_fan_votes_optimized.csv")

# Helper for ranking
def rankdata_min(a):
    arr = np.array(a)
    sorted_indices = np.argsort(arr)
    ranks = np.empty_like(sorted_indices)
    current_rank = 1
    for i in range(len(sorted_indices)):
        if i > 0 and arr[sorted_indices[i]] != arr[sorted_indices[i-1]]:
            current_rank = i + 1
        ranks[sorted_indices[i]] = current_rank
    return ranks

def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None
    return pd.read_csv(file_path)

def simulate_mechanism(df, weights):
    """
    Simulates the new mechanism with given weights [w_early, w_mid, w_late].
    Returns metrics: Fairness Score, Retention Score.
    """
    w_early, w_mid, w_late = weights
    
    # Define phases
    # Early: Week 1-4, Mid: 5-8, Late: 9+
    
    # We need to simulate week by week, removing people.
    # This is complex because removal changes the next week's pool.
    # For simplification in optimization loop:
    # We check "At Risk" status (Bottom 3) using the new weights.
    # If a High Scorer is in Bottom 3, it's a Fairness Penalty.
    # If a High Voter is in Bottom 3, it's a Retention Penalty.
    
    fairness_penalties = 0
    retention_penalties = 0
    total_checks = 0
    
    grouped = df.groupby(['season', 'week'])
    
    for (season, week), group in grouped:
        if len(group) <= 3: continue
        total_checks += 1
        
        # Determine weight
        if week <= 4: w = w_early
        elif week <= 8: w = w_mid
        else: w = w_late
        
        scores = group['score'].values
        votes = group['est_vote_share'].values
        names = group['name'].values
        
        # Normalize
        s_max = np.max(scores)
        if s_max == 0: s_max = 1
        s_norm = scores / s_max
        
        v_sum = np.sum(votes)
        if v_sum == 0: v_sum = 1
        v_norm = votes / v_sum # Share
        
        # Combined Score
        # New Mechanism: score * w + vote * (1-w)
        combined = s_norm * w + v_norm * (1 - w)
        
        # Identify Bottom 3 (At Risk)
        # combined score: Higher is better. Bottom 3 are lowest.
        sorted_idx = np.argsort(combined) # Ascending
        bottom_3_idx = sorted_idx[:3]
        
        # Check Fairness: Is a Top Scorer (Top 3 in Score) in Bottom 3?
        score_ranks = rankdata_min(-scores) # 1 is best
        top_scorers_idx = np.where(score_ranks <= 3)[0]
        
        for idx in bottom_3_idx:
            if idx in top_scorers_idx:
                fairness_penalties += 1
                
        # Check Retention: Is a Top Voter (Top 3 in Votes) in Bottom 3?
        vote_ranks = rankdata_min(-votes) # 1 is best
        top_voters_idx = np.where(vote_ranks <= 3)[0]
        
        for idx in bottom_3_idx:
            if idx in top_voters_idx:
                retention_penalties += 1
                
    # Normalize metrics (Lower is better)
    fairness_score = fairness_penalties / total_checks if total_checks > 0 else 0
    retention_score = retention_penalties / total_checks if total_checks > 0 else 0
    
    return fairness_score, retention_score

def optimize_weights(df):
    print("Starting Multi-Objective Optimization for Weights...")
    
    best_weights = None
    best_combined_score = float('inf')
    
    results = []
    
    # Grid Search
    # w_range = [0.3, 0.4, 0.5, 0.6, 0.7]
    # Expanded and finer grid for more accurate results
    w_range = np.round(np.arange(0.1, 0.95, 0.05), 2)
    
    count = 0
    for w1 in w_range: # Early
        for w2 in w_range: # Mid
            for w3 in w_range: # Late
                # Constraints: Typically technical importance grows. w1 <= w2 <= w3
                if not (w1 <= w2 <= w3): continue
                
                count += 1
                if count % 100 == 0:
                    print(f"Testing config {count}: [{w1}, {w2}, {w3}]", flush=True)
                
                weights = [w1, w2, w3]
                f_score, r_score = simulate_mechanism(df, weights)
                
                # Composite Objective: Minimize (Fairness Penalty + Retention Penalty)
                # We might weight Fairness higher for "Integrity"
                final_score = 1.0 * f_score + 0.8 * r_score
                
                results.append({
                    'Weights': weights,
                    'Fairness_Penalty': f_score,
                    'Retention_Penalty': r_score,
                    'Final_Score': final_score
                })
                
                if final_score < best_combined_score:
                    best_combined_score = final_score
                    best_weights = weights
                    
    results_df = pd.DataFrame(results).sort_values('Final_Score')
    
    # Save detailed optimization results
    opt_output_path = Path("Q4_optimization_detailed_results.csv")
    if opt_output_path.parent != Path("."):
        os.makedirs(opt_output_path.parent, exist_ok=True)
    results_df.to_csv(opt_output_path, index=False)
    print(f"Saved all {len(results_df)} tested configurations to {opt_output_path}")

    print("\nTop 10 Weight Configurations:")
    print(results_df.head(10))
    
    print(f"\nBest Weights Found: {best_weights}")
    return best_weights

def main():
    file_path = Q1_OUTPUT_PATH
    df = load_data(file_path)
    if df is None: return
    
    best_w = optimize_weights(df)
    
    print("\n--- Recommendation ---")
    print(f"Adopt Dynamic Weighting: Early={best_w[0]}, Mid={best_w[1]}, Late={best_w[2]}")
    print("Rationale: This combination minimizes the risk of Top Scorers AND Top Vote Getters falling into the Bottom 3.")
    print("Implementation: Use 'Judge Save' on the Bottom 3 to further reduce the Fairness Penalty to near zero.")

if __name__ == "__main__":
    main()
