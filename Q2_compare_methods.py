import pandas as pd
import numpy as np
import os
from pathlib import Path

# ==========================================
# Path Configuration (路径配置)
# ==========================================
Q1_OUTPUT_PATH = Path("Q1_estimated_fan_votes_optimized.csv")
# Q1_OUTPUT_PATH = Path(r"C:\_Am\mcm_outputs\Q1_estimated_fan_votes_optimized.csv")

Q2_OUTPUT_PATH = Path("Q2_method_counterfactuals.csv")
# Q2_OUTPUT_PATH = Path(r"C:\_Am\mcm_outputs\Q2_method_counterfactuals.csv")

# Helper function for ranking (Min method: 1, 2, 2, 4)
def rankdata_min(a):
    """
    Returns the rank of data, using 1-based indexing.
    Ties are assigned the minimum rank (e.g., 1, 2, 2, 4).
    """
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

def simulate_methods(df):
    """
    Simulates Ranking Method vs Percentage Method for all weeks.
    Returns a comparison dataframe and summary stats.
    """
    results = []
    
    # Group by Season and Week
    grouped = df.groupby(['season', 'week'])
    
    disagreement_count = 0
    total_weeks = 0
    
    # Stats for Bias Analysis
    # "Fan Favorite": High Vote Rank (Top 50%), Low Score Rank (Bottom 50%)
    # "Judge Favorite": High Score Rank (Top 50%), Low Vote Rank (Bottom 50%)
    method_bias = {
        'Ranking': {'saved_fan_fav': 0, 'saved_judge_fav': 0, 'total_fan_fav': 0, 'total_judge_fav': 0},
        'Percentage': {'saved_fan_fav': 0, 'saved_judge_fav': 0, 'total_fan_fav': 0, 'total_judge_fav': 0}
    }

    print("Starting Counterfactual Simulation...")
    
    for (season, week), group in grouped:
        # Skip finals or small groups
        if len(group) <= 2:
            continue
            
        total_weeks += 1
        
        names = group['name'].values
        scores = group['score'].values
        votes = group['est_vote_share'].values
        actual_status = group['status'].values
        
        n = len(names)
        
        # --- Method 1: Ranking Method (Sum of Ranks) ---
        # Note: In DWTS, lowest rank sum is BEST. Rank 1 is best.
        # Scores: Higher is better -> Rank 1 is highest score.
        # Votes: Higher is better -> Rank 1 is highest vote.
        judge_ranks = rankdata_min(-scores) # 1 is best
        fan_ranks = rankdata_min(-votes)    # 1 is best
        
        ranking_sum = judge_ranks + fan_ranks
        # Tie-breaker: Usually Fan Vote decides. 
        # So we add a small fraction of fan rank to break ties in favor of fan vote
        # (Lower metric is better)
        ranking_metric = ranking_sum + (fan_ranks / 1000.0)
        
        # Bottom 1 in Ranking Method (Highest metric value)
        ranking_elim_idx = np.argmax(ranking_metric)
        ranking_elim_name = names[ranking_elim_idx]
        
        # --- Method 2: Percentage Method (50/50 Split) ---
        # Score %: score / total_score
        # Vote %: vote_share (already %)
        # Total %: score% + vote% (Higher is better)
        total_score = np.sum(scores)
        if total_score == 0: total_score = 1 # Avoid div/0
        
        score_pct = scores / total_score
        vote_pct = votes # Assuming it sums to 1 or close
        
        # Note: Sometimes they re-normalize vote_pct to sum to 100% within the group
        if np.sum(vote_pct) > 0:
            vote_pct = vote_pct / np.sum(vote_pct)
            
        combined_pct = 0.5 * score_pct + 0.5 * vote_pct
        
        # Bottom 1 in Percentage Method (Lowest metric value)
        pct_elim_idx = np.argmin(combined_pct)
        pct_elim_name = names[pct_elim_idx]
        
        # --- Comparison ---
        disagree = (ranking_elim_name != pct_elim_name)
        if disagree:
            disagreement_count += 1
            
        # --- Bias Analysis ---
        # Define favorites relative to this group
        median_score_rank = (n + 1) / 2
        median_vote_rank = (n + 1) / 2
        
        for i in range(n):
            is_elim_ranking = (i == ranking_elim_idx)
            is_elim_pct = (i == pct_elim_idx)
            
            # Fan Favorite: Good votes (Rank < median), Bad scores (Rank > median)
            # Note: Rank 1 is best. So Rank < median is "Top half".
            is_fan_fav = (fan_ranks[i] < median_vote_rank) and (judge_ranks[i] > median_score_rank)
            
            # Judge Favorite: Good scores, Bad votes
            is_judge_fav = (judge_ranks[i] < median_score_rank) and (fan_ranks[i] > median_vote_rank)
            
            if is_fan_fav:
                method_bias['Ranking']['total_fan_fav'] += 1
                method_bias['Percentage']['total_fan_fav'] += 1
                if not is_elim_ranking: method_bias['Ranking']['saved_fan_fav'] += 1
                if not is_elim_pct: method_bias['Percentage']['saved_fan_fav'] += 1
                
            if is_judge_fav:
                method_bias['Ranking']['total_judge_fav'] += 1
                method_bias['Percentage']['total_judge_fav'] += 1
                if not is_elim_ranking: method_bias['Ranking']['saved_judge_fav'] += 1
                if not is_elim_pct: method_bias['Percentage']['saved_judge_fav'] += 1

        results.append({
            'season': season,
            'week': week,
            'ranking_eliminated': ranking_elim_name,
            'pct_eliminated': pct_elim_name,
            'disagree': disagree,
            'candidates': n
        })

    print(f"Simulation Complete. Total Weeks: {total_weeks}")
    print(f"Disagreement Count: {disagreement_count} ({disagreement_count/total_weeks:.2%})")
    
    return pd.DataFrame(results), method_bias

def analyze_controversy(df, comparison_results):
    """
    Deep dive into specific controversial figures.
    """
    targets = ['Bobby Bones', 'Jerry Rice', 'Bristol Palin', 'Billy Ray Cyrus']
    print("\n--- Controversial Case Study (Counterfactuals) ---")
    
    for name in targets:
        # Find weeks where this person was at risk
        person_data = df[df['name'] == name]
        if len(person_data) == 0:
            continue
            
        print(f"\nAnalyzing {name} (Season {person_data['season'].iloc[0]}):")
        
        # Join with comparison results
        for idx, row in person_data.iterrows():
            s, w = row['season'], row['week']
            comp = comparison_results[(comparison_results['season'] == s) & (comparison_results['week'] == w)]
            if len(comp) == 0: continue
            
            r_elim = comp.iloc[0]['ranking_eliminated']
            p_elim = comp.iloc[0]['pct_eliminated']
            
            if r_elim == name or p_elim == name:
                print(f"  Week {w}: Ranking elim -> {r_elim}, Percentage elim -> {p_elim}")
                if r_elim != p_elim:
                    print(f"    *** METHOD MATTERS HERE! ***")

def main():
    file_path = Q1_OUTPUT_PATH
    df = load_data(file_path)
    if df is None: return
    
    # Run Simulation
    comp_df, bias_stats = simulate_methods(df)
    
    # Save Results
    if Q2_OUTPUT_PATH.parent != Path("."):
        os.makedirs(Q2_OUTPUT_PATH.parent, exist_ok=True)
    comp_df.to_csv(Q2_OUTPUT_PATH, index=False)
    
    # Print Summary
    print("\n--- Method Bias Analysis ---")
    
    # Calculate Save Rates
    def get_rate(bias_dict, method, type_key):
        total = bias_dict[method][f'total_{type_key}']
        saved = bias_dict[method][f'saved_{type_key}']
        return saved / total if total > 0 else 0, saved, total

    r_fan_rate, r_fan_saved, r_fan_total = get_rate(bias_stats, 'Ranking', 'fan_fav')
    p_fan_rate, p_fan_saved, p_fan_total = get_rate(bias_stats, 'Percentage', 'fan_fav')
    
    r_judge_rate, r_judge_saved, r_judge_total = get_rate(bias_stats, 'Ranking', 'judge_fav')
    p_judge_rate, p_judge_saved, p_judge_total = get_rate(bias_stats, 'Percentage', 'judge_fav')
    
    print(f"Fan Favorites (Low Score, High Vote):")
    print(f"  Ranking Method Save Rate:    {r_fan_rate:.1%} ({r_fan_saved}/{r_fan_total})")
    print(f"  Percentage Method Save Rate: {p_fan_rate:.1%} ({p_fan_saved}/{p_fan_total})")
    print(f"  -> Winner: {'Ranking' if r_fan_rate > p_fan_rate else 'Percentage'}")
    
    print(f"\nJudge Favorites (High Score, Low Vote):")
    print(f"  Ranking Method Save Rate:    {r_judge_rate:.1%} ({r_judge_saved}/{r_judge_total})")
    print(f"  Percentage Method Save Rate: {p_judge_rate:.1%} ({p_judge_saved}/{p_judge_total})")
    print(f"  -> Winner: {'Ranking' if r_judge_rate > p_judge_rate else 'Percentage'}")
    
    # Run Controversy Analysis
    analyze_controversy(df, comp_df)

if __name__ == "__main__":
    main()
