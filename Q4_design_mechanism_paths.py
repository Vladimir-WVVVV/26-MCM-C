import pandas as pd
import numpy as np


from pathlib import Path

# ==========================================
# 路径配置（请按你的电脑修改）
# ==========================================
# 输入：Q1 的估计结果
DATA_PATH = r"C:\_Am\mcm_outputs\Q1_estimated_fan_votes_optimized.csv"
# 输出目录
OUT_DIR = r"C:\_Am\mcm_outputs"
# 输出文件名
OUT_FILE = "Q4_simulation_results.csv"

def rankdata_min(a):
    a = np.array(a)
    n = len(a)
    sort_indices = np.argsort(a) # Ascending
    ranks = np.empty(n, dtype=int)
    current_rank = 1
    for i in range(n):
        if i > 0 and a[sort_indices[i]] == a[sort_indices[i-1]]:
            # Same rank as previous
            ranks[sort_indices[i]] = ranks[sort_indices[i-1]]
        else:
            # Rank is i+1 (1-based)
            ranks[sort_indices[i]] = i + 1
    return ranks

def calculate_new_system_score(df_week, week_num, max_weeks):
    """
    Proposed Mechanism: Dynamic Weight + Two-Stage Elimination
    
    Stage 1: Dynamic Weight
    - Early (1-4): w=0.4 (Fan focus)
    - Mid (5-8): w=0.5 (Balanced)
    - Late (>8): w=0.6 (Technique focus)
    
    Standardization:
    - Score: (Score - 1) / 9  (Assuming max score per judge is 10, total avg score used here)
    - Vote: Vote Share (Already 0-1) -> Scaled to Max? No, just use share or relative to max.
      Let's use Vote / Max_Vote_in_Week to make it comparable to Score [0,1].
    """
    scores = df_week['score'].values
    votes = df_week['est_vote_share'].values
    names = df_week['name'].values
    n = len(scores)
    
    # Determine Weight w
    if week_num <= 4:
        w = 0.4
    elif week_num <= 8:
        w = 0.5
    else:
        w = 0.6
        
    # Standardize Scores (Assume score is sum of 3 or 4 judges, need to normalize to 0-1)
    # Max possible score: usually 30 or 40. Let's use Min-Max relative to current week to be safe, 
    # or absolute if we know the scale. Let's use Min-Max of the week to handle different judge counts.
    if np.max(scores) > np.min(scores):
        norm_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    else:
        norm_scores = np.zeros(n) + 0.5
        
    # Standardize Votes (Relative to Max Share)
    if np.max(votes) > 0:
        norm_votes = votes / np.max(votes)
    else:
        norm_votes = np.zeros(n)
        
    # Combined Score (Stage 1)
    combined = norm_scores * w + norm_votes * (1 - w)
    
    return combined, norm_scores

def simulate_elimination(df_week, week_num, max_weeks):
    """
    Simulate elimination with new system.
    Stage 2: Bottom 3 -> Technical Safety Score -> Judge Vote
    """
    combined, norm_scores = calculate_new_system_score(df_week, week_num, max_weeks)
    
    df_week = df_week.copy()
    df_week['new_score'] = combined
    df_week['norm_judge_score'] = norm_scores
    
    # Sort descending
    df_sorted = df_week.sort_values('new_score', ascending=False).reset_index(drop=True)
    
    if len(df_sorted) < 3:
        # Just eliminate last
        eliminated = df_sorted.iloc[-1]['name']
        return eliminated, df_sorted
        
    # Stage 2: Bottom 3
    bottom_3 = df_sorted.tail(3).copy()
    
    # Calculate "Technical Safety Score" = Current Judge * 0.7 + Hist Judge Avg * 0.3
    # Simplify: Just use current judge score for now as history needs state tracking
    # Or simulate history proxy
    bottom_3['tech_safe_score'] = bottom_3['norm_judge_score'] 
    
    # 1. Eliminate Lowest Tech Safe Score directly
    bottom_3 = bottom_3.sort_values('tech_safe_score', ascending=True)
    direct_elim = bottom_3.iloc[0] # Lowest tech score
    
    eliminated_name = direct_elim['name']
    return eliminated_name, df_sorted

def main():
    df = pd.read_csv(DATA_PATH)
    
    # Simulation Results
    results = []
    
    # Metrics
    tech_fairness = [] # Rank of eliminated person in Judge Scores (Higher rank eliminated = Unfair)
    fan_survival = [] # Rank of eliminated person in Fan Votes
    controversy_saved = 0
    
    seasons = df['season'].unique()
    for s in seasons:
        s_data = df[df['season'] == s]
        weeks = s_data['week'].unique()
        max_w = np.max(weeks)
        
        for w in weeks:
            w_data = s_data[s_data['week'] == w]
            if len(w_data) <= 3: continue
            
            # Run New System
            elim_name, sorted_df = simulate_elimination(w_data, w, max_w)
            
            # Analyze Fairness
            # Get the eliminated person's ranks
            # Use .values to ensure we don't have index alignment issues
            judge_ranks = rankdata_min(-w_data['score'].values)
            fan_ranks = rankdata_min(-w_data['est_vote_share'].values)
            
            # Find index of eliminated person in the local w_data
            # w_data is a slice, so reset index to match 0..n of ranks arrays
            w_data_reset = w_data.reset_index(drop=True)
            elim_idx_local = w_data_reset[w_data_reset['name'] == elim_name].index[0]
            
            p_row = w_data_reset.iloc[elim_idx_local]
            judge_rank = judge_ranks[elim_idx_local]
            fan_rank = fan_ranks[elim_idx_local]
            
            tech_fairness.append(judge_rank) # We want this to be high (meaning they were bad at tech)
            fan_survival.append(fan_rank)
            
            # Check Controversy (e.g. Jerry Rice)
            if elim_name in ['Jerry Rice', 'Bobby Bones']:
                controversy_saved += 1
                
            results.append({
                'season': s,
                'week': w,
                'eliminated': elim_name,
                'judge_rank': judge_rank,
                'fan_rank': fan_rank,
                'actual_status': p_row['status']
            })
            
    res_df = pd.DataFrame(results)
    
    # Calculate Metrics
    # Fairness: Avg Judge Rank of Eliminated (Closer to N is better, meaning they were actually worst)
    # But N varies. Let's use normalized rank (1 is best, 0 is worst).
    # Actually, simpler: How often do we eliminate someone in the Top 3 Judge Scores?
    bad_eliminations = len(res_df[res_df['judge_rank'] <= 3])
    total_elim = len(res_df)
    
    print("--- New System Evaluation (Dynamic Weight + Bottom 3 Tech Elim) ---")
    print(f"Total Simulations: {total_elim}")
    print(f"Fairness Violation (Eliminated Top 3 Tech): {bad_eliminations} ({bad_eliminations/total_elim:.2%})")
    print(f"Avg Fan Rank of Eliminated: {np.mean(res_df['fan_rank']):.2f} (Higher means less popular)")
    
    # Compare with Old System (Ranking Method) - From Q2 results (approx)
    # In Q2 we found Ranking method protects popular people well.
    # Here, by enforcing Tech Elim on Bottom 3, we might eliminate popular people if they bomb the judges.
    
    # Save
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(OUT_DIR) / OUT_FILE)
    res_df.to_csv(out_path, index=False)
    print("Simulation results saved to Q4_simulation_results.csv")

if __name__ == "__main__":
    main()