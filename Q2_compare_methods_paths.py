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
OUT_CONTROVERSY = "Q2_controversy_analysis.csv"

def rankdata_min(a):
    """
    Custom rankdata function using 'min' method (1 for highest score).
    """
    a = np.array(a)
    n = len(a)
    sort_indices = np.argsort(-a) # Descending sort
    ranks = np.empty(n, dtype=int)
    
    current_rank = 1
    for i in range(n):
        if i > 0 and a[sort_indices[i]] == a[sort_indices[i-1]]:
            ranks[sort_indices[i]] = ranks[sort_indices[i-1]]
        else:
            ranks[sort_indices[i]] = i + 1
            
    return ranks

def calculate_metrics(df_week, method_name, judge_weight=0.5):
    """
    Calculate combined metrics based on method.
    Returns sorted dataframe with 'is_eliminated_sim' flag.
    Standardized Score: Lower is Better for Ranking, Higher is Better for Percentage.
    To unify: We will output 'final_score' where Higher is Better.
    """
    scores = df_week['score'].values
    votes = df_week['est_vote_share'].values
    n = len(scores)
    
    # 1. Ranking Method (Standardized)
    # Original: Total Rank = JudgeRank + FanRank. Lowest is Best.
    # Standardized Score (0-1, Higher is Better): 1 - (TotalRank / (2*N))
    if method_name == 'Ranking':
        judge_ranks = rankdata_min(-scores)
        fan_ranks = rankdata_min(-votes)
        total_rank = (judge_ranks * judge_weight * 2) + (fan_ranks * (1-judge_weight) * 2)
        # Invert so higher is better
        final_metric = -total_rank 
        
    # 2. Percentage Method (Standardized)
    # Original: Combined % = 0.5 * Score% + 0.5 * Vote%. Highest is Best.
    elif method_name == 'Percentage':
        if np.sum(scores) > 0:
            score_pct = scores / np.sum(scores)
        else:
            score_pct = np.zeros(n)
        
        final_metric = (score_pct * judge_weight) + (votes * (1-judge_weight))
        
    df_week = df_week.copy()
    df_week['sim_metric'] = final_metric
    
    # Sort descending (higher metric is better)
    df_week = df_week.sort_values('sim_metric', ascending=False).reset_index(drop=True)
    
    # Mark last place as eliminated
    df_week['sim_eliminated'] = False
    if len(df_week) > 0:
        df_week.loc[len(df_week)-1, 'sim_eliminated'] = True
        
    return df_week

def simulate_judge_save(df_week, method_name, judge_weight=0.5):
    """
    Simulate 'Judge Save' mechanism: Bottom 2 face judges.
    Judges save the one with higher judge score.
    """
    # First, calculate metrics to get Bottom 2
    scores = df_week['score'].values
    votes = df_week['est_vote_share'].values
    n = len(scores)
    
    if method_name == 'Ranking':
        judge_ranks = rankdata_min(-scores)
        fan_ranks = rankdata_min(-votes)
        total_rank = (judge_ranks * judge_weight * 2) + (fan_ranks * (1-judge_weight) * 2)
        final_metric = -total_rank
    elif method_name == 'Percentage':
        if np.sum(scores) > 0:
            score_pct = scores / np.sum(scores)
        else:
            score_pct = np.zeros(n)
        final_metric = (score_pct * judge_weight) + (votes * (1-judge_weight))
        
    df_week = df_week.copy()
    df_week['sim_metric'] = final_metric
    
    # Sort descending
    df_sorted = df_week.sort_values('sim_metric', ascending=False).reset_index(drop=True)
    
    if len(df_sorted) < 2:
        return df_sorted # Can't do bottom 2
        
    # Identify Bottom 2
    bottom_2 = df_sorted.tail(2)
    p1 = bottom_2.iloc[0] # Second to last
    p2 = bottom_2.iloc[1] # Last
    
    eliminated_idx = -1
    
    # Judge Save Logic: Higher Judge Score Stays
    if p1['score'] > p2['score']:
        eliminated_idx = df_sorted.index[-1] # p2 eliminated
    elif p2['score'] > p1['score']:
        eliminated_idx = df_sorted.index[-2] # p1 eliminated
    else:
        # Tie on judge score -> Fallback to original metric
        eliminated_idx = df_sorted.index[-1]
        
    df_sorted['sim_eliminated'] = False
    df_sorted.loc[eliminated_idx, 'sim_eliminated'] = True
    
    return df_sorted

def analyze_controversy(df, controversial_names):
    """
    Analyze specific controversial participants.
    """
    results = []
    for name in controversial_names:
        p_data = df[df['name'] == name]
        for _, row in p_data.iterrows():
            # Re-simulate with both methods
            # Need context of the whole week
            week_df = df[(df['season'] == row['season']) & (df['week'] == row['week'])]
            
            # 1. Ranking
            res_rank = calculate_metrics(week_df, 'Ranking')
            is_elim_rank = res_rank[res_rank['name'] == name]['sim_eliminated'].values[0]
            
            # 2. Percentage
            res_pct = calculate_metrics(week_df, 'Percentage')
            is_elim_pct = res_pct[res_pct['name'] == name]['sim_eliminated'].values[0]
            
            # 3. Judge Save
            res_save = simulate_judge_save(week_df, 'Ranking') # Base on Ranking
            is_elim_save = res_save[res_save['name'] == name]['sim_eliminated'].values[0]
            
            results.append({
                'name': name,
                'season': row['season'],
                'week': row['week'],
                'actual_status': row['status'],
                'elim_ranking': is_elim_rank,
                'elim_percentage': is_elim_pct,
                'elim_judge_save': is_elim_save
            })
    return pd.DataFrame(results)

def main():
    # Load optimized estimates
    df = pd.read_csv(DATA_PATH)
    
    # 1. Cross-Season Comparison
    # Calculate Disagreement Rate between Ranking and Percentage
    disagreements = 0
    total_weeks = 0
    
    bias_results = []
    
    seasons = df['season'].unique()
    for s in seasons:
        s_data = df[df['season'] == s]
        weeks = s_data['week'].unique()
        
        for w in weeks:
            w_data = s_data[s_data['week'] == w]
            if len(w_data) <= 1: continue
            
            # Run both methods
            res_rank = calculate_metrics(w_data, 'Ranking')
            res_pct = calculate_metrics(w_data, 'Percentage')
            
            elim_rank = res_rank[res_rank['sim_eliminated']]['name'].values
            elim_pct = res_pct[res_pct['sim_eliminated']]['name'].values
            
            if len(elim_rank) > 0 and len(elim_pct) > 0:
                if elim_rank[0] != elim_pct[0]:
                    disagreements += 1
            
            total_weeks += 1
            
            # Bias Analysis
            # Define "Judge Favored": Top 50% Judge, Bottom 50% Fan
            # Define "Fan Favored": Top 50% Fan, Bottom 50% Judge
            n = len(w_data)
            w_data = w_data.copy()
            w_data['judge_rank'] = rankdata_min(-w_data['score'])
            w_data['fan_rank'] = rankdata_min(-w_data['est_vote_share'])
            
            judge_favored = w_data[(w_data['judge_rank'] <= n/2) & (w_data['fan_rank'] > n/2)]
            fan_favored = w_data[(w_data['fan_rank'] <= n/2) & (w_data['judge_rank'] > n/2)]
            
            for _, p in judge_favored.iterrows():
                # Check survival in Ranking vs Percentage
                survived_rank = not (p['name'] in elim_rank)
                survived_pct = not (p['name'] in elim_pct)
                bias_results.append({'type': 'Judge_Favored', 'survived_rank': survived_rank, 'survived_pct': survived_pct})
                
            for _, p in fan_favored.iterrows():
                survived_rank = not (p['name'] in elim_rank)
                survived_pct = not (p['name'] in elim_pct)
                bias_results.append({'type': 'Fan_Favored', 'survived_rank': survived_rank, 'survived_pct': survived_pct})

    print(f"Total Weeks: {total_weeks}")
    print(f"Disagreement Rate: {disagreements/total_weeks:.2%}")
    
    bias_df = pd.DataFrame(bias_results)
    print("\nBias Analysis (Survival Rate):")
    print(bias_df.groupby('type')[['survived_rank', 'survived_pct']].mean())
    
    # 2. Controversy Analysis
    controversial = ['Jerry Rice', 'Billy Ray Cyrus', 'Bristol Palin', 'Bobby Bones']
    # Filter only if they exist in dataset
    exist_names = df['name'].unique()
    target_names = [n for n in controversial if n in exist_names]
    
    if len(target_names) > 0:
        contra_res = analyze_controversy(df, target_names)
        Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
        out_path = str(Path(OUT_DIR) / OUT_CONTROVERSY)
        contra_res.to_csv(out_path, index=False)
        print("\nControversy analysis saved to Q2_controversy_analysis.csv")
        print(contra_res.groupby('name')[['elim_ranking', 'elim_percentage', 'elim_judge_save']].sum())

if __name__ == "__main__":
    main()
