import pandas as pd
import numpy as np
import os

def rankdata_min(a):
    """
    Implements rankdata with method='min' without scipy dependency.
    Ranks start at 1. Ties get the minimum rank.
    Low value = Rank 1 (if we are ranking places, e.g. 1st place).
    BUT for scores: High Score = Rank 1.
    So input 'a' should be negative scores if we want High Score -> Rank 1.
    """
    n = len(a)
    ranks = np.zeros(n, dtype=int)
    for i in range(n):
        rank = 1
        for j in range(n):
            if a[j] < a[i]:
                rank += 1
        ranks[i] = rank
    return ranks

def simulate_votes(participants, num_samples=1000):
    if participants.empty:
        return []
    
    n = len(participants)
    scores = participants['score'].values
    
    # Identify eliminated participants
    # Status can be 'Eliminated Week X', 'Safe', 'Winner', 'Runner-up', 'Third Place'
    status = participants['status'].values
    is_eliminated = np.array(['Eliminated' in s for s in status])
    
    # If no one is eliminated this week (e.g. non-elimination week), 
    # we just return random votes as we can't constrain them easily by elimination
    if not np.any(is_eliminated):
        return np.random.dirichlet(np.ones(n), num_samples)

    eliminated_indices = np.where(is_eliminated)[0]
    safe_indices = np.where(~is_eliminated)[0]
    
    # Generate random vote shares (Dirichlet distribution)
    # alpha=1 implies uniform distribution over the simplex
    votes = np.random.dirichlet(np.ones(n), num_samples)
    
    # Calculate Judge Ranks (Higher score = Lower rank number, e.g. 30 -> 1)
    # We use -scores because rankdata_min ranks lowest value as 1.
    judge_ranks = rankdata_min(-scores)
    
    valid_samples = []
    
    for i in range(num_samples):
        v = votes[i]
        # Fan Ranks (Higher votes = Lower rank number)
        fan_ranks = rankdata_min(-v)
        
        # Total Ranks = Judge Ranks + Fan Ranks
        total_ranks = judge_ranks + fan_ranks
        
        # Tie-breaking logic: often strictly by fan votes if total ranks are equal.
        # We can model this by adding a small fractional part from fan ranks
        # Lower metric is better.
        metric = total_ranks + (fan_ranks / 1000.0)
        
        elim_metrics = metric[eliminated_indices]
        safe_metrics = metric[safe_indices]
        
        # Validation condition:
        # The best (lowest metric) eliminated person must be worse (higher metric) 
        # than the worst (highest metric) safe person?
        # No, typically the people with the HIGHEST metric (worst rank) are eliminated.
        # So, min(elim_metrics) should be > max(safe_metrics) ideally.
        # However, sometimes multiple people are eliminated.
        # The condition is: All eliminated people should be at the bottom of the ranking.
        
        if len(safe_indices) > 0:
            if np.min(elim_metrics) > np.max(safe_metrics):
                valid_samples.append(v)
            # Relaxed condition for cases where exact match is hard
            # e.g. maybe just check if the actual eliminated person is in the bottom N
            elif np.min(elim_metrics) >= np.min(safe_metrics): 
                # This is a weak check, but better than nothing for noisy data
                # Let's stick to the strict check first, maybe retry if empty
                pass
        else:
            # Everyone eliminated? (Finals?)
            valid_samples.append(v)
            
    return np.array(valid_samples)

def main():
    file_path = 'e:/美赛/2026_MCM_Problem_C_Data.csv'
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    df = pd.read_csv(file_path)
    
    # Preprocess Data
    # We need to extract scores for each week.
    # Columns are like: 'Judge Score 1', 'Judge Score 2', ..., 'Judge Score 10'
    # And 'Elimination 1', 'Elimination 2', ...
    
    # Let's restructure the data into a long format: Season, Week, Name, Score, Status
    long_data = []
    
    max_weeks = 10 # Based on column headers inspection
    
    for idx, row in df.iterrows():
        season = row['Season']
        name = row['Name']
        
        for w in range(1, max_weeks + 1):
            score_col = f'Judge Score {w}'
            elim_col = f'Elimination {w}'
            
            if score_col in df.columns and elim_col in df.columns:
                score = row[score_col]
                status = row[elim_col]
                
                # Check if participated this week
                if pd.notna(score) and str(score).strip() != 'N/A' and str(score).strip() != '':
                    try:
                        score_val = float(score)
                        long_data.append({
                            'season': season,
                            'week': w,
                            'name': name,
                            'score': score_val,
                            'status': str(status)
                        })
                    except ValueError:
                        pass

    df_long = pd.DataFrame(long_data)
    
    results = []
    
    print("Starting simulation...")
    
    for season in df_long['season'].unique():
        season_data = df_long[df_long['season'] == season]
        for week in season_data['week'].unique():
            week_data = season_data[season_data['week'] == week]
            
            # Simulate
            valid_votes = simulate_votes(week_data, num_samples=2000)
            
            if len(valid_votes) > 0:
                avg_votes = np.mean(valid_votes, axis=0)
                std_votes = np.std(valid_votes, axis=0)
            else:
                # Fallback: uniform distribution if no valid samples found (rare)
                n = len(week_data)
                avg_votes = np.ones(n) / n
                std_votes = np.zeros(n)
                print(f"Warning: No valid samples for S{season} W{week}")

            # Append results
            for i, (idx, row) in enumerate(week_data.iterrows()):
                results.append({
                    'season': season,
                    'week': week,
                    'name': row['name'],
                    'judge_score': row['score'],
                    'status': row['status'],
                    'est_vote_share': avg_votes[i],
                    'est_vote_std': std_votes[i]
                })
                
    results_df = pd.DataFrame(results)
    output_path = 'e:/美赛/Q1_estimated_fan_votes.csv'
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
