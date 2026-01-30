import pandas as pd
import numpy as np
import os

def rankdata_min(a):
    n = len(a)
    ranks = np.zeros(n, dtype=int)
    for i in range(n):
        rank = 1
        for j in range(n):
            if a[j] < a[i]:
                rank += 1
        ranks[i] = rank
    return ranks

def main():
    # Load estimated votes
    votes_path = 'e:/美赛/Q1_estimated_fan_votes.csv'
    if not os.path.exists(votes_path):
        print("Estimated votes file not found. Run Task 1 first.")
        return

    df = pd.read_csv(votes_path)
    
    comparison_results = []
    
    # Process each week
    for season in df['season'].unique():
        season_df = df[df['season'] == season]
        for week in season_df['week'].unique():
            week_df = season_df[season_df['week'] == week].copy()
            
            # 1. Ranking Method (Original)
            scores = week_df['judge_score'].values
            fan_shares = week_df['est_vote_share'].values
            
            judge_ranks = rankdata_min(-scores)
            fan_ranks = rankdata_min(-fan_shares)
            
            total_ranks = judge_ranks + fan_ranks
            
            # Determine eliminated in Ranking Method (Highest total rank = Worst)
            # Tie-breaker: Fan Votes
            metric_rank = total_ranks + (fan_ranks / 1000.0)
            
            # 2. Percentage Method
            # Judge Pct + Fan Pct
            total_score = np.sum(scores)
            if total_score > 0:
                judge_pct = scores / total_score
            else:
                judge_pct = np.zeros_like(scores)
            
            # Combine 50/50
            total_pct = (judge_pct + fan_shares) / 2.0
            
            # Determine eliminated in Pct Method (Lowest pct = Worst)
            # We rank -total_pct so that Rank 1 is Best (Highest Pct)
            # Actually, let's just use the raw pct values. Lowest is eliminated.
            
            # Identify actual eliminated
            actual_eliminated = week_df[week_df['status'].str.contains('Eliminated', na=False)]['name'].tolist()
            
            # Simulate Elimination (Bottom 1)
            # Rank Method: Max metric_rank is eliminated
            rank_worst_idx = np.argmax(metric_rank)
            rank_eliminated = week_df.iloc[rank_worst_idx]['name']
            
            # Pct Method: Min total_pct is eliminated
            pct_worst_idx = np.argmin(total_pct)
            pct_eliminated = week_df.iloc[pct_worst_idx]['name']
            
            match_rank = (rank_eliminated in actual_eliminated) if actual_eliminated else (rank_eliminated == 'None') # Simplified
            match_pct = (pct_eliminated in actual_eliminated) if actual_eliminated else (pct_eliminated == 'None')
            
            comparison_results.append({
                'season': season,
                'week': week,
                'actual_eliminated': str(actual_eliminated),
                'rank_eliminated': rank_eliminated,
                'pct_eliminated': pct_eliminated,
                'match_rank': match_rank,
                'match_pct': match_pct,
                'names': week_df['name'].values,
                'judge_ranks': judge_ranks,
                'fan_ranks': fan_ranks,
                'total_pct': total_pct
            })
            
    res_df = pd.DataFrame(comparison_results)
    res_df.to_csv('e:/美赛/Q2_method_comparison.csv', index=False)
    print("Comparison saved to e:/美赛/Q2_method_comparison.csv")
    
    # Controversial Cases Analysis
    controversial = [
        ('Jerry Rice', 2),
        ('Billy Ray Cyrus', 4),
        ('Bristol Palin', 11),
        ('Bobby Bones', 27)
    ]
    
    print("\n--- Controversial Cases Analysis ---")
    for name, target_season in controversial:
        print(f"\nAnalyzing {name} (Season {target_season})")
        # Filter for this person
        # Note: Names might need partial matching
        
        # Find the rows in comparison_results
        relevant_rows = res_df[res_df['season'] == target_season]
        
        found = False
        for idx, row in relevant_rows.iterrows():
            names = row['names'] # This is a numpy array inside the dataframe cell? No, it's an object from previous append
            # When loaded from CSV it would be string, but here it is still in memory list if we use the list `comparison_results`
            # But we are iterating `res_df`.
            
            # Let's iterate the original df for details
            pass
            
        # Re-querying the results df is tricky because 'names' is stored as object/string.
        # Let's just iterate the raw week_df logic again for specific cases or use the saved CSV structure
        
        # Simplified: Just print stats from the calculated DataFrames
        person_stats = df[(df['season'] == target_season) & (df['name'] == name)]
        if person_stats.empty:
             # Try partial match
             person_stats = df[(df['season'] == target_season) & (df['name'].str.contains(name, case=False))]
        
        if not person_stats.empty:
            found = True
            for _, p_row in person_stats.iterrows():
                wk = p_row['week']
                # Get context for that week
                week_res = res_df[(res_df['season'] == target_season) & (res_df['week'] == wk)].iloc[0]
                
                # Re-calculate ranks for display
                # Note: In a real app we'd join this properly.
                # Here we just want to see: Did Pct Method eliminate them?
                
                pct_elim = week_res['pct_eliminated']
                rank_elim = week_res['rank_eliminated']
                actual = week_res['actual_eliminated']
                
                if pct_elim == name or rank_elim == name or name in actual:
                    print(f"Week {wk}: Actual Elim: {actual}. Rank Method Elim: {rank_elim}. Pct Method Elim: {pct_elim}.")
                    
                    # Calculate their specific position
                    # We need to reconstruct the week's data to see their exact rank
                    pass
        
        if not found:
            print(f"Could not find data for {name}")

if __name__ == "__main__":
    main()
