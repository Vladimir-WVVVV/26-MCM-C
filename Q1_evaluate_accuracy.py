import pandas as pd
import numpy as np
import os

def rankdata_min(a):
    """
    Standard competition ranking (1 2 2 4).
    Lower rank is better.
    Input a is 'higher is better' (like score or votes).
    """
    n = len(a)
    ranks = np.zeros(n, dtype=int)
    for i in range(n):
        rank = 1
        for j in range(n):
            if a[j] > a[i]: # strictly better than me
                rank += 1
        ranks[i] = rank
    return ranks

def main():
    file_path = 'e:/美赛/Q1_estimated_fan_votes_optimized.csv'
    if not os.path.exists(file_path):
        print("File not found.")
        return
        
    df = pd.read_csv(file_path)
    
    total_weeks = 0
    consistent_weeks = 0
    total_std = 0
    count_std = 0
    
    # Analyze by season and week
    for season in df['season'].unique():
        season_data = df[df['season'] == season]
        for week in season_data['week'].unique():
            week_data = season_data[season_data['week'] == week]
            
            # Skip if no one eliminated
            eliminated = week_data[week_data['status'] == 'Eliminated']
            if len(eliminated) == 0:
                continue
                
            total_weeks += 1
            
            # Reconstruct elimination logic
            scores = week_data['judge_score'].values
            votes = week_data['est_vote_share'].values
            names = week_data['name'].values
            
            # Ranks (1 is best)
            judge_ranks = rankdata_min(scores)
            fan_ranks = rankdata_min(votes)
            
            total_ranks = judge_ranks + fan_ranks
            
            # Tie breaker: Fan Vote (Lower fan rank is better)
            # We want to eliminate the WORST (Highest rank number)
            # If tie in total ranks, the one with WORSE fan rank (Higher number) is eliminated first.
            # So metric = total_ranks + (fan_ranks / 1000.0)
            # Max metric is eliminated.
            
            metric = total_ranks + (fan_ranks / 1000.0)
            
            # Identify simulated eliminated
            # Handle double elimination? Usually 1.
            # Let's see who has the max metric.
            
            # Sort by metric descending
            sorted_indices = np.argsort(-metric) # Descending
            
            # Number of people actually eliminated
            num_elim = len(eliminated)
            
            simulated_eliminated_indices = sorted_indices[:num_elim]
            simulated_eliminated_names = names[simulated_eliminated_indices]
            
            actual_eliminated_names = eliminated['name'].values
            
            # Check if sets match
            if set(simulated_eliminated_names) == set(actual_eliminated_names):
                consistent_weeks += 1
            else:
                pass
                # print(f"Mismatch Season {season} Week {week}: Actual {actual_eliminated_names}, Sim {simulated_eliminated_names}")
                
            # Mean Std Dev
            total_std += week_data['est_vote_std'].mean()
            count_std += 1

    accuracy = consistent_weeks / total_weeks if total_weeks > 0 else 0
    avg_std = total_std / count_std if count_std > 0 else 0
    
    print(f"Total Weeks Analyzed: {total_weeks}")
    print(f"Consistent Weeks: {consistent_weeks}")
    print(f"Accuracy (Consistency): {accuracy:.2%}")
    print(f"Average Certainty (Std Dev): {avg_std:.4f}")

if __name__ == "__main__":
    main()