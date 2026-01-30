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
            if a[j] > a[i]:
                rank += 1
        ranks[i] = rank
    return ranks

def calculate_metrics(df_week, method_name):
    """
    Simulate elimination for a single week and return the name of the eliminated participant.
    """
    scores = df_week['judge_score'].values
    votes = df_week['est_vote_share'].values
    names = df_week['name'].values
    
    n = len(names)
    if n <= 1:
        return None, None

    # 1. Ranking Method
    if method_name == 'Ranking':
        judge_ranks = rankdata_min(scores)
        fan_ranks = rankdata_min(votes)
        total_ranks = judge_ranks + fan_ranks
        # Tie-breaker: Fan Votes (lower rank is better, so higher vote share is better)
        # We want to eliminate MAX total_rank.
        # If tie, eliminate the one with WORSE fan rank (Higher number).
        # Metric = total_ranks + (fan_ranks / 1000.0)
        metric = total_ranks + (fan_ranks / 1000.0)
        elim_idx = np.argmax(metric)
        return names[elim_idx], metric

    # 2. Percentage Method
    elif method_name == 'Percentage':
        # Normalize scores to %
        if np.sum(scores) > 0:
            score_share = scores / np.sum(scores)
        else:
            score_share = np.zeros(n)
        
        # Combined = 0.5 * Score% + 0.5 * Vote%
        combined = 0.5 * score_share + 0.5 * votes
        # Eliminate MIN combined
        elim_idx = np.argmin(combined)
        return names[elim_idx], combined

    # 3. New Mechanism: Percentage + Judge Save
    elif method_name == 'New_Mechanism':
        # Same combined score as Percentage
        if np.sum(scores) > 0:
            score_share = scores / np.sum(scores)
        else:
            score_share = np.zeros(n)
        combined = 0.5 * score_share + 0.5 * votes
        
        # Find Bottom 2
        sorted_indices = np.argsort(combined) # Ascending: 0 is lowest
        bottom_2_indices = sorted_indices[:2]
        
        if len(bottom_2_indices) < 2:
            return names[bottom_2_indices[0]], combined
            
        # Judge Save: Judges save the one with higher Judge Score
        p1_idx = bottom_2_indices[0]
        p2_idx = bottom_2_indices[1]
        
        if scores[p1_idx] > scores[p2_idx]:
            # Save p1, eliminate p2
            return names[p2_idx], combined
        elif scores[p2_idx] > scores[p1_idx]:
            # Save p2, eliminate p1
            return names[p1_idx], combined
        else:
            # Tie in judge score? Fallback to original combined score (Fan vote decides)
            # Original combined score: Lower one is eliminated.
            if combined[p1_idx] < combined[p2_idx]:
                return names[p1_idx], combined
            else:
                return names[p2_idx], combined

    return None, None

def main():
    print("Starting Task 4: New Mechanism Design and Evaluation...")
    
    # Load Data
    votes_file = 'e:/美赛/Q1_estimated_fan_votes_optimized.csv'
    if not os.path.exists(votes_file):
        print("Error: Q1_estimated_fan_votes_optimized.csv not found.")
        return
        
    df = pd.read_csv(votes_file)
    
    # We need to simulate full seasons.
    # However, our data is static (estimated votes for the *actual* history).
    # Changing the elimination mechanism would change who is present in subsequent weeks.
    # This is a complex counter-factual simulation.
    # SIMPLIFICATION: We will evaluate "Single Week Fairness".
    # We check: "If this method was used this week, who would go home?"
    # And compare the characteristics of the eliminated person.
    
    results = []
    
    methods = ['Ranking', 'Percentage', 'New_Mechanism']
    
    for season in df['season'].unique():
        season_data = df[df['season'] == season]
        
        # Filter for weeks where elimination actually happened
        # (To compare apples to apples)
        eliminated_weeks = season_data[season_data['status'] == 'Eliminated']['week'].unique()
        
        for week in eliminated_weeks:
            week_data = season_data[season_data['week'] == week]
            
            # Actual Eliminated
            actual_elim = week_data[week_data['status'] == 'Eliminated']
            if len(actual_elim) == 0: continue
            actual_elim_name = actual_elim.iloc[0]['name']
            actual_elim_score = actual_elim.iloc[0]['judge_score']
            actual_elim_vote = actual_elim.iloc[0]['est_vote_share']
            
            row_base = {
                'season': season,
                'week': week,
                'actual_eliminated': actual_elim_name,
                'actual_elim_score': actual_elim_score,
                'actual_elim_vote': actual_elim_vote
            }
            
            for method in methods:
                sim_elim_name, _ = calculate_metrics(week_data, method)
                
                # Get stats of simulated eliminated person
                if sim_elim_name:
                    p_data = week_data[week_data['name'] == sim_elim_name].iloc[0]
                    
                    res = row_base.copy()
                    res['method'] = method
                    res['sim_eliminated'] = sim_elim_name
                    res['sim_elim_score'] = p_data['judge_score']
                    res['sim_elim_vote'] = p_data['est_vote_share']
                    
                    # "Unfairness" Metric: Did we eliminate a high scorer?
                    # Score Rank (1 is best). If we eliminate rank 1, that's bad.
                    # Normalized Score (0-1)
                    max_score = week_data['judge_score'].max()
                    min_score = week_data['judge_score'].min()
                    if max_score > min_score:
                        norm_score = (p_data['judge_score'] - min_score) / (max_score - min_score)
                    else:
                        norm_score = 0.5 # Neutral
                    
                    res['talent_loss_score'] = norm_score # Higher means we lost a better dancer
                    
                    results.append(res)

    res_df = pd.DataFrame(results)
    res_df.to_csv('e:/美赛/Q4_simulation_results.csv', index=False)
    
    # Aggregate Metrics
    print("\n--- Evaluation Results ---")
    summary = res_df.groupby('method').agg({
        'talent_loss_score': 'mean', # Lower is better (we want to eliminate low scorers)
        'sim_elim_score': 'mean',    # Lower is better
        'sim_elim_vote': 'mean'      # Lower is better (we want to eliminate low vote getters too?)
    }).reset_index()
    
    summary['Talent_Retention_Score'] = 1 - summary['talent_loss_score'] # Higher is better
    
    print(summary)
    
    # Save Summary
    summary.to_csv('e:/美赛/Q4_mechanism_comparison_summary.csv', index=False)
    print("\nDetailed results saved to e:/美赛/Q4_simulation_results.csv")
    print("Summary saved to e:/美赛/Q4_mechanism_comparison_summary.csv")

if __name__ == "__main__":
    main()