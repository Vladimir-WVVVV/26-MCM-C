import pandas as pd
import numpy as np
import os
import sys

# Import HybridSolver from Q1_hybrid_solver
# Assuming Q1_hybrid_solver.py is in the same directory
from Q1_hybrid_solver import HybridSolver

def load_data():
    file_path = 'e:/美赛/2026_MCM_Problem_C_Data.csv'
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}")
        return None
    
    df = pd.read_csv(file_path)
    
    # Check if this is the raw data file or already processed
    if 'week1_judge1_score' in df.columns:
        print("Detected raw wide-format data. Converting to long format...")
        # Convert Wide to Long (similar to Q1_estimate_votes_optimized.py logic)
        long_data = []
        max_weeks = 11 # As seen in read output
        
        for idx, row in df.iterrows():
            try:
                season = row['season']
                name = row['celebrity_name']
                result_str = str(row['results'])
                
                # Determine status per week
                # This is tricky without week-by-week status columns.
                # But we can infer: if score exists, they were present.
                # If they were eliminated in Week X, then for Week >= X they might be absent or have special status?
                # Actually, data has scores up to elimination.
                
                # Parse elimination week
                elim_week = 999
                status_map = {} # Week -> Status
                
                if 'Eliminated Week' in result_str:
                    try:
                        elim_week = int(result_str.split('Week')[-1].strip())
                    except:
                        pass
                elif 'Winner' in result_str or '1st' in str(row['placement']):
                    elim_week = 999
                elif '2nd' in str(row['placement']):
                    elim_week = 999
                elif '3rd' in str(row['placement']):
                    elim_week = 999
                
                for w in range(1, max_weeks + 1):
                    # Calculate total judge score
                    score_sum = 0
                    count = 0
                    for j in range(1, 5):
                        col = f'week{w}_judge{j}_score'
                        if col in df.columns:
                            val = row[col]
                            if pd.notna(val) and str(val).strip() not in ['N/A', '']:
                                try:
                                    score_sum += float(val)
                                    count += 1
                                except:
                                    pass
                    
                    if count > 0:
                        # Determine status for this week
                        # If this is the elimination week, Status = Eliminated
                        # Else Status = Safe
                        # Note: This is a simplification. Sometimes multiple people eliminated?
                        # Or if they are still playing (w < elim_week), they are Safe.
                        
                        current_status = 'Safe'
                        if w == elim_week:
                            current_status = 'Eliminated'
                        elif w > elim_week:
                            continue # Should not happen if scores are N/A, but just in case
                            
                        long_data.append({
                            'season': season,
                            'week': w,
                            'contestant': name,
                            'judge_score': score_sum,
                            'result': current_status
                        })
            except Exception as e:
                pass
                
        df = pd.DataFrame(long_data)

    # Basic cleaning
    df = df.dropna(subset=['season', 'week', 'contestant', 'judge_score', 'result'])
    return df

def calculate_accuracy():
    print("Loading data...")
    df = load_data()
    if df is None:
        return

def run_single_trial(trial_id, df, log_file=None):
    print(f"  [Debug] Trial {trial_id+1} started. DF shape: {df.shape}", flush=True)
    if log_file:
        log_file.write(f"  [Debug] Trial {trial_id+1} started. DF shape: {df.shape}\n")
        log_file.flush()
        
    # Initialize Solver (Fresh instance for each trial to reset history)
    solver = HybridSolver()
    
    # Group by Season and Week
    grouped = df.groupby(['season', 'week'])
    sorted_groups = sorted(grouped, key=lambda x: (x[0][0], x[0][1]))
    print(f"  [Debug] Found {len(sorted_groups)} weeks to process.", flush=True)
    if log_file:
        log_file.write(f"  [Debug] Found {len(sorted_groups)} weeks to process.\n")
        log_file.flush()
    
    total_weeks = 0
    correct_weeks = 0
    
    # print(f"Trial {trial_id+1} started...")
    
    for (season, week), group in sorted_groups:
        if total_weeks % 20 == 0:
             msg = f"  [Debug] Processing week {total_weeks} (S{season} W{week})..."
             print(msg, flush=True)
             if log_file:
                 log_file.write(msg + "\n")
                 log_file.flush()
                 
        names = group['contestant'].values
        scores = group['judge_score'].values
        statuses = group['result'].values
        
        # Dummy industries/ages
        industries = ['Unknown'] * len(names) 
        ages = [0] * len(names)
        
        try:
            # Run Solver
            est_votes = solver.solve_week(season, week, names, scores, statuses, industries, ages)
            
            # Verify correctness
            method = 'Ranking'
            if 3 <= int(season) <= 27:
                method = 'Percentage'
                
            is_correct = True
            
            # Logic from check_validity inside solver
            if method == 'Ranking':
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
                    
                judge_ranks = rankdata_min(-scores)
                fan_ranks = rankdata_min(-est_votes)
                metric = (judge_ranks + fan_ranks) + (fan_ranks / 1000.0) # Lower is better
                
                # Check if Eliminated has WORST (Highest) metric
                elim_indices = np.where(statuses == 'Eliminated')[0]
                safe_indices = np.where(statuses == 'Safe')[0]
                
                if len(elim_indices) > 0 and len(safe_indices) > 0:
                    min_elim_metric = np.min(metric[elim_indices])
                    max_safe_metric = np.max(metric[safe_indices])
                    
                    if min_elim_metric <= max_safe_metric:
                        is_correct = False
                        # print(f"    Mismatch S{season} W{week} (Ranking)")
                        
            elif method == 'Percentage':
                if np.sum(scores) > 0:
                    s_pct = scores / np.sum(scores)
                else:
                    s_pct = np.zeros(len(scores))
                    
                combined = 0.5 * s_pct + 0.5 * est_votes
                # Check if Eliminated has WORST (Lowest) combined
                elim_indices = np.where(statuses == 'Eliminated')[0]
                safe_indices = np.where(statuses == 'Safe')[0]
                
                if len(elim_indices) > 0 and len(safe_indices) > 0:
                    max_elim_combined = np.max(combined[elim_indices])
                    min_safe_combined = np.min(combined[safe_indices])
                    
                    if max_elim_combined >= min_safe_combined:
                        is_correct = False
                        # print(f"    Mismatch S{season} W{week} (Percentage)")
            
            if is_correct:
                correct_weeks += 1
            total_weeks += 1
            
        except Exception as e:
            print(f"Error in S{season} W{week}: {e}", flush=True)
            # Continue to next week
            total_weeks += 1
        
    accuracy = correct_weeks / total_weeks if total_weeks > 0 else 0
    return accuracy, correct_weeks, total_weeks

def calculate_accuracy():
    print("Loading data...")
    df = load_data()
    if df is None:
        return

    num_trials = 5
    accuracies = []
    
    for trial_id in range(num_trials):
        print(f"Running Trial {trial_id+1}...", flush=True)
        
        acc, corr, tot = run_single_trial(trial_id, df, log_file=None)
        accuracies.append(acc)
        print(f"Trial {trial_id+1} Accuracy: {acc:.4%}", flush=True)
        
    avg_acc = np.mean(accuracies)
    min_acc = np.min(accuracies)
    max_acc = np.max(accuracies)
    std_acc = np.std(accuracies)
    
    with open('e:/美赛/accuracy_report.txt', 'w') as f:
        f.write("="*50 + "\n")
        f.write(f"Final Robustness Evaluation Results ({num_trials} Trials)\n")
        f.write("="*50 + "\n")
        f.write(f"Average Accuracy: {avg_acc:.4%}\n")
        f.write(f"Min Accuracy:     {min_acc:.4%}\n")
        f.write(f"Max Accuracy:     {max_acc:.4%}\n")
        f.write(f"Std Deviation:    {std_acc:.6f}\n")
        f.write(f"Stability:        {'Perfectly Stable' if std_acc < 1e-6 else 'Slight Variance'}\n")
        f.write("="*50 + "\n")
        
    print("\n" + "="*50)
    print(f"Final Robustness Evaluation Results ({num_trials} Trials)")
    print("="*50)
    print(f"Average Accuracy: {avg_acc:.4%}")
    print(f"Min Accuracy:     {min_acc:.4%}")
    print(f"Max Accuracy:     {max_acc:.4%}")
    print(f"Std Deviation:    {std_acc:.6f}")
    print(f"Stability:        {'Perfectly Stable' if std_acc < 1e-6 else 'Slight Variance'}")
    print("="*50)

if __name__ == "__main__":
    calculate_accuracy()
