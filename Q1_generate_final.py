import pandas as pd
import numpy as np
import os
from Q1_hybrid_solver import HybridSolver

def load_data():
    file_path = 'e:/美赛/2026_MCM_Problem_C_Data.csv'
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}")
        return None
    
    df = pd.read_csv(file_path)
    
    if 'week1_judge1_score' in df.columns:
        print("Detected raw wide-format data. Converting to long format...")
        long_data = []
        max_weeks = 11
        
        for idx, row in df.iterrows():
            try:
                season = row['season']
                name = row['celebrity_name']
                result_str = str(row['results'])
                
                # Determine status per week
                elim_week = 999
                
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
                        current_status = 'Safe'
                        if w == elim_week:
                            current_status = 'Eliminated'
                        elif w > elim_week:
                            continue
                            
                        # Extract demographics
                        industry = str(row.get('celebrity_industry', 'Unknown'))
                        try:
                            age = float(row.get('celebrity_age_during_season', 30))
                        except:
                            age = 30
                        
                        long_data.append({
                            'season': season,
                            'week': w,
                            'contestant': name,
                            'judge_score': score_sum,
                            'result': current_status,
                            'industry': industry,
                            'age': age
                        })
            except Exception as e:
                pass
                
        df = pd.DataFrame(long_data)

    df = df.dropna(subset=['season', 'week', 'contestant', 'judge_score', 'result'])
    return df

def generate_final_csv():
    print("Loading data...")
    df = load_data()
    if df is None:
        return

    print(f"Data loaded. Shape: {df.shape}")
    
    # Initialize Solver
    solver = HybridSolver()
    
    # Group by Season and Week
    grouped = df.groupby(['season', 'week'])
    sorted_groups = sorted(grouped, key=lambda x: (x[0][0], x[0][1]))
    
    print(f"Processing {len(sorted_groups)} weeks...")
    
    results = []
    
    for (season, week), group in sorted_groups:
        names = group['contestant'].values
        scores = group['judge_score'].values
        statuses = group['result'].values
        
        # We need industries and ages for the solver to work optimally (ML prior)
        # But in this simplified load, we have placeholders.
        # Ideally we should merge with demographic data if available.
        # For now, let's proceed. The solver has fallback to score-based prior if demographics weak.
        
        industries = group['industry'].values
        ages = group['age'].values
        
        try:
            est_votes = solver.solve_week(season, week, names, scores, statuses, industries, ages)
            
            for i in range(len(names)):
                results.append({
                    'season': season,
                    'week': week,
                    'name': names[i],
                    'score': scores[i],
                    'est_vote_share': est_votes[i],
                    'status': statuses[i],
                    'industry': industries[i],
                    'age': ages[i]
                })
        except Exception as e:
            print(f"Error processing S{season} W{week}: {e}")
            
    # Save to CSV
    output_df = pd.DataFrame(results)
    output_path = 'e:/美赛/Q1_estimated_fan_votes_optimized.csv'
    output_df.to_csv(output_path, index=False)
    print(f"Successfully saved {len(output_df)} rows to {output_path}")

if __name__ == "__main__":
    generate_final_csv()
