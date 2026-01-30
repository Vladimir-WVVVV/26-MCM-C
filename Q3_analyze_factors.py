import pandas as pd
import numpy as np
import os

def main():
    votes_path = 'e:/美赛/Q1_estimated_fan_votes.csv'
    data_path = 'e:/美赛/2026_MCM_Problem_C_Data.csv'
    
    if not os.path.exists(votes_path) or not os.path.exists(data_path):
        print("Required files not found.")
        return

    # Load data
    votes_df = pd.read_csv(votes_path)
    raw_df = pd.read_csv(data_path)
    
    # We need to map Name/Season in votes_df to metadata in raw_df
    # raw_df columns: celebrity_name, celebrity_industry, celebrity_age_during_season, season
    
    # Create a metadata lookup dictionary
    meta_dict = {}
    for _, row in raw_df.iterrows():
        key = (row['season'], row['celebrity_name'])
        meta_dict[key] = {
            'industry': row['celebrity_industry'],
            'age': row['celebrity_age_during_season']
        }
    
    # Add metadata to votes_df
    industries = []
    ages = []
    
    for _, row in votes_df.iterrows():
        key = (row['season'], row['name'])
        # Try exact match
        if key in meta_dict:
            industries.append(meta_dict[key]['industry'])
            ages.append(meta_dict[key]['age'])
        else:
            # Try fuzzy match if needed, for now use None
            # Some names might have slight differences
            found = False
            for k in meta_dict:
                if k[0] == row['season'] and (k[1] in row['name'] or row['name'] in k[1]):
                    industries.append(meta_dict[k]['industry'])
                    ages.append(meta_dict[k]['age'])
                    found = True
                    break
            if not found:
                industries.append(None)
                ages.append(None)
                
    votes_df['industry'] = industries
    votes_df['age'] = ages
    
    # Drop rows with missing metadata
    df_clean = votes_df.dropna(subset=['age', 'industry'])
    
    print(f"Analyzed {len(df_clean)} records out of {len(votes_df)}")
    
    # 1. Age Analysis
    print("\n--- Correlation with Age ---")
    corr_score = df_clean['judge_score'].corr(df_clean['age'])
    corr_vote = df_clean['est_vote_share'].corr(df_clean['age'])
    print(f"Age vs Judge Score: {corr_score:.4f}")
    print(f"Age vs Fan Vote: {corr_vote:.4f}")
    
    # 2. Industry Analysis
    print("\n--- Industry Analysis (Top 10) ---")
    industry_stats = df_clean.groupby('industry').agg({
        'judge_score': 'mean',
        'est_vote_share': 'mean',
        'name': 'count'
    }).rename(columns={'name': 'count'})
    
    # Filter industries with enough data
    industry_stats = industry_stats[industry_stats['count'] > 20]
    
    # Sort by Fan Votes
    print(industry_stats.sort_values('est_vote_share', ascending=False).head(10))
    
    # 3. Consistency between Judge and Fan
    print("\n--- Judge vs Fan Consistency ---")
    corr_jf = df_clean['judge_score'].corr(df_clean['est_vote_share'])
    print(f"Correlation between Judge Score and Fan Votes: {corr_jf:.4f}")
    
    # Save for visualization
    df_clean.to_csv('e:/美赛/Q3_factor_analysis_data.csv', index=False)
    print("\nFactor analysis data saved to e:/美赛/Q3_factor_analysis_data.csv")

if __name__ == "__main__":
    main()
