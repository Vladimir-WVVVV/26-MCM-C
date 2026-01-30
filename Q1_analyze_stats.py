import pandas as pd
import numpy as np

# Load the estimation results
df = pd.read_csv('e:/美赛/estimated_fan_votes.csv')

# 1. Certainty Analysis (Standard Deviation)
mean_std = df['est_vote_std'].mean()
max_std = df['est_vote_std'].max()
min_std = df['est_vote_std'].min()

# Analyze std by placement (judge rank tiers) to see if certainty varies by performance
# We'll categorize judge scores into quartiles for this
df['score_quartile'] = pd.qcut(df['judge_score'], 4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
std_by_score = df.groupby('score_quartile')['est_vote_std'].mean()

print(f"Global Mean Std Dev (Certainty Metric): {mean_std:.4f}")
print(f"Max Std Dev: {max_std:.4f}")
print(f"Min Std Dev: {min_std:.4f}")
print("\nMean Std Dev by Judge Score Tier:")
print(std_by_score)

# 2. Consistency Analysis
# Since we don't have the raw simulation logs here, we can infer consistency 
# by checking how "sharp" the distributions are. 
# A high std dev implies multiple vote combinations could lead to the same result (Low Certainty/Consistency).
# A low std dev implies the solution space is tight.

# Also, let's look at the "Safe" vs "Eliminated" vote shares
elim_avg_share = df[df['status'].str.contains('Eliminated')]['est_vote_share'].mean()
safe_avg_share = df[~df['status'].str.contains('Eliminated')]['est_vote_share'].mean()

print(f"\nAvg Vote Share (Eliminated): {elim_avg_share:.4f}")
print(f"Avg Vote Share (Safe): {safe_avg_share:.4f}")
