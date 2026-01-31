import pandas as pd
import numpy as np
import os

# ==========================================
# 核心工具函数
# ==========================================

def rankdata_min(a):
    """
    不依赖 scipy 的排名函数 (Method='min')
    输入: 负的分数 (因为我们要 High Score -> Rank 1)
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

def check_validity(judge_scores, fan_shares, eliminated_indices, safe_indices, method='Ranking'):
    """
    验证一组粉丝投票是否符合淘汰结果
    """
    n = len(judge_scores)
    
    # 1. 计算排名 (Ranking Method)
    if method == 'Ranking':
        judge_ranks = rankdata_min(-judge_scores)
        fan_ranks = rankdata_min(-fan_shares)
        
        # 构造 Metric: TotalRank + FanRank/1000 (Tie-breaker)
        metric = (judge_ranks + fan_ranks) + (fan_ranks / 1000.0)
        
    # 2. 百分比法 (Percentage Method)
    elif method == 'Percentage':
        if np.sum(judge_scores) > 0:
            score_pct = judge_scores / np.sum(judge_scores)
        else:
            score_pct = np.zeros(n)
        
        # Combined = 0.5 * Score% + 0.5 * Vote%
        # Eliminate MIN combined. So Metric should be NEGATIVE Combined to keep logic (Max Metric Eliminated)
        combined = 0.5 * score_pct + 0.5 * fan_shares
        metric = -combined # The smaller combined, the larger metric -> Eliminated
        
    else:
        # Default fallback
        metric = np.random.rand(n)

    if len(eliminated_indices) == 0:
        return True 
        
    elim_metrics = metric[eliminated_indices]
    
    if len(safe_indices) > 0:
        safe_metrics = metric[safe_indices]
        # 严格约束：淘汰者的表现（Metric）必须比所有幸存者都差
        # 即 Min(Elim) > Max(Safe)
        if np.min(elim_metrics) > np.max(safe_metrics):
            return True
        return False
    else:
        return True

# ==========================================
# 自适应估算器类
# ==========================================

class AdaptiveVoteEstimator:
    def __init__(self):
        self.participant_alphas = {}
        self.base_alpha = 2.0 
    
    def get_prior_alphas(self, names, judge_scores, industries=None, ages=None):
        """
        构造本周的先验分布参数 Alpha
        Alpha = 历史人气 + 本周评委表现 + 人气特征偏好
        """
        n = len(names)
        alphas = np.ones(n)
        
        # 归一化评委分 (0-1)
        if np.max(judge_scores) > np.min(judge_scores):
            norm_scores = (judge_scores - np.min(judge_scores)) / (np.max(judge_scores) - np.min(judge_scores) + 1e-6)
        else:
            norm_scores = np.zeros(n) + 0.5
            
        for i, name in enumerate(names):
            # 1. 历史成分
            hist_alpha = self.participant_alphas.get(name, 1.0)
            
            # 2. 评委成分
            score_boost = norm_scores[i] * 1.0 
            
            # 3. 人气特征偏好 (Heuristic)
            # 假设：年轻选手 (>20, <40) 和 运动员/歌手 可能更受欢迎
            feature_boost = 0.0
            if industries is not None:
                ind = str(industries[i]).lower()
                if 'athlete' in ind or 'singer' in ind or 'actor' in ind:
                    feature_boost += 0.5
            if ages is not None:
                try:
                    age = float(ages[i])
                    if 18 <= age <= 35:
                        feature_boost += 0.3
                except: pass
            
            # 综合 Alpha
            alphas[i] = hist_alpha * 0.6 + score_boost * 0.8 + feature_boost + 0.5
            
        return alphas

    def update_history(self, names, estimated_shares):
        """
        根据本周估算结果，更新选手的历史人气参数
        """
        # 将 Share 转换回 Alpha 规模，用于下一周
        # Share = 0.2 -> Alpha ~ 20 (假设总强度为100)
        scale_factor = 20.0 
        for name, share in zip(names, estimated_shares):
            new_alpha = max(1.0, share * scale_factor)
            # 平滑更新：Old * 0.4 + New * 0.6
            old_alpha = self.participant_alphas.get(name, 1.0)
            self.participant_alphas[name] = old_alpha * 0.4 + new_alpha * 0.6

    def solve_week(self, season, week, df_week, num_samples=3000):
        """
        Estimate vote shares for a specific week
        """
        names = df_week['name'].values
        scores = df_week['score'].values
        statuses = df_week['status'].values
        industries = df_week.get('industry', pd.Series(['']*len(names))).values
        ages = df_week.get('age', pd.Series([0]*len(names))).values
        n = len(names)
        
        # 确定使用哪种方法 (Ranking vs Percentage)
        # S1-S2, S28-S34: Ranking
        # S3-S27: Percentage
        method = 'Ranking'
        try:
            s_str = str(season).replace('Season', '').strip()
            s_num = int(float(s_str))
            if 3 <= s_num <= 27:
                method = 'Percentage'
        except:
            pass # Default to Ranking if parsing fails
        
        # 构造先验
        alphas = self.get_prior_alphas(names, scores, industries, ages)
        
        valid_samples = []
        
        # Monte Carlo Simulation
        # Batch generation for speed
        batch_size = 1000
        total_generated = 0
        
        elim_indices = np.where(statuses == 'Eliminated')[0]
        safe_indices = np.where(statuses == 'Safe')[0]
        
        while len(valid_samples) < num_samples and total_generated < num_samples * 20:
            # Dirichlet sampling
            samples = np.random.dirichlet(alphas, batch_size)
            total_generated += batch_size
            
            for i in range(batch_size):
                fan_shares = samples[i]
                if check_validity(scores, fan_shares, elim_indices, safe_indices, method):
                    valid_samples.append(fan_shares)
                    if len(valid_samples) >= num_samples:
                        break
        
        # Fallback if hard constraints are too strict (Soft Relaxation)
        if len(valid_samples) < 50:
            # print(f"Warning: Low acceptance rate for {season} Week {week}. Relaxing constraints...")
            # Use Prior Mean directly but add noise
            prior_mean = alphas / np.sum(alphas)
            for _ in range(num_samples - len(valid_samples)):
                noise = np.random.normal(0, 0.01, n)
                noisy_sample = prior_mean + noise
                noisy_sample = np.abs(noisy_sample)
                noisy_sample /= np.sum(noisy_sample)
                valid_samples.append(noisy_sample)
        
        valid_samples = np.array(valid_samples)
        
        # Calculate estimates (Mean & Std)
        est_means = np.mean(valid_samples, axis=0)
        est_stds = np.std(valid_samples, axis=0)
        
        # Update history for next week
        self.update_history(names, est_means)
        
        # Calculate Consistency (Accuracy) for this week
        # Re-check if the Mean Estimate satisfies the rule
        is_consistent = check_validity(scores, est_means, elim_indices, safe_indices, method)
        
        # Calculate Certainty (Average Relative Width of 95% CI)
        # Simple proxy: 1 - (2 * std / mean) or just avg std
        certainty_metric = 1.0 - (np.mean(est_stds) * 4) # Rough scaling
        
        results = []
        for i, name in enumerate(names):
            results.append({
                'season': season,
                'week': week,
                'name': name,
                'score': scores[i],
                'status': statuses[i],
                'industry': str(industries[i]),
                'age': str(ages[i]),
                'est_vote_share': est_means[i],
                'est_vote_std': est_stds[i],
                'method_used': method,
                'is_consistent': is_consistent,
                'certainty_score': certainty_metric
            })
            
        return results

# ==========================================
# 主程序
# ==========================================

def main():
    file_path = 'e:/美赛/2026_MCM_Problem_C_Data.csv'
    if not os.path.exists(file_path):
        print("Data file not found.")
        return

    # 读取并预处理数据
    df = pd.read_csv(file_path)
    print("Columns:", df.columns.tolist())
    long_data = []
    max_weeks = 11
    
    for idx, row in df.iterrows():
        try:
            season = row['season']
            name = row['celebrity_name']
            result_str = str(row['results'])
            
            # Determine elimination week
            elim_week = 999
            if 'Eliminated Week' in result_str:
                try:
                    elim_week = int(result_str.split('Week')[-1].strip())
                except:
                    pass
            
            for w in range(1, max_weeks + 1):
                # Calculate total judge score for the week
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
                
                # Handling N/A and 0 scores
                # If count < 3 and score_sum == 0: Skip (likely already eliminated or not present)
                # If count == 3: Scale to 4 judges equivalent (score * 4/3)
                
                if count == 0:
                    continue
                
                if count == 3:
                    score_sum = score_sum * (4/3)
                
                # Determine status for this week
                current_status = 'Safe'
                if w == elim_week:
                    current_status = 'Eliminated'
                elif w > elim_week:
                    continue # Already left
                
                long_data.append({
                    'season': season, 
                    'week': w, 
                    'name': name,
                    'score': score_sum, 
                    'status': current_status,
                    'industry': row.get('celebrity_industry', 'Unknown'),
                    'age': row.get('celebrity_age_during_season', 30)
                })
        except Exception as e:
            print(f"Skipping row {idx}: {e}")
            pass

    df_long = pd.DataFrame(long_data)
    print(f"Processed {len(df_long)} rows.")
    if len(df_long) == 0:
        print("No data processed. Exiting.")
        return
    
    estimator = AdaptiveVoteEstimator()
    results = []
    
    print("Starting Optimized Simulation...")
    
    for season in sorted(df_long['season'].unique()):
        # 每个赛季开始重置历史记忆
        estimator.participant_alphas = {} 
        
        season_data = df_long[df_long['season'] == season]
        for week in sorted(season_data['week'].unique()):
            week_data = season_data[season_data['week'] == week]
            
            # Skip if no one is eliminated (unless it's the finale, but even then)
            # Logic inside solve_week handles this, but we should be careful
            
            # Solve for this week
            week_results = estimator.solve_week(season, week, week_data)
            
            results.extend(week_results)
            
            # print(f"  {season} Week {week}: Processed {len(week_results)} participants")

    # Save Results
    results_df = pd.DataFrame(results)
    results_df.to_csv('e:/美赛/Q1_estimated_fan_votes_optimized.csv', index=False)
    print("Optimization Complete. Results saved to Q1_estimated_fan_votes_optimized.csv")

if __name__ == "__main__":
    main()
