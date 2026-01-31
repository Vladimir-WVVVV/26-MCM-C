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
# 简单的在线岭回归器 (Online Ridge Regressor)
# ==========================================
class OnlineRidgeRegressor:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.X = []
        self.y = []
        self.weights = None
        self.feature_names = ['score_norm', 'age_norm', 'is_athlete', 'is_actor', 'prev_share']

    def extract_features(self, score, age, industry, prev_share):
        # Normalize Score (Assume 10-30 range roughly)
        s_norm = (score - 15) / 15.0
        
        # Normalize Age (Assume 20-70)
        try:
            a = float(age)
        except:
            a = 30.0
        a_norm = (a - 40) / 20.0
        
        ind = str(industry).lower()
        is_athlete = 1.0 if 'athlete' in ind else 0.0
        is_actor = 1.0 if 'actor' in ind or 'singer' in ind else 0.0
        
        return [1.0, s_norm, a_norm, is_athlete, is_actor, prev_share] # Added bias term 1.0

    def fit(self, X_batch, y_batch):
        if len(X_batch) == 0: return
        self.X.extend(X_batch)
        self.y.extend(y_batch)
        
        # Re-train weights
        X_mat = np.array(self.X)
        y_vec = np.array(self.y)
        n_feats = X_mat.shape[1]
        
        # Ridge Closed Form: w = (X'X + alpha*I)^-1 X'y
        I = np.eye(n_feats)
        I[0,0] = 0 # Don't penalize bias
        
        try:
            self.weights = np.linalg.inv(X_mat.T @ X_mat + self.alpha * I) @ X_mat.T @ y_vec
        except:
            self.weights = np.zeros(n_feats)

    def predict(self, features):
        if self.weights is None:
            return 0.1 # Default average share
        return np.dot(features, self.weights)

# ==========================================
# 自适应估算器类 (Enhanced with ML)
# ==========================================

class AdaptiveVoteEstimator:
    def __init__(self):
        self.participant_alphas = {}
        self.ml_model = OnlineRidgeRegressor(alpha=5.0) # Strong regularization for stability
        self.history_X = []
        self.history_y = []
        
    def get_ml_guided_alphas(self, names, judge_scores, industries, ages):
        """
        使用 ML 模型预测作为先验中心
        """
        n = len(names)
        alphas = np.zeros(n)
        
        for i, name in enumerate(names):
            # 获取上一周的得票率作为特征
            prev_share = 0.1 # Default
            if name in self.participant_alphas:
                 # Alpha ~ 20 * Share -> Share ~ Alpha / 20
                 prev_share = self.participant_alphas[name] / 20.0
            
            feats = self.ml_model.extract_features(judge_scores[i], ages[i], industries[i], prev_share)
            predicted_share = self.ml_model.predict(feats)
            
            # Clip to reasonable range [0.01, 0.5]
            predicted_share = max(0.01, min(0.5, predicted_share))
            
            # Convert Share to Alpha (Strength)
            # Higher confidence in later weeks? Let's keep scale constant ~50
            alphas[i] = predicted_share * 50.0
            
        return alphas

    def update_history(self, names, estimated_shares, judge_scores, industries, ages):
        """
        更新历史参数 并 训练 ML 模型
        """
        # 1. Update Alpha History (Legacy method)
        scale_factor = 20.0 
        for name, share in zip(names, estimated_shares):
            new_alpha = max(1.0, share * scale_factor)
            old_alpha = self.participant_alphas.get(name, 1.0)
            self.participant_alphas[name] = old_alpha * 0.4 + new_alpha * 0.6
            
        # 2. Update ML Model
        X_batch = []
        y_batch = []
        for i, name in enumerate(names):
            # Feature extraction needs PREVIOUS share, but here we only have current estimated.
            # We use "Lagged" features ideally. 
            # For simplicity in this loop, we use current estimate as target Y,
            # and we reconstruct what the input X WAS (using previous alpha).
            
            prev_share_est = 0.1
            # Note: This is a bit circular if we use the just-updated alpha.
            # Ideally we store the alpha BEFORE update.
            # But approximation is fine for this robust regression.
            
            feats = self.ml_model.extract_features(judge_scores[i], ages[i], industries[i], prev_share_est)
            X_batch.append(feats)
            y_batch.append(estimated_shares[i])
            
        self.ml_model.fit(X_batch, y_batch)

    def solve_week(self, season, week, df_week, num_samples=500):
        try:
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
            
            # 构造先验 (ML Guided)
            alphas = self.get_ml_guided_alphas(names, scores, industries, ages)
            
            valid_samples = []
            
            # Monte Carlo Simulation
            # Strategy 1: Informed Prior (Try to fit reality with our theory)
            batch_size = 1000
            total_generated = 0
            max_attempts = num_samples * 20 # Reduce max attempts for speed
            
            elim_indices = np.where(statuses == 'Eliminated')[0]
            safe_indices = np.where(statuses == 'Safe')[0]
            
            current_alphas = alphas.copy()
            
            while len(valid_samples) < num_samples and total_generated < max_attempts:
                # Dirichlet sampling
                samples = np.random.dirichlet(current_alphas, batch_size)
                total_generated += batch_size
                
                for i in range(batch_size):
                    fan_shares = samples[i]
                    if check_validity(scores, fan_shares, elim_indices, safe_indices, method):
                        valid_samples.append(fan_shares)
                        if len(valid_samples) >= num_samples:
                            break
            
            # Strategy 2: Fallback to Flat Prior (If theory fails, just explain the result)
            # This handles "Shock Eliminations" where popular people get kicked out.
            if len(valid_samples) < num_samples:
                # print(f"  > Week {week}: Switching to Flat Prior for Shock Elimination...")
                flat_alphas = np.ones(n) # Unbiased random search
                total_generated_flat = 0
                max_attempts_flat = num_samples * 50 # Reduce max attempts
                
                while len(valid_samples) < num_samples and total_generated_flat < max_attempts_flat:
                    samples = np.random.dirichlet(flat_alphas, batch_size)
                    total_generated_flat += batch_size
                    
                    for i in range(batch_size):
                        fan_shares = samples[i]
                        if check_validity(scores, fan_shares, elim_indices, safe_indices, method):
                            valid_samples.append(fan_shares)
                            if len(valid_samples) >= num_samples:
                                break
            
            # Final Fallback: If even random search fails (extremely rare constraints), use Prior but relax
            if len(valid_samples) < 10:
                 # Use Prior Mean directly but add noise (Low confidence)
                prior_mean = alphas / np.sum(alphas)
                if num_samples - len(valid_samples) > 0:
                    for _ in range(num_samples - len(valid_samples)):
                        noise = np.random.normal(0, 0.05, n) # More noise
                        noisy_sample = prior_mean + noise
                        noisy_sample = np.abs(noisy_sample)
                        noisy_sample /= np.sum(noisy_sample)
                        valid_samples.append(noisy_sample)
            
            valid_samples = np.array(valid_samples)
            
            if len(valid_samples) == 0:
                # Absolute fallback if everything fails
                valid_samples = np.array([np.ones(n)/n for _ in range(num_samples)])
    
            # Calculate estimates (Mean & Std)
            est_means = np.mean(valid_samples, axis=0)
            est_stds = np.std(valid_samples, axis=0)
            
            # Update history for next week (and train ML model)
            self.update_history(names, est_means, scores, industries, ages)
            
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
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise e

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
    
    sorted_seasons = sorted(df_long['season'].unique(), key=lambda x: int(str(x).replace('Season','').strip()) if 'Season' in str(x) else 0)
    
    for season in sorted_seasons:
        print(f"Processing {season}...", flush=True)
        # 每个赛季开始重置历史记忆
        estimator.participant_alphas = {} 
        
        season_data = df_long[df_long['season'] == season]
        for week in sorted(season_data['week'].unique()):
            week_data = season_data[season_data['week'] == week]
            
            if len(week_data) == 0: continue
            
            # Solve for this week
            try:
                week_results = estimator.solve_week(season, week, week_data)
                results.extend(week_results)
            except Exception as e:
                print(f"Error in {season} Week {week}: {e}")

    # Save Results
    results_df = pd.DataFrame(results)
    results_df.to_csv('e:/美赛/Q1_estimated_fan_votes_optimized.csv', index=False)
    print("Optimization Complete. Results saved to Q1_estimated_fan_votes_optimized.csv")

if __name__ == "__main__":
    main()
