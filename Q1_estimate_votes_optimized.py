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

def check_validity(judge_scores, fan_shares, eliminated_indices, safe_indices):
    """
    验证一组粉丝投票是否符合淘汰结果
    """
    n = len(judge_scores)
    
    # 1. 计算排名
    judge_ranks = rankdata_min(-judge_scores)
    fan_ranks = rankdata_min(-fan_shares)
    
    # 2. 计算总排名 (Rank Method)
    # 平局规则：总排名相同，粉丝排名低(数值大)者淘汰
    # 构造 Metric: TotalRank + FanRank/1000
    # Metric 越大 -> 表现越差
    metric = (judge_ranks + fan_ranks) + (fan_ranks / 1000.0)
    
    if len(eliminated_indices) == 0:
        return True # 无人淘汰，任何分布都算对
        
    elim_metrics = metric[eliminated_indices]
    
    if len(safe_indices) > 0:
        safe_metrics = metric[safe_indices]
        # 核心约束：最强的淘汰者，其表现必须比最弱的安全者更差（Metric更大）
        # 或者至少：所有淘汰者都在安全者之后
        
        # 严格约束：min(elim) > max(safe) 
        # (即淘汰圈的第一名，也比安全圈的最后一名差)
        if np.min(elim_metrics) > np.max(safe_metrics):
            return True
        return False
    else:
        # 所有人都是淘汰者（决赛？）
        return True

# ==========================================
# 自适应估算器类
# ==========================================

class AdaptiveVoteEstimator:
    def __init__(self):
        # 记录每位选手的历史人气参数 (Alpha for Dirichlet)
        # Key: Name, Value: Current Alpha
        self.participant_alphas = {}
        self.base_alpha = 2.0 # 基础集中度
    
    def get_prior_alphas(self, names, judge_scores):
        """
        构造本周的先验分布参数 Alpha
        Alpha = 历史人气 + 本周评委表现 + 基础底噪
        """
        n = len(names)
        alphas = np.ones(n)
        
        # 归一化评委分 (0-1)
        if np.max(judge_scores) > np.min(judge_scores):
            norm_scores = (judge_scores - np.min(judge_scores)) / (np.max(judge_scores) - np.min(judge_scores) + 1e-6)
        else:
            norm_scores = np.zeros(n) + 0.5
            
        for i, name in enumerate(names):
            # 1. 历史成分 (Temporal Consistency)
            hist_alpha = self.participant_alphas.get(name, 1.0)
            
            # 2. 评委成分 (Informative Prior)
            # 假设评委分高的人，粉丝通常也多一点点 (0.5权重)
            score_boost = norm_scores[i] * 1.0 
            
            # 综合 Alpha
            alphas[i] = hist_alpha * 0.7 + score_boost + 0.5
            
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
        names = df_week['name'].values
        scores = df_week['score'].values
        status = df_week['status'].values
        
        # 识别淘汰者
        is_elim = np.array(['Eliminated' in s for s in status])
        elim_idx = np.where(is_elim)[0]
        safe_idx = np.where(~is_elim)[0]
        
        # 1. 获取先验 Alpha
        alphas = self.get_prior_alphas(names, scores)
        
        valid_samples = []
        
        # 2. 采样 (Batch Sampling)
        # 既然我们有了更好的 Prior，命中率会提高
        batch_size = num_samples
        
        # 生成 Dirichlet 分布
        # samples shape: (batch_size, n_participants)
        samples = np.random.dirichlet(alphas, batch_size)
        
        for v in samples:
            if check_validity(scores, v, elim_idx, safe_idx):
                valid_samples.append(v)
        
        # 3. 结果处理
        valid_samples = np.array(valid_samples)
        
        if len(valid_samples) < 10:
            # 如果样本太少，说明 Prior 可能误导了，或者约束太紧
            # 尝试回退到均匀分布再采一次 (Rescue Mode)
            fallback_samples = np.random.dirichlet(np.ones(len(names)), 2000)
            for v in fallback_samples:
                if check_validity(scores, v, elim_idx, safe_idx):
                    valid_samples = np.vstack([valid_samples, v]) if len(valid_samples) > 0 else np.array([v])
        
        if len(valid_samples) == 0:
            # print(f"  Warning: S{season} W{week} - No valid solution found. Using Prior Mean.")
            # 兜底：直接使用先验分布的均值 (Alpha / Sum(Alpha))
            final_est = alphas / np.sum(alphas)
            final_std = np.zeros_like(final_est) + 0.05 # 假定一个不确定性
        else:
            final_est = np.mean(valid_samples, axis=0)
            final_std = np.std(valid_samples, axis=0)
            
            # 4. 更新历史
            self.update_history(names, final_est)
            
        return final_est, final_std

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
                
                # If no scores, skip (not participating)
                if count == 0:
                    continue
                    
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
                    'status': current_status
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
            
            est_votes, est_std = estimator.solve_week(season, week, week_data)
            
            for i, (idx, row) in enumerate(week_data.iterrows()):
                results.append({
                    'season': season,
                    'week': week,
                    'name': row['name'],
                    'judge_score': row['score'],
                    'status': row['status'],
                    'est_vote_share': est_votes[i],
                    'est_vote_std': est_std[i]
                })
                
    # 保存结果
    res_df = pd.DataFrame(results)
    res_df.to_csv('e:/美赛/Q1_estimated_fan_votes_optimized.csv', index=False)
    print("Done. Saved to e:/美赛/Q1_estimated_fan_votes_optimized.csv")

if __name__ == "__main__":
    main()
