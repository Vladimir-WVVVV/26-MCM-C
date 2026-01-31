import pandas as pd
import numpy as np
import os
# from scipy.optimize import minimize  # Removed to keep zero-dependency

# ==========================================
# 工具函数 (Tools)
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
    验证一组粉丝投票是否符合淘汰结果 (硬约束)
    """
    n = len(judge_scores)
    
    # 1. Ranking Method
    if method == 'Ranking':
        judge_ranks = rankdata_min(-judge_scores)
        fan_ranks = rankdata_min(-fan_shares)
        metric = (judge_ranks + fan_ranks) + (fan_ranks / 1000.0)
        
    # 2. Percentage Method
    elif method == 'Percentage':
        if np.sum(judge_scores) > 0:
            score_pct = judge_scores / np.sum(judge_scores)
        else:
            score_pct = np.zeros(n)
        
        combined = 0.5 * score_pct + 0.5 * fan_shares
        metric = -combined 
        
    else:
        metric = np.random.rand(n)

    if len(eliminated_indices) == 0:
        return True 
        
    elim_metrics = metric[eliminated_indices]
    
    if len(safe_indices) > 0:
        safe_metrics = metric[safe_indices]
        # 严格约束：淘汰者的表现（Metric）必须比所有幸存者都差
        if np.min(elim_metrics) > np.max(safe_metrics):
            return True
        return False
    else:
        return True

def softmax(x):
    """
    Softmax function with numerical stability
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# ==========================================
# MAP 优化器 (Global Coarse Locator)
# ==========================================

class MAPOptimizer:
    def __init__(self, lambda_reg=0.1):
        self.lambda_reg = lambda_reg

    def solve(self, prior_mu, judge_scores, elim_indices, safe_indices, method='Ranking'):
        """
        使用简单的梯度下降寻找最可能的投票分布 (软约束)
        Zero-dependency implementation of MAP optimization
        """
        n = len(prior_mu)
        
        # Initialize logits (log of prior)
        logits = np.log(prior_mu + 1e-9)
        learning_rate = 0.01
        iterations = 200
        
        for _ in range(iterations):
            shares = softmax(logits)
            
            # Gradient of L2 Loss: 2 * (shares - prior)
            # But we are optimizing logits. 
            # Grad_logit = Jacobian_softmax^T * Grad_loss
            # Jacobian_softmax is complex.
            # Simplified Update: Move logits towards log(prior)
            # And apply penalty gradient.
            
            # 1. Prior Pull
            grad = (shares - prior_mu)
            
            # 2. Penalty Push (Simplified Heuristic Gradient)
            if method == 'Percentage':
                if np.sum(judge_scores) > 0:
                    s_pct = judge_scores / np.sum(judge_scores)
                else:
                    s_pct = np.zeros(n)
                
                c_shares = 0.5 * s_pct + 0.5 * shares
                
                # Identify violators
                # We want Min(Elim) < Max(Safe) is FALSE
                # i.e., Min(Elim) > Max(Safe) is TRUE (Eliminated performed worse/higher metric in Ranking? No.)
                # Percentage Method: Lower Combined = Eliminated.
                # So we want Min(Elim) < Max(Safe)
                # Wait, earlier check_validity says: 
                # Metric = -Combined. Eliminated has LARGER Metric (worse).
                # So -Combined(Elim) > -Combined(Safe) => Combined(Elim) < Combined(Safe).
                # Correct.
                
                # So if Min(Combined_Elim) > Max(Combined_Safe), it's a VIOLATION.
                # We need to DECREASE Elim shares and INCREASE Safe shares.
                
                if len(elim_indices) > 0 and len(safe_indices) > 0:
                    min_elim_idx = elim_indices[np.argmin(c_shares[elim_indices])]
                    max_safe_idx = safe_indices[np.argmax(c_shares[safe_indices])]
                    
                    if c_shares[min_elim_idx] > c_shares[max_safe_idx]:
                        # Violation!
                        # Gradient: Push min_elim DOWN, Push max_safe UP
                        grad[min_elim_idx] += 0.5 # Positive grad -> Logit decreases? No, usually x = x - lr * grad
                        # We want logit to decrease for elim. So grad should be positive?
                        # x_new = x - lr * grad. Yes.
                        
                        grad[max_safe_idx] -= 0.5 # Negative grad -> Logit increases
            
            # Update logits
            logits = logits - learning_rate * grad
            
        return softmax(logits)

# ==========================================
# 融合求解器 (Iterative MAP-Guided MCMC)
# ==========================================

class HybridSolver:
    def __init__(self):
        self.map_optimizer = MAPOptimizer()
        self.history_shares = {} # name -> [share_w1, share_w2, ...]
        
    def get_temporal_prior(self, name, current_ml_prior):
        """
        结合 ML 先验和历史平滑 (Kalman Filter 思想简化版)
        """
        if name not in self.history_shares or len(self.history_shares[name]) == 0:
            return current_ml_prior
        
        # Exponential Smoothing
        last_share = self.history_shares[name][-1]
        
        # Alpha=0.7 (Trust recent history more than general ML model)
        # return 0.7 * last_share + 0.3 * current_ml_prior
        
        # 为了防止滞后，我们还是主要信 ML，但是用 History 限制范围
        return 0.5 * last_share + 0.5 * current_ml_prior

    def solve_week(self, season, week, names, scores, statuses, industries, ages):
        n = len(names)
        
        # 1. Determine Method
        method = 'Ranking'
        try:
            if 3 <= int(season) <= 27:
                method = 'Percentage'
        except:
            pass
            
        elim_indices = np.where(statuses == 'Eliminated')[0]
        safe_indices = np.where(statuses == 'Safe')[0]
        
        # 2. Construct Prior (ML + Temporal)
        # 这里简化：假设 ML Prior 已经有了 (可以是均匀分布，或者简单的归一化 Score)
        # 实际应用中应调用 Q1 的 ML 模型。
        # 这里用 Score 作为简单的 Prior Proxy
        if np.sum(scores) > 0:
            base_prior = scores / np.sum(scores)
        else:
            base_prior = np.ones(n) / n
            
        # Temporal Adjustment
        priors = np.array([self.get_temporal_prior(name, base_prior[i]) for i, name in enumerate(names)])
        priors /= np.sum(priors) # Re-normalize
        
        # 3. Stage 1: Coarse MAP
        # 用优化方法快速找到一个“大概可行”的中心
        map_center = self.map_optimizer.solve(priors, scores, elim_indices, safe_indices, method)
        
        # Stage 2: Guided MCMC
        valid_samples = []
        concentration = 50.0
        alphas = map_center * concentration
        
        target_samples = 200 # Reduced from 1000 for speed
        max_attempts = 5000  # Reduced from 20000 for speed
        
        attempts = 0
        while len(valid_samples) < target_samples and attempts < max_attempts:
            batch_size = 100
            samples = np.random.dirichlet(alphas, batch_size)
            attempts += batch_size
            
            for i in range(batch_size):
                if check_validity(scores, samples[i], elim_indices, safe_indices, method):
                    valid_samples.append(samples[i])
            
            # Adaptive relaxation
            if len(valid_samples) < target_samples * 0.1 and attempts > 1000:
                concentration *= 0.8
                alphas = map_center * concentration
                
        if len(valid_samples) == 0:
            return map_center # Fallback to MAP if MCMC fails
            
        est_mean = np.mean(valid_samples, axis=0)
            
        # 5. Update History
        for i, name in enumerate(names):
            if name not in self.history_shares:
                self.history_shares[name] = []
            self.history_shares[name].append(est_mean[i])
            
        return est_mean

# ==========================================
# Main Execution
# ==========================================

def main():
    file_path = 'e:/美赛/Q1_estimated_fan_votes_optimized.csv' # Load existing data to re-run specific weeks
    if not os.path.exists(file_path):
        # Fallback to raw data if optimized not found
        file_path = 'e:/美赛/2026_MCM_Problem_C_Data.csv'
        
    # For demonstration, we simulate a Controversy Case (Bobby Bones S27 Week 5)
    # Or load raw data and run.
    # To be safe and quick, let's just show the logic runs.
    
    solver = HybridSolver()
    
    print("Initialize Hybrid Solver (Iterative MAP-Guided MCMC)...")
    print("Testing on Synthetic Controversy Case (Simulating S27 Week 5)...")
    
    # Mock Data for S27 Week 5 (Bobby Bones Low Score but Safe)
    names = ['Bobby Bones', 'Milo Manheim', 'Evanna Lynch', 'DeMarcus Ware', 'John Schneider']
    scores = np.array([20, 29, 28, 26, 24]) # Bobby lowest
    statuses = np.array(['Safe', 'Safe', 'Safe', 'Eliminated', 'Safe']) # DeMarcus Eliminated
    industries = ['Radio', 'Actor', 'Actor', 'Athlete', 'Actor']
    ages = [38, 17, 27, 36, 58]
    
    # Run Solver
    est_votes = solver.solve_week(27, 5, names, scores, statuses, industries, ages)
    
    print("\nResults for S27 Week 5:")
    for i, name in enumerate(names):
        print(f"{name}: Score={scores[i]}, Status={statuses[i]}, Est.Vote={est_votes[i]:.2%}")
        
    print("\n--- Verification ---")
    print("Check if Bobby Bones (Score 20) survived against DeMarcus (Score 26).")
    # Percentage Method in S27
    # Bobby Combined: 0.5*Score% + 0.5*Vote%
    # DeMarcus Combined: ...
    
    total_score = np.sum(scores)
    bobby_score_pct = 20 / total_score
    demarcus_score_pct = 26 / total_score
    
    bobby_vote = est_votes[0]
    demarcus_vote = est_votes[3]
    
    bobby_comb = 0.5 * bobby_score_pct + 0.5 * bobby_vote
    demarcus_comb = 0.5 * demarcus_score_pct + 0.5 * demarcus_vote
    
    print(f"Bobby Combined: {bobby_comb:.4f}")
    print(f"DeMarcus Combined: {demarcus_comb:.4f}")
    
    if bobby_comb > demarcus_comb:
        print("✅ SUCCESS: Bobby Saved (Higher Combined Score).")
    else:
        print("❌ FAIL: Bobby Eliminated.")

if __name__ == "__main__":
    main()
