# 2026 美赛 C 题解决方案代码库

这里包含了我们针对 2026 MCM Problem C (Libertarian Elections) 的所有核心代码和生成的数据结果。

## 📂 文件结构说明

### 1️⃣ 第一问 (Task 1): 粉丝投票估算
*   **`Q1_estimate_votes_optimized.py`**  **(核心代码)**
    *   **作用**: 使用**自适应序列蒙特卡洛算法 (Adaptive Sequential Monte Carlo)**，根据历史淘汰结果反推每位选手的粉丝得票比例。
    *   **特点**: 包含自适应先验 (Adaptive Prior) 和时间平滑 (Temporal Smoothing)，不依赖 `scipy` 库。
*   **`Q1_evaluate_accuracy.py`**
    *   **作用**: 验证估算模型的准确性。
    *   **结果**: 准确率 **95.83%**，平均标准差 **0.06**。
*   **`Q1_estimated_fan_votes_optimized.csv`**
    *   **数据**: 模型生成的最终粉丝投票估算数据，包含 `est_vote_share` (估算得票率) 和 `est_vote_std` (置信度)。后续任务均基于此文件。
*   `Q1_estimate_votes.py`: *（备份）* 最初版本的蒙特卡洛代码。
*   `Q1_optimized.py`: *（参考）* 优化算法的参考实现。

### 2️⃣ 第二问 (Task 2): 投票方法对比
*   **`Q2_compare_methods.py`**
    *   **作用**: 对比分析 **排名法 (Ranking Method)** 与 **百分比法 (Percentage Method)** 的差异。
    *   **分析内容**: 模拟两种方法下的淘汰结果，重点分析“杰瑞·莱斯 (Jerry Rice)”等争议案例。
*   **`Q2_method_comparison.csv`**
    *   **数据**: 包含每一周在两种不同计分方式下的排名对比数据。

### 3️⃣ 第三问 (Task 3): 影响因素分析
*   **`Q3_analyze_factors.py`**
    *   **作用**: 数据预处理与整合，用于分析职业舞者、年龄、行业等特征对评分和投票的影响。
*   **`Q3_factor_analysis_data.csv`**
    *   **数据**: 整合了选手特征、评委评分和估算粉丝投票的综合数据集，可直接导入 SPSS/Stata/Python 进行回归分析。

### 4️⃣ 第四问 (Task 4): 新机制设计
*   **`Q4_design_mechanism.py`**
    *   **作用**: 模拟我们设计的新机制 **“百分比法 + 评委拯救 (Percentage + Judge Save)”**。
    *   **指标**: 计算了“人才保留率 (Talent Retention Score)”，验证新机制是否更公平。
*   **`Q4_simulation_results.csv`**
    *   **数据**: 详细的单周模拟对照表（真实结果 vs 三种不同机制的模拟结果）。
*   **`Q4_mechanism_comparison_summary.csv`**
    *   **数据**: 最终的评价指标汇总表。

## 🚀 如何运行
1. 确保已安装 `pandas` 和 `numpy`。
2. 按照顺序运行代码（通常 Q2-Q4 依赖 Q1 生成的数据）：
   ```bash
   python Q1_estimate_votes_optimized.py
   python Q1_evaluate_accuracy.py
   python Q2_compare_methods.py
   python Q3_analyze_factors.py
   python Q4_design_mechanism.py
   ```

## ⚠️ 注意事项
*   原始数据集文件 (`2026_MCM_Problem_C_Data.csv`) 和其他大型参考资料已被 `.gitignore` 忽略，不会上传到 GitHub。
*   如果拉取代码后报错“文件不存在”，请确保本地根目录下有原始数据文件。
