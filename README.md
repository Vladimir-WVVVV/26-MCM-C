# MCM Problem C 解决方案代码库

本仓库包含了针对 MCM Problem C (Dancing with the Stars) 的完整解决方案代码。代码涵盖了数据预处理、观众投票估算、模型分析与优化、以及新赛制设计等四个主要任务。

## 文件结构说明

### 核心代码
- **Q1_estimate_votes_optimized.py**: 任务1核心代码。使用自适应序列蒙特卡洛方法（Adaptive Sequential Monte Carlo）估算每周的观众投票数。包含数据清洗、特征提取和自适应先验构建。
- **Q1_enhance_model_ml.py**: 任务1增强模型。使用岭回归（Ridge Regression）和多项式特征进一步优化和校准观众投票估算值，利用选手特征（年龄、行业）提高预测精度。
- **Q1_evaluate_accuracy.py**: 评估Q1估算模型的一致性准确率（Accuracy）和确定性（Certainty）。
- **Q2_compare_methods.py**: 任务2核心代码。对比“排名法”与“百分比法”的差异，量化偏向性，并对争议选手（如Jerry Rice, Bobby Bones）进行专项分析。
- **Q3_analyze_factors.py**: 任务3核心代码。建立多元回归模型，量化分析年龄、行业（如运动员、模特）等因素对“评委评分”和“观众投票”的不同影响。
- **Q4_design_mechanism.py**: 任务4核心代码。设计并模拟新的“动态权重 + 双阶段技术淘汰”机制，评估其在公平性和观赏性上的表现。

### 结果文件
- **Q1_estimated_fan_votes_optimized.csv**: 基础优化模型的投票估算结果。
- **Q1_enhanced_ml_estimates.csv**: 机器学习增强后的最终投票估算结果（推荐使用）。
- **Q2_controversy_analysis.csv**: 争议选手在不同赛制下的生存情况分析。
- **Q3_factor_regression_results.csv**: 因素影响分析的详细回归系数。
- **Q4_simulation_results.csv**: 新赛制模拟的详细结果。

## 如何运行

1. **环境准备**:
   确保安装了 Python 3.x 以及以下库：
   ```bash
   pip install pandas numpy
   ```
   *注意：代码已针对无 `scipy` 和 `sklearn` 环境进行了适配，仅依赖 `numpy` 和 `pandas` 即可运行所有核心功能。*

2. **按顺序运行**:
   建议按照以下顺序运行脚本，以确保数据流的正确性：
   
   ```bash
   # 1. 生成基础估算数据
   python Q1_estimate_votes_optimized.py
   
   # 2. 生成增强ML估算数据 (依赖上一步)
   python Q1_enhance_model_ml.py
   
   # 3. 运行分析任务
   python Q2_compare_methods.py
   python Q3_analyze_factors.py
   
   # 4. 运行新赛制模拟
   python Q4_design_mechanism.py
   ```

## 核心发现摘要

- **投票估算**: 我们的模型达到了约 78% 的预测准确度（R2 Score），成功复现了绝大多数淘汰结果。
- **方法对比**: “排名法”略微偏向保护高人气选手，而“百分比法”更容易淘汰虽然人气高但评委分极低的争议选手。
- **因素分析**: 
  - **年龄**: 年长选手在评委打分上明显吃亏，但观众投票对其相对宽容。
  - **运动员**: 极受观众欢迎（影响系数 +0.56），但评委对其无特殊偏好。
  - **模特**: 在评委和观众两端均表现不佳。
- **新赛制**: 设计的“动态权重 + Bottom 3 技术淘汰”机制，在模拟中仅有 0.3% 的概率误淘汰技术前三的选手，显著提升了比赛的技术公平性，同时保留了适度的观众影响力。

## 团队协作
请基于 `main` 分支进行开发。提交代码前请确保本地运行通过。
