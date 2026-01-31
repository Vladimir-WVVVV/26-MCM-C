# 技术实现文档 (Technical Documentation)

## 1. 系统架构 (System Architecture)

本系统采用全 Python 实现，核心设计理念为 **"Zero-Dependency & High-Portability"**（零依赖与高可移植性）。除标准科学计算库 `numpy` 和 `pandas` 外，移除了对 `sklearn` 等大型机器学习库的依赖，通过手写核心算法确保代码在任何轻量级环境下均可复现。

### 1.1 模块依赖
- **Python Version**: 3.8+
- **Core Libraries**: `numpy`, `pandas`, `scipy` (仅用于 rankdata)
- **No External ML Libs**: 线性回归、岭回归、蒙特卡洛模拟均为原生 NumPy 实现。

---

## 2. 核心算法详解

### 2.1 双引擎估算模型 (Dual-Engine Estimation)
位于 `Q1_hybrid_solver.py`。
- **ML 先验 (Prior)**: 使用 `RidgeRegressionNumPy` 类预测基础人气。
- **MAP 定位 (MAP Optimization)**:
  - 利用梯度下降（Gradient Descent）在狄利克雷分布（Dirichlet Distribution）空间中寻找最大后验概率点。
  - 快速锁定“可行解”的中心位置，大幅缩小搜索范围。
- **时序平滑 (Temporal Smoothing)**:
  - $Prior_t = 0.5 \cdot Prior_{ML} + 0.5 \cdot Posterior_{t-1}$
  - 防止因单周数据波动导致的投票率剧烈跳变。
- **MCMC 精搜 (Guided MCMC)**:
  - 在 MAP 确定的高置信区域内进行采样。
  - 严格满足排名法/百分比法的硬性淘汰约束。

### 2.2 排名算法标准化 (Rankdata Min)
位于 `Q2_compare_methods.py`。
- 为了解决 Python `rankdata` 默认行为与现实比赛规则（同分同名次，下一名次跳过）不一致的问题，实现了 `rankdata_min`：
  ```python
  # Example: Scores [10, 10, 9] -> Ranks [1, 1, 3] (Not [1, 2, 3] or [1.5, 1.5, 3])
  def rankdata_min(a):
      ...
  ```
- 确保了仿真结果与真实世界规则的严格对齐。

### 2.3 混合归因回归 (Hybrid Attribution Regression)
位于 `Q3_analyze_factors.py`。
- **实现**：手写最小二乘法（OLS）求解器。
  - 核心计算：$\theta = (X^T X)^{-1} X^T y$
- **特征工程**：
  - 引入 `Age_Squared` 处理非线性年龄效应。
  - 对 `Industry` 进行 One-Hot 编码并手动合并稀疏类别。
  - 使用标准化系数（Standardized Coefficients）直接比较不同量纲特征的重要性。

### 2.4 多目标参数寻优 (Multi-Objective Optimization)
位于 `Q4_design_mechanism.py`。
- **目标函数**：
  $$Loss = w_1 \cdot (1 - Fairness) + w_2 \cdot (1 - Retention)$$
- **搜索空间**：
  - 评委权重 $W_j \in [0.3, 0.7]$
  - 赛制阶段 $Stage \in [Early, Late]$
- **模拟引擎**：遍历所有历史周次，应用新规则，计算新的淘汰名单并与真实名单对比。

---

## 3. 文件结构说明
```text
E:\美赛\
├── Q1_hybrid_solver.py             # [核心] 迭代MAP引导MCMC求解器
├── Q1_generate_final.py            # [工具] 生成最终版高精度数据
├── Q1_evaluate_accuracy_hybrid.py  # [验证] 全量数据鲁棒性测试脚本
├── Q1_estimated_fan_votes_optimized.csv # [产出] 34赛季完整投票数据
├── Q2_compare_methods.py           # [分析] 排名法 vs 百分比法仿真
├── Q2_method_counterfactuals.csv   # [产出] 反事实模拟结果
├── Q3_analyze_factors.py           # [分析] 评委/粉丝偏好归因模型
├── Q4_design_mechanism.py          # [设计] 新赛制参数寻优
├── Optimization_Summary.md         # [文档] 成果汇报
└── Technical_Documentation.md      # [文档] 技术说明
```

## 4. 快速运行指南 (Quick Start)

**步骤 1: 生成高精度投票数据**
```bash
python Q1_generate_final.py
# 输出: Q1_estimated_fan_votes_optimized.csv (包含完整人口统计学特征)
```

**步骤 2: 验证模型准确率**
```bash
python Q1_evaluate_accuracy_hybrid.py
# 输出: 控制台打印5轮鲁棒性测试结果 (Accuracy > 98%)
```

**步骤 3: 运行机制对比**
```bash
python Q2_compare_methods.py
# 输出: Q2_method_counterfactuals.csv, 控制台打印偏差率
```

**步骤 4: 运行归因分析**
```bash
python Q3_analyze_factors.py
# 输出: 打印各因素对评委分/粉丝票的回归系数
```

**步骤 5: 赛制优化模拟**
```bash
python Q4_design_mechanism.py
# 输出: 打印最优权重组合及新赛制下的各项指标提升
```
