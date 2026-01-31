# 项目文件结构与使用说明 (Project File Guide)

本文档详细介绍了本项目中每个文件的作用、功能以及使用方法，旨在帮助团队成员快速理解项目结构并进行复现或扩展。

## 0. 环境配置 (Environment Setup)

在运行任何代码之前，请确保已安装必要的 Python 依赖库。

```bash
pip install -r requirements.txt
```

## 1. 核心模型与数据生成 (Core Model & Data Generation)

这部分代码是整个项目的基石，负责实现高精度（98.21%）的投票率估算模型，并生成后续分析所需的基础数据。

### `Q1_hybrid_solver.py`
*   **类型**: 核心算法库 (Class Library)
*   **作用**: 定义了 `HybridSolver` 类。这是本项目的核心引擎，结合了 **MAP（最大后验概率）** 定位和 **MCMC（马尔可夫链蒙特卡洛）** 搜索，并引入了 **时序平滑（Temporal Smoothing）** 机制。
*   **关键特性**: 零依赖（纯 NumPy 实现），支持 Ranking 和 Percentage 两种赛制规则，能够自动处理淘汰约束。
*   **如何使用**: 被其他脚本引用（`from Q1_hybrid_solver import HybridSolver`），通常不需要直接运行。

### `Q1_generate_final.py`
*   **类型**: 数据生产脚本 (Data Generation Script)
*   **作用**: 读取原始数据，调用 `HybridSolver` 对全历史（S01-S34）进行逐周估算，并将结果（包括估算的粉丝投票率、选手年龄、行业等特征）整合保存为 CSV 文件。
*   **输出**: `Q1_estimated_fan_votes_optimized.csv`
*   **如何使用**:
    ```bash
    python Q1_generate_final.py
    ```
    *注意：这是进行 Q2-Q4 分析前的必做步骤。*

### `Q1_estimated_fan_votes_optimized.csv`
*   **类型**: 核心数据文件 (Dataset)
*   **作用**: `Q1_generate_final.py` 的产出物。包含了比赛所有周次的详细数据，是 Q2、Q3、Q4 分析的**唯一数据源**。
*   **包含字段**: Season, Week, Name, Score (评委分), Est_Vote_Share (估算粉丝票), Status, Industry, Age 等。

---

## 2. 模型验证 (Validation)

### `Q1_evaluate_accuracy_hybrid.py`
*   **类型**: 测试脚本 (Test Script)
*   **作用**: 对核心模型进行严格的准确率测试。采用 **5轮鲁棒性测试**（5-Trial Robustness Test），确保准确率（98.21%）不是偶然结果。
*   **输出**: 控制台打印每一轮的准确率统计。
*   **如何使用**:
    ```bash
    python Q1_evaluate_accuracy_hybrid.py
    ```

---

## 3. 问题分析与求解 (Analysis & Solutions)

这部分脚本分别对应赛题的 Q2、Q3、Q4 问题，基于 Q1 生成的数据进行深入分析。

### `Q2_compare_methods.py`
*   **类型**: 分析脚本 (Analysis Script)
*   **作用**: **(Q2 赛制对比)** 模拟“如果用排名法代替百分比法（或反之）会发生什么”。
*   **核心逻辑**: 反事实模拟（Counterfactual Simulation）。计算两种方法对“粉丝最爱”和“评委最爱”选手的保护能力差异。
*   **输出**:
    1.  `Q2_method_counterfactuals.csv`: 详细的模拟结果表。
    2.  控制台输出：两种方法的偏差率（17.01%）及各类保护率指标。
*   **如何使用**:
    ```bash
    python Q2_compare_methods.py
    ```

### `Q3_analyze_factors.py`
*   **类型**: 分析脚本 (Analysis Script)
*   **作用**: **(Q3 归因分析)** 量化评委评分和粉丝投票的影响因子。
*   **核心逻辑**: 使用混合回归模型（Hybrid Regression），分析年龄（线性 vs 非线性）和行业（职业背景）对成绩的影响。
*   **输出**: 控制台打印回归系数（Coefficients），例如评委对年龄的惩罚系数、粉丝对特定职业的偏好等。
*   **如何使用**:
    ```bash
    python Q3_analyze_factors.py
    ```

### `Q4_design_mechanism.py`
*   **类型**: 优化脚本 (Optimization Script)
*   **作用**: **(Q4 机制设计)** 寻找最优的赛制权重组合。
*   **核心逻辑**: 多目标网格搜索（Multi-Objective Grid Search）。在公平性（Fairness）和保留率（Retention）之间寻找最佳平衡点，并测试“评委拯救机制”的效果。
*   **输出**: 控制台打印最优的权重参数（如前期 0.3 / 后期 0.6）及其带来的性能提升。
*   **如何使用**:
    ```bash
    python Q4_design_mechanism.py
    ```

---

## 4. 项目文档 (Documentation)

### `Optimization_Summary.md`
*   **类型**: 成果汇报文档
*   **作用**: 面向非技术人员或团队队友。高度概括了项目的核心成果、关键数据指标（如 98.21% 准确率、17.01% 差异率）和最终结论。
*   **使用场景**: 写论文或做展示时的主要参考材料。

### `Technical_Documentation.md`
*   **类型**: 技术细节文档
*   **作用**: 面向开发人员。详细解释了算法原理（MAP、MCMC、时序平滑）、数学公式、代码架构和环境依赖。
*   **使用场景**: 代码维护、算法复现或技术审查。

### `README.md`
*   **类型**: 项目入口
*   **作用**: GitHub 仓库的首页。提供项目的背景介绍、环境安装指南和快速导航链接。

---

## 5. 建议运行顺序 (Recommended Workflow)

如果您刚刚拿到本项目，建议按照以下顺序操作：

1.  **数据生成**: 运行 `python Q1_generate_final.py` (确保生成最新数据)
2.  **模型验证**: 运行 `python Q1_evaluate_accuracy_hybrid.py` (确认模型精度)
3.  **结果复现**: 依次运行 Q2, Q3, Q4 的脚本，查看各问题的分析结果。
4.  **阅读文档**: 参考 `Optimization_Summary.md` 撰写最终报告。
