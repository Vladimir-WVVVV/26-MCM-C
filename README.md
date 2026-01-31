# MCM Problem C 解决方案代码库 (Dancing with the Stars Solution)

本仓库包含了针对 MCM Problem C (Dancing with the Stars) 的完整解决方案代码。代码涵盖了数据预处理、观众投票估算、模型分析与优化、以及新赛制设计等四个主要任务。

## 📚 文档导航 (Documentation)

为了方便团队协作与成果展示，我们准备了以下详细文档：

### 汇总报告
*   **[📄 优化总结与核心成果 (Optimization Summary)](Optimization_Summary.md)**: 包含所有关键发现、数据结论和赛制改革建议。
*   **[⚙️ 技术实现文档 (Technical Documentation)](Technical_Documentation.md)**: 包含核心算法细节和系统架构。

### 分项详细报告 (Detailed Reports)
*   **[Q1 粉丝投票数据精准还原](Q1_Results_Report.md)**: 解释双引擎算法与 98.21% 准确率。
*   **[Q2 计分机制博弈分析](Q2_Results_Report.md)**: 解析排名法 vs 百分比法的公平性博弈。
*   **[Q3 评分影响因素归因](Q3_Results_Report.md)**: 揭示年龄、行业对评委和粉丝的不同影响。
*   **[Q4 最佳赛制机制设计](Q4_Results_Report.md)**: 阐述“粉丝中心论”动态权重优化方案。

---

## 📂 文件结构 (File Structure)

### 核心脚本 (Core Scripts)
| 文件名 | 描述 |
| :--- | :--- |
| **`Q1_hybrid_solver.py`** | **[Task 1]** 核心估算脚本。使用“迭代MAP引导MCMC”机制（Iterative MAP-Guided MCMC）反向估算粉丝投票。 |
| **`Q2_compare_methods.py`** | **[Task 2]** 排名法 vs 百分比法仿真脚本。包含反事实模拟、偏向性分析和争议案例复盘。 |
| **`Q3_analyze_factors.py`** | **[Task 3]** 归因分析脚本。使用混合非线性模型量化年龄、行业对评委分/粉丝票的影响。 |
| **`Q4_design_mechanism.py`** | **[Task 4]** 赛制设计脚本。实现多目标参数寻优，寻找最佳的动态权重组合。 |
| **`generate_charts.py`** | **[Viz]** 自动化绘图脚本。生成 Q1-Q4 的分析图表（保存在 `visualization_results/`）。 |

### 数据文件 (Data Files)
| 文件名 | 描述 |
| :--- | :--- |
| `Q1_estimated_fan_votes_optimized.csv` | **最终投票数据集**。包含 34 个赛季完整的评委分、估算粉丝得票率（准确率 98.21%）。 |
| `Q2_method_counterfactuals.csv` | Task 2 的反事实模拟结果，记录了每一周两种计分方法的差异。 |

---

## 🚀 快速开始 (Quick Start)

### 1. 环境准备
本项目坚持 **"Zero-Dependency"** 原则，仅依赖标准科学计算库。
```bash
pip install -r requirements.txt
```

### 2. 运行流程
建议按以下顺序执行，以确保数据流连贯：

```bash
# 1. [数据] 生成高精度投票估算数据 (必做)
python Q1_generate_final.py

# 2. [验证] 运行鲁棒性验证 (可选)
python Q1_evaluate_accuracy_hybrid.py

# 3. [分析] 运行排名法 vs 百分比法对比模拟
python Q2_compare_methods.py

# 3. [分析] 运行影响因素归因分析
python Q3_analyze_factors.py

# 4. [设计] 运行新赛制参数寻优与验证
python Q4_design_mechanism.py
```

---

## 📊 核心发现摘要 (Key Findings)

*   **精准还原**: 我们的双引擎模型在 335 个历史周次中实现了 **98.21%** 的还原准确率。
*   **排名法陷阱**: 排名法对“人气选手”的保护率高达 **99.7%**，是导致技术高分选手被误杀的主要原因（差异率 **17.01%**）。
*   **年龄悖论**: 评委严格遵循“年轻优势”（线性负相关），而粉丝对 60 岁以上的“传奇选手”展现出显著的宽容（U型曲线）。
*   **最优赛制**: 推荐采用 **"前期放权 (10/90) -> 后期收权 (45/55)" + "Bottom 3 评委拯救"** 机制，可将技术不公率降至 **<0.5%**。

---

## 🤝 贡献指南
请基于 `main` 分支进行开发。提交代码前请确保运行通过且不引入新的外部依赖（如 sklearn）。
