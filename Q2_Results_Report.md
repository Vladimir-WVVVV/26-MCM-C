# Q2 核心成果：计分机制博弈分析 (Ranking vs Percentage)

## 1. 核心任务
对比两种核心计分机制对比赛结果的深层影响：
*   **排名法 (Ranking Method)**：将评委分和粉丝票转化为排名（1, 2, ...），再相加。
*   **百分比法 (Percentage Method)**：将两者转化为百分比得分，再加权相加。

## 2. 定量分析结果
基于 Q1 还原的高精度数据，我们进行了全历史周次的反事实模拟（Counterfactual Simulation）：

*   **差异率 (Disagreement Rate)**：**17.91%**
    *   这意味着在约 18% 的淘汰周次中，如果改变计分规则，被淘汰的人就会改变。赛制决定命运！

*   **公平性与保护力分析**：
    *   **评委偏好保护 (Judge Favorites)**：
        *   **Ranking Method**: 96.5% 存活率（更尊重技术）
        *   **Percentage Method**: 92.0% 存活率
        *   **结论**：Ranking Method 是技术流选手的“安全网”，能有效防止高分低人气的专业舞者意外出局。
    *   **粉丝偏好保护 (Fan Favorites)**：
        *   两者差异不大（~99.7%），只要人气够高，任何赛制都很难淘汰你。

## 3. 经典案例复盘 (Case Studies)
*   **Bristol Palin (Season 11, Week 6)**：
    *   **Ranking下场**：被淘汰。
    *   **Percentage下场**：Audrina Patridge 被淘汰，Bristol 存活。
    *   **解读**：百分比法放大了 Bristol 极高的粉丝投票权重，掩盖了她较低的评委分。

## 4. 结论
*   **Ranking Method** 更加**温和且精英主义**，倾向于保护评委认可的技术型选手。
*   **Percentage Method** 更加**残酷且民粹主义**，它会无情地淘汰掉那些虽然技术不错但缺乏粉丝基础的选手。
