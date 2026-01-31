# -*- coding: utf-8 -*-
"""
Q1_evaluate_accuracy_hybrid_fixed.py

修复点：
1) 原脚本 calculate_accuracy() 里没有真正跑 trial，也没有向 accuracies 里 append，
   导致 np.mean/np.min 对空数组计算 -> RuntimeWarning / ValueError: zero-size array.
2) 增加：数据为空时直接退出并给提示。
3) 输出路径改为可配置 OUT_DIR，自动创建目录，避免 e:/美赛 不存在。
4) Windows 路径用 raw string，避免反斜杠转义问题。
5) 保留你原有的“检测宽表->转长表”的逻辑，尽量少改其它代码。

依赖：
- 同目录下需要存在 Q1_hybrid_solver.py，并提供 HybridSolver 类。
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

from Q1_hybrid_solver import HybridSolver  # noqa: F401


# =========================
# 0) 配置区：按你电脑修改
# =========================
DATA_PATH = r"C:\_Am\Data.csv"         # 你的原始CSV路径
OUT_DIR   = r"C:\_Am\q1_outputs"       # 报告输出目录（自动创建）
NUM_TRIALS = 5                         # 跑几次 robustness trial（如果 solver 有随机性）


def load_data():
    """读取数据；若检测到宽表(weekX_judgeY_score)，则转换为 long 表：season-week-contestant-judge_score-result"""
    file_path = DATA_PATH
    if not os.path.exists(file_path):
        print(f"[Error] Data file not found at {file_path}")
        return None

    df = pd.read_csv(file_path)

    # 宽表检测：官方C题原始数据通常包含 week1_judge1_score 等列
    if 'week1_judge1_score' in df.columns:
        print("Detected raw wide-format data. Converting to long format...")

        long_data = []
        max_weeks = 11  # 你原脚本写死 11；如果数据最大周数更大，可加一段自动探测

        for _, row in df.iterrows():
            try:
                season = row.get('season', None)
                name = row.get('celebrity_name', None)
                result_str = str(row.get('results', ''))

                # 粗略解析淘汰周
                elim_week = 999
                if 'Eliminated Week' in result_str:
                    try:
                        elim_week = int(result_str.split('Week')[-1].strip())
                    except Exception:
                        elim_week = 999

                for w in range(1, max_weeks + 1):
                    score_sum = 0.0
                    count = 0
                    for j in range(1, 5):
                        col = f'week{w}_judge{j}_score'
                        if col in df.columns:
                            val = row[col]
                            if pd.notna(val) and str(val).strip() not in ['N/A', '']:
                                try:
                                    score_sum += float(val)
                                    count += 1
                                except Exception:
                                    pass

                    if count > 0:
                        current_status = 'Safe'
                        if w == elim_week:
                            current_status = 'Eliminated'
                        elif w > elim_week:
                            # 淘汰后周次跳过
                            continue

                        long_data.append({
                            'season': season,
                            'week': w,
                            'contestant': name,
                            'judge_score': score_sum,
                            'result': current_status
                        })
            except Exception:
                pass

        df = pd.DataFrame(long_data)

    # 基础清洗
    if df is None or len(df) == 0:
        return pd.DataFrame()

    df = df.dropna(subset=['season', 'week', 'contestant', 'judge_score', 'result'])
    return df


def run_single_trial(trial_id, df, log_fp=None):
    """跑一轮 trial：逐周调用 HybridSolver 求 vote share，然后检查是否与真实淘汰一致。"""
    if log_fp:
        log_fp.write(f"[Trial {trial_id+1}] start, df={df.shape}\n")
        log_fp.flush()

    solver = HybridSolver()  # 每个 trial 新建实例以清空历史状态
    grouped = df.groupby(['season', 'week'])
    sorted_groups = sorted(grouped, key=lambda x: (x[0][0], x[0][1]))

    total_weeks = 0
    correct_weeks = 0

    for (season, week), group in sorted_groups:
        names = group['contestant'].values
        scores = group['judge_score'].values.astype(float)
        statuses = group['result'].values

        # Dummy industries/ages（HybridSolver 的接口需要）
        industries = ['Unknown'] * len(names)
        ages = [0] * len(names)

        try:
            est_votes = solver.solve_week(season, week, names, scores, statuses, industries, ages)

            # === 一致性验算（与你原脚本一致） ===
            method = 'Ranking'
            if 3 <= int(season) <= 27:
                method = 'Percentage'

            is_correct = True

            if method == 'Ranking':
                def rankdata_min(a):
                    n = len(a)
                    ranks = np.zeros(n, dtype=int)
                    for i in range(n):
                        rank = 1
                        for j in range(n):
                            if a[j] < a[i]:
                                rank += 1
                        ranks[i] = rank
                    return ranks

                judge_ranks = rankdata_min(-scores)
                fan_ranks = rankdata_min(-est_votes)
                metric = (judge_ranks + fan_ranks) + (fan_ranks / 1000.0)  # 越小越好

                elim_idx = np.where(statuses == 'Eliminated')[0]
                safe_idx = np.where(statuses == 'Safe')[0]
                if len(elim_idx) > 0 and len(safe_idx) > 0:
                    min_elim = np.min(metric[elim_idx])
                    max_safe = np.max(metric[safe_idx])
                    if min_elim <= max_safe:
                        is_correct = False

            else:  # Percentage
                if np.sum(scores) > 0:
                    s_pct = scores / np.sum(scores)
                else:
                    s_pct = np.zeros(len(scores))

                combined = 0.5 * s_pct + 0.5 * est_votes  # 越大越好
                elim_idx = np.where(statuses == 'Eliminated')[0]
                safe_idx = np.where(statuses == 'Safe')[0]
                if len(elim_idx) > 0 and len(safe_idx) > 0:
                    max_elim = np.max(combined[elim_idx])
                    min_safe = np.min(combined[safe_idx])
                    if max_elim >= min_safe:
                        is_correct = False

            if is_correct:
                correct_weeks += 1
            total_weeks += 1

        except Exception as e:
            if log_fp:
                log_fp.write(f"[Trial {trial_id+1}] Error in S{season} W{week}: {e}\n")
                log_fp.flush()
            total_weeks += 1

    acc = correct_weeks / total_weeks if total_weeks > 0 else 0.0
    return acc, correct_weeks, total_weeks


def calculate_accuracy():
    print("Loading data...")
    df = load_data()
    if df is None:
        return

    if len(df) == 0:
        print("[Error] After preprocessing, df is empty. 请检查：")
        print("  1) DATA_PATH 是否指向正确的原始CSV；")
        print("  2) 原始CSV是否确实包含 weekX_judgeY_score 列；")
        print("  3) results 字段是否存在、且格式包含 'Eliminated Week X'。")
        return

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    report_path = str(Path(OUT_DIR) / "accuracy_report.txt")
    log_path = str(Path(OUT_DIR) / "accuracy_debug.log")

    accuracies = []
    with open(log_path, "w", encoding="utf-8") as log_fp:
        for t in range(NUM_TRIALS):
            acc, correct, total = run_single_trial(t, df, log_fp=log_fp)
            accuracies.append(acc)
            print(f"Trial {t+1}/{NUM_TRIALS}: acc={acc:.4%} ({correct}/{total})")

    # === 修复：accuracies 不为空 ===
    avg_acc = float(np.mean(accuracies))
    min_acc = float(np.min(accuracies))
    max_acc = float(np.max(accuracies))
    std_acc = float(np.std(accuracies))

    with open(report_path, 'w', encoding="utf-8") as f:
        f.write("="*56 + "\n")
        f.write(f"Final Robustness Evaluation Results ({NUM_TRIALS} Trials)\n")
        f.write("="*56 + "\n")
        f.write(f"Average Accuracy: {avg_acc:.4%}\n")
        f.write(f"Min Accuracy:     {min_acc:.4%}\n")
        f.write(f"Max Accuracy:     {max_acc:.4%}\n")
        f.write(f"Std Deviation:    {std_acc:.6f}\n")
        f.write(f"Stability:        {'Perfectly Stable' if std_acc < 1e-6 else 'Slight Variance'}\n")
        f.write("="*56 + "\n")

    print("\n" + "="*56)
    print(f"Final Robustness Evaluation Results ({NUM_TRIALS} Trials)")
    print("="*56)
    print(f"Average Accuracy: {avg_acc:.4%}")
    print(f"Min Accuracy:     {min_acc:.4%}")
    print(f"Max Accuracy:     {max_acc:.4%}")
    print(f"Std Deviation:    {std_acc:.6f}")
    print(f"Stability:        {'Perfectly Stable' if std_acc < 1e-6 else 'Slight Variance'}")
    print("="*56)
    print(f"[Saved] {report_path}")
    print(f"[Saved] {log_path}")


if __name__ == "__main__":
    calculate_accuracy()
