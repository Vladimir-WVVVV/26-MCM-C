
"""
2026 MCM/ICM Problem C - Q1 (优化版，中文注释)

你将得到：
1) q1_df_votes.csv  : 每个 season-week-选手 的反推投票份额 vote_share、合成结果（适配 Rank/Percentage）
2) q1_metrics.csv   : 每个 season-week 的一致性指标（acc_single / cover_bottomk / jaccard / delta_margin）
3) q1_elim_by_week.csv : 从数据推断的每周淘汰名单 true_elims（集合差分法）
4) q1_elim_people.csv  : 每位选手最后存活周、是否决赛结束（用于复核）

核心思想（写报告时可用）：
- 观众投票为潜变量；可识别的是“相对投票份额”而非绝对票数；
- 每周用 MAP（最大后验）反演 vote_share：
  * softmax(x) 作为投票份额（非负且和为1）
  * x 接近先验中心 mu（由 Ridge 生成）+ L2 正则避免极端
  * 用“概率淘汰损失”（softmin likelihood）鼓励复现真实淘汰
- 自动按 season 切换赛制（Rank vs Percentage）
- 用上一周 x_hat 做时间平滑与 warm start（让票份额随周更平滑、更稳）

依赖：
pip install numpy pandas scipy scikit-learn
"""

import os
import re
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ===================== 你需要改的两个参数 =====================
# Windows 路径建议写成：CSV_PATH = r"C:\_Am\Data.csv"
CSV_PATH = r"/mnt/data/2026_MCM_Problem_C_Data.csv"
OUT_DIR  = r"/mnt/data/q1_outputs_test"
# ============================================================

# 是否输出 Excel（同名 .xlsx），默认 False（CSV 足够）
SAVE_EXCEL = False

# Step5（可选）：bootstrap 不确定性（只建议对少数周做）
RUN_BOOTSTRAP = False
BOOTSTRAP_SEASON = 1
BOOTSTRAP_WEEK   = 1
BOOTSTRAP_B = 200

# 输出编码（Windows Excel 友好）
CSV_ENCODING = "utf-8-sig"


# ------------------------------------------------------------
# 工具：安全保存
# ------------------------------------------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_csv(df: pd.DataFrame, filepath: str):
    df.to_csv(filepath, index=False, encoding=CSV_ENCODING)

def save_excel(df: pd.DataFrame, filepath: str, sheet_name: str = "sheet1"):
    # 只在需要时写 xlsx（避免额外开销）
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)


# ============================================================
# Step 0：读取宽表 + 宽转长
# ============================================================
def load_raw(csv_path: str) -> pd.DataFrame:
    """
    输入：宽表（包含 week1_judge1_score ...）
    输出：长表（每行是一条 season-week-选手-评委 的评分）
    """
    df = pd.read_csv(csv_path)

    # 必需列检查（避免“列名不一致”导致静默错误）
    base_cols = [
        "celebrity_name",
        "ballroom_partner",
        "celebrity_industry",
        "celebrity_homestate",
        "celebrity_homecountry/region",
        "celebrity_age_during_season",
        "season",
        "results",
        "placement",
    ]
    missing = [c for c in base_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少必要列：{missing}。请检查列名是否一致。")

    # 所有周-评委分数列
    score_cols = [c for c in df.columns if re.match(r"week\d+_judge\d+_score$", str(c))]
    if not score_cols:
        raise ValueError("CSV 中没找到 week1_judge1_score 这种列名，请检查文件。")

    # 宽转长
    long = df[base_cols + score_cols].melt(
        id_vars=base_cols,
        value_vars=score_cols,
        var_name="wk_judge",
        value_name="score",
    )

    # 解析 week/judge
    m = long["wk_judge"].str.extract(r"week(?P<week>\d+)_judge(?P<judge>\d+)_score")
    long["week"] = m["week"].astype(int)
    long["judge"] = m["judge"].astype(int)
    long.drop(columns=["wk_judge"], inplace=True)

    # 分数可能包含 'N/A' 等字符串，强制转为数值，无法转换的记为 NaN
    long["score"] = pd.to_numeric(long["score"], errors="coerce")

    return long


# ============================================================
# Step 0.5：构建周表：评委总分 J + alive 标记
# ============================================================
def build_week_table(long_scores: pd.DataFrame) -> pd.DataFrame:
    """
    输出 week_tbl：每行是 (season, week, celebrity_name) 的当周信息
    - J：当周评委总分（NaN 忽略）
    - alive：J>0 视为当周仍在比赛
    """
    week_tbl = (
        long_scores
        .groupby(["season", "week", "celebrity_name"], as_index=False)
        .agg(
            J=("score", lambda s: np.nansum(s.to_numpy())),
            ballroom_partner=("ballroom_partner", "first"),
            industry=("celebrity_industry", "first"),
            homestate=("celebrity_homestate", "first"),
            homecountry=("celebrity_homecountry/region", "first"),
            age=("celebrity_age_during_season", "first"),
            placement=("placement", "first"),
            results=("results", "first"),
        )
    )
    week_tbl["alive"] = (week_tbl["J"] > 0).astype(int)
    return week_tbl


# ============================================================
# Step 1：推断每周淘汰者（集合差分法，修复“未播出周”陷阱）
# ============================================================
def infer_eliminations(week_tbl: pd.DataFrame):
    """
    关键修复点：
    - 很多赛季并非 11 周全播，未播出的周会出现全 NaN -> J=0 -> alive=0
    - 如果把这些“未播出周”也当作 week 序列，会把决赛周误判为“大淘汰”
    解决：
    - weeks 只取“实际播出周”：至少有人 alive==1 的 week
    - 淘汰集合 E_t = S_t \ S_next（下一场比赛周）
    """
    rows = []
    season_last_rows = []

    for season, g in week_tbl.groupby("season"):
        # ✅ 只保留“实际播出周”（至少有人 alive==1）
        weeks = sorted(g[g["alive"] == 1]["week"].unique().tolist())
        if len(weeks) == 0:
            continue

        last_week = int(max(weeks))
        season_last_rows.append({"season": int(season), "season_last_week": last_week})

        # 逐周差分（最后一周没有 next，不推淘汰）
        for idx in range(len(weeks) - 1):
            t = int(weeks[idx])
            t_next = int(weeks[idx + 1])

            S_t = set(g[(g["week"] == t) & (g["alive"] == 1)]["celebrity_name"].astype(str))
            S_next = set(g[(g["week"] == t_next) & (g["alive"] == 1)]["celebrity_name"].astype(str))

            elim = sorted(list(S_t - S_next))
            if elim:
                rows.append({"season": int(season), "week": t, "true_elims": elim})

    elim_by_week = pd.DataFrame(rows).sort_values(["season", "week"]).reset_index(drop=True)
    season_last = pd.DataFrame(season_last_rows).sort_values(["season"]).reset_index(drop=True)

    # 额外输出：每个选手最后存活周（便于复核）
    last_alive = (
        week_tbl[week_tbl["alive"] == 1]
        .groupby(["season", "celebrity_name"])["week"]
        .max()
        .rename("last_alive_week")
        .reset_index()
    )
    elim_people = last_alive.merge(season_last, on="season", how="left")
    elim_people["is_finale_end"] = (elim_people["last_alive_week"] == elim_people["season_last_week"]).astype(int)

    # 自检：最后播出周不应该出现在 elim_by_week 中（否则说明数据/推断有问题）
    if not elim_by_week.empty and not season_last.empty:
        tmp = elim_by_week.merge(season_last, on="season", how="left")
        bad = tmp[tmp["week"] >= tmp["season_last_week"]]
        if len(bad) > 0:
            print("⚠️ 警告：发现最后播出周也被推断出淘汰，可能仍混入未播出周或周序不连续：")
            print(bad.head(20))

    return season_last, elim_people, elim_by_week


# ============================================================
# Step 2：先验中心 mu（Ridge，弱监督 proxy）
# ============================================================
def fit_prior_mu(week_tbl: pd.DataFrame):
    """
    先验中心 mu 的作用：
    - 投票不可识别：满足淘汰的投票解有无穷多
    - 我们用 mu 指定“更合理/更保守”的中心，让解不会乱飞

    训练信号（弱监督）：
      y0 = log(J + 1)

    特征（可解释且易跑）：
    - 数值：J、age、week
    - 类别：industry、homecountry、homestate、ballroom_partner
    """
    df = week_tbl.copy()

    train = df[df["alive"] == 1].copy()
    y0 = np.log(train["J"].clip(lower=0) + 1.0)

    num_cols = ["J", "age", "week"]
    cat_cols = ["industry", "homecountry", "homestate", "ballroom_partner"]

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    model = Ridge(alpha=1.0, random_state=0)
    pipe = Pipeline([("pre", pre), ("ridge", model)])
    pipe.fit(train, y0)

    df["mu"] = pipe.predict(df)
    return df, pipe


# ============================================================
# Step 3：每周 MAP 反演 vote_share（赛制自动切换 + 概率淘汰损失 + 时间平滑）
# ============================================================
def softmax(x: np.ndarray) -> np.ndarray:
    """把任意实数向量映射为概率分布（非负且和为1）"""
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-12)

def season_rule(season: int) -> str:
    """
    赛制切换（题面常见设定）：
    - Season 1-2 与 28+：Rank
    - Season 3-27：Percentage
    """
    if season in (1, 2) or season >= 28:
        return "rank"
    return "percentage"

def soft_rank(values: np.ndarray, temp: float = 0.05) -> np.ndarray:
    """
    可微“软名次”（数值越大越好 -> 名次越小越好）
      rank_i = 1 + sum_{j!=i} sigmoid((v_j - v_i)/temp)
    """
    v = np.asarray(values, dtype=float)
    diff = (v.reshape(1, -1) - v.reshape(-1, 1)) / max(temp, 1e-6)
    sig = 1.0 / (1.0 + np.exp(-diff))
    np.fill_diagonal(sig, 0.0)
    return 1.0 + np.sum(sig, axis=1)

class Q1Config:
    """
    lam: L2 正则（防极端）
    gamma: 淘汰一致性项权重（越大越强制复现淘汰）
    tau: 概率淘汰温度（越小越接近硬约束）
    rank_temp: soft-rank 温度（越小越接近真实名次，但梯度更尖锐）
    smooth_w: 周间平滑权重（0~1）
    scaled_total_votes: 仅用于展示（将份额乘一个总票数尺度）
    """
    def __init__(
        self,
        lam=0.1,
        gamma=120.0,
        tau=0.02,
        x_bounds=(-12, 12),
        rank_temp=0.05,
        smooth_w=0.30,
        scaled_total_votes=1_000_000,
    ):
        self.lam = lam
        self.gamma = gamma
        self.tau = tau
        self.x_bounds = x_bounds
        self.rank_temp = rank_temp
        self.smooth_w = smooth_w
        self.scaled_total_votes = scaled_total_votes

def compute_elim_score_and_extras(season: int, J: np.ndarray, vote_share: np.ndarray, cfg: Q1Config):
    """
    返回：
    - elim_score：越大表示越“差/更可能淘汰”（统一标准，便于排名）
      * Percentage：elim_score = -C（C越小越差）
      * Rank：elim_score = combined_rank（名次和越大越差）
    - extras：用于写作/画图的中间量
    """
    rule = season_rule(int(season))
    J = np.asarray(J, dtype=float)
    vote_share = np.asarray(vote_share, dtype=float)

    if rule == "percentage":
        judge_share = J / (J.sum() + 1e-12)
        C = 0.5 * judge_share + 0.5 * vote_share
        elim_score = -C
        extras = {
            "rule": rule,
            "C": C,
            "judge_share": judge_share,
            "judge_rank": np.full_like(C, np.nan),
            "vote_rank": np.full_like(C, np.nan),
            "combined_rank": np.full_like(C, np.nan),
        }
        return elim_score, extras

    # Rank
    judge_rank = soft_rank(J, temp=cfg.rank_temp)
    vote_rank = soft_rank(vote_share, temp=cfg.rank_temp)
    combined_rank = 0.5 * judge_rank + 0.5 * vote_rank
    elim_score = combined_rank  # 越大越差

    extras = {
        "rule": rule,
        "C": np.full_like(combined_rank, np.nan),
        "judge_share": np.full_like(combined_rank, np.nan),
        "judge_rank": judge_rank,
        "vote_rank": vote_rank,
        "combined_rank": combined_rank,
    }
    return elim_score, extras

def elim_nll(elim_score: np.ndarray, elim_indices: list, tau: float) -> float:
    """
    概率淘汰损失（负对数似然）：
    将淘汰者视为从 softmax(elim_score/tau) 抽到的结果。
    多淘汰周：对每个淘汰者 NLL 取平均（简单稳健，竞赛足够用）
    """
    if not elim_indices:
        return 0.0

    z = np.asarray(elim_score, dtype=float) / max(tau, 1e-6)
    z = z - np.max(z)
    p = np.exp(z)
    p = p / (np.sum(p) + 1e-12)

    return float(np.mean([-np.log(p[i] + 1e-12) for i in elim_indices]))

def week_objective(x, mu_eff, J, season, elim_indices, cfg: Q1Config):
    """
    每周优化目标：
    1) (x - mu_eff)^2 贴近先验中心（含时间平滑）
    2) lam * x^2       防极端
    3) gamma * NLL     概率淘汰一致性
    """
    vote_share = softmax(x)
    elim_score, _ = compute_elim_score_and_extras(int(season), J, vote_share, cfg)

    base = np.sum((x - mu_eff) ** 2) + cfg.lam * np.sum(x ** 2)
    penalty = cfg.gamma * elim_nll(elim_score, elim_indices, tau=cfg.tau)
    return base + penalty

def solve_one_week(df_week: pd.DataFrame, true_elims: list, cfg: Q1Config, prev_x_map: dict | None):
    """
    对单个 (season, week) 求解 vote_share：
    - mu_eff = (1-smooth_w)*mu + smooth_w*prev_x（若上一周存在）
    - 初值 x0 = mu_eff（warm start）
    """
    season = int(df_week["season"].iloc[0])
    names = df_week["celebrity_name"].astype(str).tolist()
    J = df_week["J"].to_numpy(float)
    mu = df_week["mu"].to_numpy(float)

    # 真实淘汰者索引（可能为空/多个）
    elim_indices = [names.index(e) for e in true_elims if e in names]

    # 时间平滑后的先验中心 mu_eff
    mu_eff = mu.copy()
    if prev_x_map:
        prev_vec = np.array([prev_x_map.get(n, np.nan) for n in names], dtype=float)
        mask = ~np.isnan(prev_vec)
        if mask.any():
            mu_eff[mask] = (1.0 - cfg.smooth_w) * mu[mask] + cfg.smooth_w * prev_vec[mask]

    x0 = mu_eff.copy()
    bounds = [cfg.x_bounds] * len(names)

    res = minimize(
        week_objective,
        x0=x0,
        args=(mu_eff, J, season, elim_indices, cfg),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 600},
    )

    x_hat = res.x
    vote_share = softmax(x_hat)
    elim_score, extras = compute_elim_score_and_extras(season, J, vote_share, cfg)

    # 预测淘汰：elim_score 最大的 k 个（越大越差）
    k = max(1, len(true_elims))
    pred_idx = np.argsort(elim_score)[-k:][::-1]
    pred_elims = [names[i] for i in pred_idx]

    out = df_week[["season", "week", "celebrity_name", "J", "mu"]].copy()
    out["mu_eff"] = mu_eff
    out["vote_share"] = vote_share
    out["votes_scaled"] = vote_share * cfg.scaled_total_votes
    out["rule"] = extras["rule"]
    out["elim_score"] = elim_score

    # Percentage 专属 / Rank 专属字段（另一个规则下为 NaN）
    out["C"] = extras["C"]
    out["judge_share"] = extras["judge_share"]
    out["judge_rank"] = extras["judge_rank"]
    out["vote_rank"] = extras["vote_rank"]
    out["combined_rank"] = extras["combined_rank"]

    # 优化信息
    out["opt_success"] = bool(res.success)
    out["opt_fun"] = float(res.fun)

    # 真实/预测淘汰集合（便于核对）
    out["true_elims"] = [true_elims] * len(out)
    out["pred_elims"] = [pred_elims] * len(out)

    # 返回 x_hat 供下一周 warm-start（不必保存到文件）
    return out, {name: float(x) for name, x in zip(names, x_hat)}

def run_all_weeks(week_tbl_mu: pd.DataFrame, elim_by_week: pd.DataFrame, cfg: Q1Config) -> pd.DataFrame:
    """
    遍历所有赛季所有周，逐周求解，并用 prev_x_map 维持时间连续性
    """
    alive_tbl = week_tbl_mu[week_tbl_mu["alive"] == 1].copy()

    # (season, week) -> true_elims
    elim_map = {(int(r.season), int(r.week)): r.true_elims for r in elim_by_week.itertuples(index=False)}

    outs = []

    for season in sorted(alive_tbl["season"].unique().tolist()):
        gS = alive_tbl[alive_tbl["season"] == season].copy()
        # 只遍历实际播出周（alive==1 出现的周）
        weeks = sorted(gS["week"].unique().tolist())
        prev_map = {}

        for week in weeks:
            gW = gS[gS["week"] == week].copy()
            true_elims = elim_map.get((int(season), int(week)), [])
            out_week, prev_map = solve_one_week(gW, true_elims, cfg, prev_map)
            outs.append(out_week)


    return pd.concat(outs, ignore_index=True)


# ============================================================
# Step 4：一致性指标
# ============================================================
def week_metrics(df_week_out: pd.DataFrame) -> dict:
    """
    指标定义：
    - acc_single：单淘汰周准确率
    - cover_bottomk：多淘汰周 bottom-k 覆盖率
    - jaccard：真实淘汰集合与预测淘汰集合的集合相似度
    - delta_margin：elim_score 第一与第二的差（越小表示越“临界/不确定”）
    """
    true_elims = df_week_out["true_elims"].iloc[0]
    pred_elims = df_week_out["pred_elims"].iloc[0]
    true_set, pred_set = set(true_elims), set(pred_elims)

    cover = float(true_set.issubset(pred_set)) if true_set else np.nan
    jacc = (len(true_set & pred_set) / len(true_set | pred_set)) if true_set else np.nan
    acc = float(list(true_set)[0] == list(pred_set)[0]) if len(true_set) == 1 else np.nan

    score = df_week_out["elim_score"].to_numpy(float)
    order = np.argsort(score)[::-1]  # 大->小（越大越可能淘汰）
    delta = float(score[order[0]] - score[order[1]]) if len(score) >= 2 else np.nan

    return {"acc_single": acc, "cover_bottomk": cover, "jaccard": jacc, "delta_margin": delta}

def evaluate_all(df_votes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (s, w), g in df_votes.groupby(["season", "week"]):
        rows.append({"season": int(s), "week": int(w), **week_metrics(g)})
    return pd.DataFrame(rows).sort_values(["season", "week"]).reset_index(drop=True)


# ============================================================
# Step 5（可选）：bootstrap 不确定性（只建议对少数周做）
# ============================================================
def bootstrap_one_week(df_week: pd.DataFrame, true_elims: list, cfg: Q1Config,
                       B=200, sigma_mu=0.15, sigma_J=0.02, seed=0) -> pd.DataFrame:
    """
    对一个 season-week 做 bootstrap：
    - mu 加噪声：模拟先验不确定
    - J 加相对噪声：模拟评分误差
    每次重解，得到 vote_share 分布
    """
    rng = np.random.default_rng(seed)
    samples = []

    for b in range(B):
        g = df_week.copy()

        g["mu"] = g["mu"] + rng.normal(0.0, sigma_mu, size=len(g))

        J = g["J"].to_numpy(float)
        Jp = J * (1.0 + rng.normal(0.0, sigma_J, size=len(J)))
        g["J"] = np.clip(Jp, 0.0, None)

        out, _ = solve_one_week(g, true_elims, cfg, prev_x_map=None)
        out["b"] = b
        samples.append(out[["season", "week", "celebrity_name", "b", "vote_share", "elim_score", "rule"]])

    return pd.concat(samples, ignore_index=True)

def summarize_bootstrap(bs: pd.DataFrame) -> pd.DataFrame:
    def q(x, p): return float(np.quantile(x, p))
    return (
        bs.groupby(["season", "week", "celebrity_name"])["vote_share"]
          .agg(vote_p05=lambda x: q(x, 0.05),
               vote_p50=lambda x: q(x, 0.50),
               vote_p95=lambda x: q(x, 0.95),
               vote_std="std")
          .reset_index()
    )


# ============================================================
# 主程序：跑通 + 保存输出
# ============================================================
def main():
    ensure_dir(OUT_DIR)

    # Step 0
    long_scores = load_raw(CSV_PATH)

    # Step 0.5
    week_tbl = build_week_table(long_scores)

    # Step 1
    season_last, elim_people, elim_by_week = infer_eliminations(week_tbl)

    # Step 2
    week_tbl_mu, _prior_model = fit_prior_mu(week_tbl)

    # Step 3
    cfg = Q1Config(lam=0.1, gamma=120.0, tau=0.02, smooth_w=0.30, scaled_total_votes=1_000_000)
    df_votes = run_all_weeks(week_tbl_mu, elim_by_week, cfg)

    # Step 4
    metrics = evaluate_all(df_votes)

    # 保存输出
    out_votes = os.path.join(OUT_DIR, "q1_df_votes.csv")
    out_metrics = os.path.join(OUT_DIR, "q1_metrics.csv")
    out_elims = os.path.join(OUT_DIR, "q1_elim_by_week.csv")
    out_people = os.path.join(OUT_DIR, "q1_elim_people.csv")

    save_csv(df_votes, out_votes)
    save_csv(metrics, out_metrics)
    save_csv(elim_by_week, out_elims)
    save_csv(elim_people, out_people)

    if SAVE_EXCEL:
        save_excel(df_votes, os.path.join(OUT_DIR, "q1_df_votes.xlsx"), "df_votes")
        save_excel(metrics, os.path.join(OUT_DIR, "q1_metrics.xlsx"), "metrics")

    print("✅ 已保存：")
    print(" -", out_votes)
    print(" -", out_metrics)
    print(" -", out_elims)
    print(" -", out_people)

    print("\n📌 指标均值（快速自检）：")
    print(metrics[["acc_single", "cover_bottomk", "jaccard", "delta_margin"]].mean(numeric_only=True))

    # Step 5（可选）
    if RUN_BOOTSTRAP:
        df_week = week_tbl_mu[(week_tbl_mu["season"] == BOOTSTRAP_SEASON) &
                              (week_tbl_mu["week"] == BOOTSTRAP_WEEK) &
                              (week_tbl_mu["alive"] == 1)].copy()

        sub = elim_by_week[(elim_by_week["season"] == BOOTSTRAP_SEASON) &
                           (elim_by_week["week"] == BOOTSTRAP_WEEK)]
        true_elims = sub["true_elims"].iloc[0] if len(sub) > 0 else []

        bs = bootstrap_one_week(df_week, true_elims, cfg, B=BOOTSTRAP_B, seed=42)
        bs_summary = summarize_bootstrap(bs)

        out_bs = os.path.join(OUT_DIR, f"q1_bootstrap_summary_s{BOOTSTRAP_SEASON}_w{BOOTSTRAP_WEEK}.csv")
        save_csv(bs_summary, out_bs)
        print("\n✅ Bootstrap 已保存：", out_bs)


if __name__ == "__main__":
    main()