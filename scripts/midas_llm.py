from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from datetime import datetime

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from forecasting_system.tools.events import extract_quarter_events
from forecasting_system.tools.news import preprocess_quarterly_news
from forecasting_system.tools.rules import load_rules, match_rules_for_event
from forecasting_system.tools.financial_model import linear_adjustment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEWS_PATHS = [
    PROJECT_ROOT / "data" / "news" / "news_2021.xlsx",
    PROJECT_ROOT / "data" / "news" / "news_2022.xlsx",
    PROJECT_ROOT / "data" / "news" / "news_2023.xlsx",
    PROJECT_ROOT / "data" / "news" / "news_2024.xlsx"
]
BASE_RULES_PATH = PROJECT_ROOT / "data" / "rules" / "rules.json"
FUNDAMENTAL_OUTPUT_DIR = PROJECT_ROOT / "logs" / "fundamental_analyst_report_test" / "fundamental_analyst_outputs"

MAX_LEN = 31
ALPHA = 1.0

def _component_impacts_for_rule(rule: dict[str, Any]) -> list[dict[str, Any]]:
    component_impacts = rule.get("component_impacts")
    if component_impacts:
        return [
            {"target_component": item["target_component"], "base_impact": float(item["base_impact"])}
            for item in component_impacts
        ]
    return [{"target_component": rule["target_component"], "base_impact": float(rule["params"]["base_impact"])}]

def _cycle_weight_for_event(event: dict[str, Any], rule: dict[str, Any], target_component: str) -> float:
    weights = event.get("cycle_weights") or {}
    candidates = [
        f"{rule['rule_id']}::{target_component}",
        f"{rule['rule_id']}:{target_component}",
        rule["rule_id"],
        f"{event.get('scenario')}::{target_component}",
        f"{event.get('scenario')}:{target_component}",
        str(event.get("scenario")),
        "default",
    ]
    for key in candidates:
        value = weights.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 1.0

def load_report_metrics() -> dict[str, dict[str, Any]]:
    metrics = {}
    for path in FUNDAMENTAL_OUTPUT_DIR.glob("*.json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        period = data.get("period")
        if not period:
            continue
        derived = data.setdefault("derived_state", {})
        if "reported_roa" not in derived:
            profit_margin = float(derived.get("profit_margin", 0.0))
            asset_turnover = float(derived.get("asset_turnover", 0.0))
            derived["reported_roa"] = profit_margin * asset_turnover
        metrics[period] = data
    return metrics

def get_quarter_roa(quarter: str, report_metrics: dict) -> float:
    year = quarter[:4]
    q = quarter[-1]
    if q == '1':
        return float(report_metrics[f"{year}Q1"]["derived_state"]["reported_roa"])
    elif q == '2':
        return float(report_metrics[f"{year}H1"]["derived_state"]["reported_roa"])
    elif q == '3':
        return float(report_metrics[f"{year}Q3"]["derived_state"]["reported_roa"])
    elif q == '4':
        return float(report_metrics[f"{year}FY"]["derived_state"]["reported_roa"])
    else:
        raise ValueError(f"Invalid quarter: {quarter}")

def compute_quarter_delta_series(quarter_news_titles: list[str], rules: list[dict]) -> tuple[list[float], list[float]]:
    if not quarter_news_titles:
        return [], []
    payload = extract_quarter_events(quarter_news_titles)
    events = payload.get("events", [])
    delta_pm_series = []
    delta_at_series = []
    for event in events:
        if event.get("noise", False):
            continue
        matched_rules = match_rules_for_event(event, rules)
        for rule in matched_rules:
            if rule.get("function_name") != "linear_adjustment":
                continue
            impacts = _component_impacts_for_rule(rule)
            for imp in impacts:
                base_impact = imp["base_impact"]
                target = imp["target_component"]
                cycle_weight = _cycle_weight_for_event(event, rule, target)
                effective_strength = float(event.get("strength", 0.0)) * cycle_weight
                delta = linear_adjustment(base_impact, float(event.get("relevance", 0.0)), effective_strength)
                if target == "profit_margin":
                    delta_pm_series.append(delta)
                elif target == "asset_turnover":
                    delta_at_series.append(delta)
    return delta_pm_series, delta_at_series

def build_midas_feature_matrix(quarters_news_list: list[list[str]], rules: list[dict], max_len: int = MAX_LEN) -> np.ndarray:
    X = []
    for news_titles in quarters_news_list:
        pm_series, at_series = compute_quarter_delta_series(news_titles, rules)
        pm_padded = pm_series[:max_len] + [0.0] * (max_len - len(pm_series))
        at_padded = at_series[:max_len] + [0.0] * (max_len - len(at_series))
        X.append(pm_padded + at_padded)
    return np.array(X)

def plot_actual_vs_predicted(
    quarters: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: Path = None,
) -> None:

    plt.figure(figsize=(10, 5))
    plt.plot(quarters, y_true, marker='o', linestyle='-', label='Actual')
    plt.plot(quarters, y_pred, marker='s', linestyle='--', label='Predicted')
    plt.xlabel('Quarter')
    plt.ylabel('ROA Change')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"图已保存至 {save_path}")
    plt.show()


def main():
    print("加载新闻数据...")
    quarterly_news = preprocess_quarterly_news(NEWS_PATHS)
    print(f"新闻数据覆盖的自然季度数: {len(quarterly_news)}")

    print("加载财务报告...")
    report_metrics = load_report_metrics()
    print("加载基础规则...")
    rules = load_rules(BASE_RULES_PATH)

    all_quarters = [f"{year}Q{q}" for year in range(2021, 2025) for q in range(1, 5)]
    valid_quarters = []
    for q in all_quarters:
        try:
            get_quarter_roa(q, report_metrics)
            valid_quarters.append(q)
        except KeyError:
            print(f"跳过 {q}：缺少对应的报告（可能没有H1或FY）")
    print(f"将处理的季度序列: {valid_quarters}")

    quarters_news_list = []
    roa_list = []
    for quarter in valid_quarters:
        year = int(quarter[:4])
        q_num = int(quarter[-1])
        news_titles = quarterly_news.get((year, q_num), [])
        quarters_news_list.append(news_titles)
        roa = get_quarter_roa(quarter, report_metrics)
        roa_list.append(roa)

    print("计算季度 delta 序列特征...")
    X = build_midas_feature_matrix(quarters_news_list, rules, max_len=MAX_LEN)
    print(f"特征矩阵形状: {X.shape}")

    y = []
    for i in range(1, len(roa_list)):
        y.append(roa_list[i] - roa_list[i-1])
    y = np.array(y)

    X = X[1:]
    print(f"样本数: {len(y)}")

    if len(X) < 2:
        print("样本不足，退出")
        return


    train_indices = []
    test_indices = []
    for i, quarter in enumerate(valid_quarters[1:]):
        year = int(quarter[:4])
        if year <= 2023:
            train_indices.append(i)
        else:
            test_indices.append(i)

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")

    if len(X_train) == 0 or len(X_test) == 0:
        print("训练集或测试集为空，退出")
        return

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Ridge(alpha=ALPHA)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\n测试集 MSE : {mse:.6f}")
    print(f"测试集 MAE : {mae:.6f}")

    print("\n预测 vs 真实 ROA 变化:")
    for i, (true, pred) in enumerate(zip(y_test, y_pred)):
        print(f"  样本{i}: 真实={true:.6f}, 预测={pred:.6f}, 误差={true-pred:.6f}")


    y_train_pred = model.predict(X_train_scaled)

    train_quarters = [valid_quarters[i + 1] for i in train_indices]

    plot_actual_vs_predicted(
        quarters=train_quarters,
        y_true=y_train,
        y_pred=y_train_pred,
        title="Training Set (2021-2023) – Actual vs Predicted ROA Change",
        save_path= PROJECT_ROOT / "logs" / "midas" / "midas_training_plot.png"
    )


    test_quarters = [valid_quarters[i + 1] for i in test_indices]
    plot_actual_vs_predicted(
        quarters=test_quarters,
        y_true=y_test,
        y_pred=y_pred,
        title="Test Set (2024) – Actual vs Predicted ROA Change",
        save_path=PROJECT_ROOT / "logs" / "midas" / "midas_test_plot.png"
    )

    params_to_save = {
        "timestamp": datetime.now().isoformat(),
        "model_type": "Ridge",
        "alpha": ALPHA,
        "max_len": MAX_LEN,
        "feature_dim": X.shape[1],
        "coefficients": model.coef_.tolist(),
        "intercept": model.intercept_.item(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_quarters": [valid_quarters[i + 1] for i in train_indices],
        "test_quarters": [valid_quarters[i + 1] for i in test_indices],
        "mse": mse,
        "mae": mae,
    }


    save_dir = PROJECT_ROOT / "logs" / "midas"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "midas_training_parameters.json"

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(params_to_save, f, indent=2, ensure_ascii=False)

    print(f"模型参数已保存至 {save_path}")

if __name__ == "__main__":
    main()