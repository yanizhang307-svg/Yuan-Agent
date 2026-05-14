from __future__ import annotations

import copy
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from forecasting_system.agents.student import Student
from forecasting_system.agents.teacher import Teacher
from forecasting_system.exceptions import SchemaValidationError
from forecasting_system.tools.cycle_state import infer_company_cycle_state
from forecasting_system.tools.events import extract_quarter_events
from forecasting_system.tools.global_reasoning_llm import analyze_global_rule_updates
from forecasting_system.tools.news import preprocess_quarterly_news
from forecasting_system.tools.rules import load_rules, match_rules_for_event, validate_rules


NEWS_PATHS = [
    PROJECT_ROOT / "data" / "news" / "news_2021.xlsx",
    PROJECT_ROOT / "data" / "news" / "news_2022.xlsx",
    PROJECT_ROOT / "data" / "news" / "news_2023.xlsx",
]
BASE_RULES_PATH = PROJECT_ROOT / "data" / "rules" / "rules.json"
FUNDAMENTAL_OUTPUT_DIR = PROJECT_ROOT / "logs" / "fundamental_analyst_report_test" / "fundamental_analyst_outputs"
OUTPUT_DIR = PROJECT_ROOT / "logs" / "quarterly_2021_2023_training"
EDA_CACHE_PATH = PROJECT_ROOT / "logs" / "quarterly_eda_only" / "quarterly_event_classification.json"
TEACHER_TOLERANCE = 0.03
TOTAL_PASSES = 3

TRAINING_PERIODS = [
    (0, "2021_START", "year_start", "2020FY"),
    (1, "2021Q1", "quarter", "2021Q1"),
    (2, "2021H1", "quarter", "2021H1"),
    (3, "2021Q3", "quarter", "2021Q3"),
    (4, "2021FY", "quarter", "2021FY"),
    (5, "2022_START", "year_start", "2021FY"),
    (6, "2022Q1", "quarter", "2022Q1"),
    (7, "2022H1", "quarter", "2022H1"),
    (8, "2022Q3", "quarter", "2022Q3"),
    (9, "2022FY", "quarter", "2022FY"),
    (10, "2023_START", "year_start", "2022FY"),
    (11, "2023Q1", "quarter", "2023Q1"),
    (12, "2023H1", "quarter", "2023H1"),
    (13, "2023Q3", "quarter", "2023Q3"),
    (14, "2023FY", "quarter", "2023FY"),
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    eda_payload = _time_stage("load_or_build_training_eda", _load_or_build_training_eda)
    report_metrics = _time_stage("load_report_metrics", _load_report_metrics)
    active_rules = _time_stage("load_base_rules", load_rules, BASE_RULES_PATH)

    event_strength_overrides: list[dict[str, Any]] = []
    cycle_weight_suggestions: list[dict[str, Any]] = []
    period_cycle_states: list[dict[str, Any]] = []
    prior_passes: list[dict[str, Any]] = []
    pass_summaries: list[dict[str, Any]] = []

    for pass_id in range(1, TOTAL_PASSES + 1):
        pass_dir = OUTPUT_DIR / f"pass_{pass_id}"
        pass_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n===== TRAINING PASS {pass_id}/{TOTAL_PASSES} =====", flush=True)

        records, event_counter = _time_stage(
            f"pass_{pass_id} student_teacher",
            _run_student_teacher,
            eda_payload,
            report_metrics,
            active_rules,
            event_strength_overrides,
            cycle_weight_suggestions,
            period_cycle_states,
        )
        summary = _build_summary(records)
        training_context = _build_training_context(records, active_rules, prior_passes, summary)

        _write_json(pass_dir / "quarterly_records.json", records)
        _write_json(pass_dir / "summary_metrics.json", summary)
        _write_json(pass_dir / "global_reasoning_payload.json", training_context)
        _write_summary_csv(records, pass_dir / "quarterly_summary.csv")
        _write_component_metrics(records, pass_dir / "teacher_component_metrics.csv")
        _write_event_frequency(event_counter, pass_dir / "event_frequency.csv")
        _write_agent_attribution(records, pass_dir / "agent_attribution.csv")

        reasoning_response = _time_stage(
            f"pass_{pass_id} global_reasoning",
            analyze_global_rule_updates,
            pass_id=pass_id,
            total_passes=TOTAL_PASSES,
            training_context=training_context,
            active_rules=active_rules,
        )
        active_rules, rule_update_records = _time_stage(
            f"pass_{pass_id} apply_rule_updates",
            _apply_global_updates,
            active_rules,
            reasoning_response,
        )
        validate_rules(active_rules)

        event_strength_overrides.extend(reasoning_response.get("event_strength_overrides", []))
        cycle_weight_suggestions.extend(reasoning_response.get("cycle_weight_suggestions", []))
        period_cycle_states = _merge_period_cycle_states(
            period_cycle_states,
            reasoning_response.get("period_cycle_states", []),
        )

        _write_json(pass_dir / "global_reasoning_response.json", reasoning_response)
        _write_json(pass_dir / "active_rules.json", active_rules)
        _write_json(pass_dir / "event_strength_overrides_active.json", event_strength_overrides)
        _write_json(pass_dir / "cycle_weight_suggestions_active.json", cycle_weight_suggestions)
        _write_json(pass_dir / "period_cycle_states_active.json", period_cycle_states)
        _write_rule_log(rule_update_records, pass_dir / "rule_modification_log.txt")
        _write_global_report(reasoning_response, rule_update_records, pass_dir / "global_reasoning_report.md")

        pass_summary = {
            "pass_id": pass_id,
            **summary,
            "rule_update_count": len(rule_update_records),
        }
        pass_summaries.append(pass_summary)
        prior_passes.append(
            {
                "pass_id": pass_id,
                "summary_metrics": summary,
                "rule_update_count": len(rule_update_records),
                "analysis_summary": reasoning_response["analysis_summary"],
            }
        )
        _write_json(OUTPUT_DIR / "active_rules_final.json", active_rules)
        _write_json(OUTPUT_DIR / "pass_summaries.json", pass_summaries)
        _write_pass_summary_csv(pass_summaries, OUTPUT_DIR / "pass_summaries.csv")
        print(f"PASS {pass_id} complete: updates={len(rule_update_records)}", flush=True)

    print(f"\nTraining complete: {OUTPUT_DIR}", flush=True)
    print(f"Final trained rules: {OUTPUT_DIR / 'active_rules_final.json'}", flush=True)


def _run_student_teacher(
    eda_payload: dict[str, Any],
    report_metrics: dict[str, dict[str, Any]],
    rules: list[dict[str, Any]],
    event_strength_overrides: list[dict[str, Any]],
    cycle_weight_suggestions: list[dict[str, Any]],
    period_cycle_states: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    student = Student()
    teacher = Teacher()
    records: list[dict[str, Any]] = []
    event_counter: Counter[str] = Counter()
    scheduled_events_by_t: dict[int, list[dict[str, Any]]] = defaultdict(list)
    current_state: dict[str, float] | None = None
    current_roa = 0.0
    inferred_cycle_states = _initial_cycle_state_map(report_metrics)
    inferred_cycle_states.update({item["period"]: item for item in period_cycle_states if "period" in item})

    for t, period, kind, report_period in TRAINING_PERIODS:
        report = report_metrics[report_period]
        if kind == "year_start":
            current_state = _year_start_state(report)
            current_roa = 0.0
            records.append(
                {
                    "t": t,
                    "period": period,
                    "kind": kind,
                    "report_period_used": report_period,
                    "events": [],
                    "baseline_state": copy.deepcopy(current_state),
                    "predicted_state": copy.deepcopy(current_state),
                    "actual_state": None,
                    "predicted_roa": 0.0,
                    "actual_roa": 0.0,
                    "student_record": None,
                    "teacher_feedback": None,
                }
            )
            continue

        if current_state is None:
            raise SchemaValidationError("Quarter reached before a year-start state.")

        quarter_payload = eda_payload[period]
        enriched_events = _enrich_events_with_news_names(quarter_payload)
        enriched_events = _apply_strength_overrides(period, enriched_events, event_strength_overrides)
        _schedule_eda_timed_events(
            source_t=t,
            source_period=period,
            events=enriched_events,
            rules=rules,
            cycle_weight_suggestions=cycle_weight_suggestions,
            scheduled_events_by_t=scheduled_events_by_t,
        )
        active_events = scheduled_events_by_t.get(t, [])
        non_noise_events = [event for event in active_events if not event.get("noise", False)]
        for event in non_noise_events:
            event_counter[f"{event['scenario']}::{event['event_type']}::{event['direction']}"] += 1

        student_record = _time_stage(f"{period} student", student.run, non_noise_events, current_state, rules)
        predicted_state = _predicted_state_from_student_record(student_record)
        actual_state = copy.deepcopy(report["derived_state"])
        actual_roa = float(actual_state["reported_roa"])
        predicted_roa = float(student_record["final_prediction"]["roa"])
        cycle_state = inferred_cycle_states.get(period)
        student_record["quarterly_training_context"] = {
            "period": period,
            "t": t,
            "supervision_frequency": "quarterly",
            "actual_roa": actual_roa,
            "fundamental_analyst_report": report.get("fundamental_analyst_report"),
            "derived_state": actual_state,
            "cycle_state": cycle_state,
        }
        teacher_feedback = _time_stage(
            f"{period} teacher",
            teacher.run,
            student_record,
            {
                "actual_roa": actual_roa,
                "actual_state": actual_state,
                "baseline_roa": current_roa,
                "tolerance": TEACHER_TOLERANCE,
                "component_tolerance": TEACHER_TOLERANCE,
                "frequency": "quarterly",
                "period": period,
                "supervision_source": "reported_actual_roa",
            },
        )
        records.append(
            {
                "t": t,
                "period": period,
                "kind": kind,
                "report_period_used": report_period,
                "raw_news_count": len(quarter_payload["raw_news"]),
                "deduped_news_count": len(quarter_payload["deduped_news"]),
                "non_noise_event_count": len(non_noise_events),
                "events": active_events,
                "source_period_events": enriched_events,
                "cycle_state": cycle_state,
                "baseline_state": copy.deepcopy(current_state),
                "predicted_state": predicted_state,
                "actual_state": actual_state,
                "predicted_roa": predicted_roa,
                "actual_roa": actual_roa,
                "predicted_delta": predicted_roa - current_roa,
                "supervision_delta": actual_roa - current_roa,
                "student_record": student_record,
                "teacher_feedback": teacher_feedback,
            }
        )

        current_state = {
            "profit_margin": float(actual_state["profit_margin"]),
            "asset_turnover": float(actual_state["asset_turnover"]),
            "equity_multiplier": float(actual_state.get("equity_multiplier", current_state["equity_multiplier"])),
        }
        current_roa = actual_roa
        next_period = _next_quarter_period(period)
        if next_period is not None:
            inferred_cycle_states[next_period] = infer_company_cycle_state(
                source_period=period,
                target_period=next_period,
                report_payload=report,
            )

    return records, event_counter


def _load_or_build_training_eda() -> dict[str, Any]:
    cached_payload = _load_json(EDA_CACHE_PATH, default={}) if EDA_CACHE_PATH.exists() else {}
    quarterly_news = preprocess_quarterly_news(NEWS_PATHS)
    for year in (2021, 2022, 2023):
        for quarter, suffix in ((1, "Q1"), (2, "H1"), (3, "Q3"), (4, "FY")):
            period = f"{year}{suffix}"
            if period in cached_payload:
                print(f"SKIP {period}: loaded cached EDA output", flush=True)
                continue
            raw_titles = quarterly_news.get((year, quarter), [])
            print(f"START {period}: raw_news={len(raw_titles)}", flush=True)
            batch_payload = extract_quarter_events(raw_titles) if raw_titles else {
                "deduped_news": [],
                "duplicate_groups": [],
                "events": [],
            }
            cached_payload[period] = {
                "period": period,
                "year": year,
                "quarter": quarter,
                "raw_news": raw_titles,
                "deduped_news": batch_payload["deduped_news"],
                "duplicate_groups": batch_payload["duplicate_groups"],
                "events": batch_payload["events"],
            }
            _write_json(EDA_CACHE_PATH, cached_payload)
    return cached_payload


def _load_report_metrics() -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for path in FUNDAMENTAL_OUTPUT_DIR.glob("*.json"):
        payload = _load_json(path, default={})
        _ensure_reported_roa(payload)
        metrics[payload["period"]] = payload
    required = sorted({report_period for _, _, _, report_period in TRAINING_PERIODS})
    missing = [period for period in required if period not in metrics]
    if missing:
        raise SystemExit(
            "Missing Fundamental Analyst JSON outputs: "
            f"{missing}. Run: python scripts\\run_fundamental_analyst_reports.py"
        )
    return metrics


def _apply_global_updates(
    active_rules: list[dict[str, Any]],
    reasoning_response: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    updated_rules = copy.deepcopy(active_rules)
    rules_by_id = {rule["rule_id"]: rule for rule in updated_rules}
    update_records: list[dict[str, Any]] = []

    for suggestion in reasoning_response.get("rule_update_suggestions", []):
        rule_id = suggestion["rule_id"]
        if rule_id not in rules_by_id:
            raise SchemaValidationError(f"Global reasoning suggested unknown rule_id: {rule_id}")
        rule = rules_by_id[rule_id]
        field = suggestion["field"]
        previous_value = _read_rule_field(rule, field)
        print(f"导致错误的 suggestion 内容: {suggestion}")
        new_value = _normalize_update_value(rule, field, suggestion["new_value"])
        if previous_value == new_value:
            continue
        _write_rule_field(rule, field, new_value)
        update_records.append(
            {
                "rule_id": rule_id,
                "field": field,
                "previous_value": previous_value,
                "new_value": new_value,
                "reason": suggestion["reason"],
            }
        )
    return updated_rules, update_records


def _normalize_update_value(rule: dict[str, Any], field: str, new_value: Any) -> Any:
    if field == "target_component":
        if new_value not in {"profit_margin", "asset_turnover"}:
            raise SchemaValidationError("target_component must be profit_margin or asset_turnover.")
        return new_value
    if field == "params.base_impact":
        if not isinstance(new_value, (int, float)):
            raise SchemaValidationError("params.base_impact must be numeric.")
        current = float(rule["params"]["base_impact"])
        candidate = float(new_value)
        if candidate == 0 or (current > 0 and candidate < 0) or (current < 0 and candidate > 0):
            raise SchemaValidationError("params.base_impact update must preserve non-zero sign.")
        lower = current * 0.70 if current > 0 else current * 1.30
        upper = current * 1.30 if current > 0 else current * 0.70
        if not min(lower, upper) <= candidate <= max(lower, upper):
            raise SchemaValidationError(f"params.base_impact update exceeds 30% bound: {current} -> {candidate}")
        return candidate
    if field in {"lag", "duration"}:
        if not isinstance(new_value, int) or new_value < 0:
            raise SchemaValidationError(f"{field} must be a non-negative integer.")
        return int(new_value)
    if field in {"explanation", "business_chain", "fundamental_basis"}:
        if not isinstance(new_value, str) or not new_value.strip():
            raise SchemaValidationError(f"{field} must be a non-empty string.")
        return new_value.strip()
    if field == "operating_metric_links":
        if not isinstance(new_value, list) or not all(isinstance(item, str) and item.strip() for item in new_value):
            raise SchemaValidationError("operating_metric_links must be a non-empty string list.")
        return [item.strip() for item in new_value]
    if field == "component_impacts":
        if isinstance(new_value, str):
            import json
            try:
                new_value = json.loads(new_value)
            except json.JSONDecodeError:
                raise SchemaValidationError("component_impacts must be a valid JSON list string.")
        if not isinstance(new_value, list) or not new_value:
            raise SchemaValidationError("component_impacts must be a non-empty list.")
        normalized = []
        for item in new_value:
            if not isinstance(item, dict):
                raise SchemaValidationError("component_impacts items must be objects.")
            component = item.get("target_component")
            impact = item.get("base_impact")
            if component not in {"profit_margin", "asset_turnover"}:
                raise SchemaValidationError("component_impacts target_component is invalid.")
            if not isinstance(impact, (int, float)) or float(impact) == 0:
                raise SchemaValidationError("component_impacts base_impact must be non-zero numeric.")
            normalized.append({"target_component": component, "base_impact": float(impact)})
        return normalized
    raise SchemaValidationError(f"Unsupported update field: {field}")


def _read_rule_field(rule: dict[str, Any], field: str) -> Any:
    if field == "params.base_impact":
        return rule["params"]["base_impact"]
    return rule.get(field)


def _write_rule_field(rule: dict[str, Any], field: str, value: Any) -> None:
    if field == "params.base_impact":
        rule["params"]["base_impact"] = value
    else:
        rule[field] = value


def _build_training_context(
    records: list[dict[str, Any]],
    active_rules: list[dict[str, Any]],
    prior_passes: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    quarter_records = [record for record in records if record["kind"] == "quarter"]
    return {
        "target_metric": "roa",
        "allowed_rule_targets": ["profit_margin", "asset_turnover"],
        "summary_metrics": summary,
        "prior_passes": prior_passes,
        "quarter_error_table": [_quarter_error_row(record) for record in quarter_records],
        "rule_attribution_table": _build_rule_attribution_table(quarter_records),
        "teacher_primary_error_driver_counts": _primary_error_driver_counts(quarter_records),
        "active_rule_count": len(active_rules),
    }


def _quarter_error_row(record: dict[str, Any]) -> dict[str, Any]:
    feedback = record["teacher_feedback"]
    metrics = feedback["evaluation_metrics"]
    return {
        "period": record["period"],
        "baseline_state": _operating_state(record["baseline_state"]),
        "student_predicted_state": _operating_state(record["predicted_state"]),
        "actual_state": _operating_state(record["actual_state"]),
        "predicted_roa": round(float(record["predicted_roa"]), 6),
        "actual_roa": round(float(record["actual_roa"]), 6),
        "level_error": _safe_float(metrics.get("roa", {}).get("level_error")),
        "squared_error": _safe_float(metrics.get("roa", {}).get("squared_error")),
        "teacher_label": feedback["evaluation_label"],
        "error_type": feedback["error_type"],
        "primary_error_driver": feedback.get("component_feedback", {}).get("primary_error_driver"),
        "cycle_state": record.get("cycle_state"),
        "fundamental_summary": (record["student_record"].get("quarterly_training_context") or {}).get(
            "fundamental_analyst_report"
        ),
    }


def _build_rule_attribution_table(quarter_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in quarter_records:
        feedback = record["teacher_feedback"]
        for match in record["student_record"].get("matched_rules", []):
            event = match.get("inputs", {}).get("event", {})
            rows.append(
                {
                    "period": record["period"],
                    "rule_id": match.get("rule_id"),
                    "target_component": match.get("target_component"),
                    "delta": _safe_float(match.get("delta")),
                    "event_key": event.get("event_key")
                    or f"{event.get('scenario')}::{event.get('event_type')}::{event.get('direction')}",
                    "canonical_news": event.get("canonical_news"),
                    "raw_news": event.get("raw_news"),
                    "duplicate_news": event.get("duplicate_news", []),
                    "teacher_label": feedback["evaluation_label"],
                    "primary_error_driver": feedback.get("component_feedback", {}).get("primary_error_driver"),
                }
            )
    return rows


def _build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    quarters = [record for record in records if record["kind"] == "quarter"]
    labels = [record["teacher_feedback"]["evaluation_label"] for record in quarters]
    abs_errors = [abs(float(record["predicted_roa"]) - float(record["actual_roa"])) for record in quarters]
    return {
        "quarter_count": len(quarters),
        "year_start_count": sum(1 for record in records if record["kind"] == "year_start"),
        "reasonable_count": labels.count("reasonable"),
        "too_optimistic_count": labels.count("too_optimistic"),
        "too_pessimistic_count": labels.count("too_pessimistic"),
        "mean_abs_error": sum(abs_errors) / len(abs_errors) if abs_errors else 0.0,
        "mse": _aggregate_mse(quarters),
        "final_predicted_roa": float(quarters[-1]["predicted_roa"]) if quarters else 0.0,
        "final_actual_roa": float(quarters[-1]["actual_roa"]) if quarters else 0.0,
        "uses_reasoning": True,
        "supervision_source": "reported_actual_roa_only",
    }


def _aggregate_mse(quarters: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    buckets = {metric: [] for metric in ("roa", "profit_margin", "asset_turnover")}
    for record in quarters:
        metrics = record["teacher_feedback"]["evaluation_metrics"]
        roa_se = metrics.get("roa", {}).get("squared_error")
        if roa_se is not None:
            buckets["roa"].append(float(roa_se))
        for component, values in metrics.get("components", {}).items():
            if component in buckets:
                buckets[component].append(float(values["squared_error"]))
    return {metric: {"mse": sum(values) / len(values) if values else 0.0} for metric, values in buckets.items()}


def _primary_error_driver_counts(quarter_records: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in quarter_records:
        driver = record["teacher_feedback"].get("component_feedback", {}).get("primary_error_driver")
        if driver:
            counter[str(driver)] += 1
    return dict(counter)


def _schedule_eda_timed_events(
    *,
    source_t: int,
    source_period: str,
    events: list[dict[str, Any]],
    rules: list[dict[str, Any]],
    cycle_weight_suggestions: list[dict[str, Any]],
    scheduled_events_by_t: dict[int, list[dict[str, Any]]],
) -> None:
    max_t = max(t for t, _, kind, _ in TRAINING_PERIODS if kind == "quarter")
    for event in events:
        if event.get("noise", False):
            continue
        for rule in match_rules_for_event(event, rules):
            lag = _event_lag(event)
            duration = _event_duration(event)
            for offset in range(duration):
                active_t = source_t + lag + offset
                if active_t > max_t:
                    continue
                scheduled = copy.deepcopy(event)
                scheduled["active_rule_ids"] = [rule["rule_id"]]
                scheduled["cycle_weights"] = _cycle_weights_for_event(source_period, scheduled, rule, cycle_weight_suggestions)
                scheduled["timing"] = {
                    "source_t": source_t,
                    "source_period": source_period,
                    "active_t": active_t,
                    "rule_id": rule["rule_id"],
                    "event_lag": lag,
                    "event_duration": duration,
                    "duration_offset": offset,
                }
                scheduled_events_by_t[active_t].append(scheduled)


def _cycle_weights_for_event(
    period: str,
    event: dict[str, Any],
    rule: dict[str, Any],
    suggestions: list[dict[str, Any]],
) -> dict[str, float]:
    weights: dict[str, float] = {}
    event_key = event.get("event_key") or f"{event.get('scenario')}::{event.get('event_type')}::{event.get('direction')}"
    for item in suggestions:
        if item.get("period") not in {period, "*", "all"}:
            continue
        scope = item.get("scope")
        key = item.get("key")
        weight = item.get("weight")
        if not isinstance(weight, (int, float)):
            continue
        if scope == "rule" and key == rule["rule_id"]:
            weights[rule["rule_id"]] = float(weight)
        elif scope == "rule_component":
            weights[str(key)] = float(weight)
        elif scope == "scenario" and key in {event.get("scenario"), event_key}:
            weights[str(event.get("scenario"))] = float(weight)
        elif scope == "default":
            weights["default"] = float(weight)
    return weights


def _apply_strength_overrides(
    period: str,
    events: list[dict[str, Any]],
    overrides: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key = {
        (item.get("period"), item.get("event_key"), item.get("canonical_news")): item
        for item in overrides
        if isinstance(item.get("new_strength"), (int, float))
    }
    updated = []
    for event in events:
        cloned = copy.deepcopy(event)
        event_key = cloned.get("event_key") or f"{cloned.get('scenario')}::{cloned.get('event_type')}::{cloned.get('direction')}"
        candidates = [
            (period, event_key, cloned.get("canonical_news")),
            (period, event_key, None),
            ("all", event_key, cloned.get("canonical_news")),
            ("all", event_key, None),
        ]
        for candidate in candidates:
            override = by_key.get(candidate)
            if override:
                cloned["strength"] = float(override["new_strength"])
                cloned["strength_override_reason"] = override.get("reason")
                break
        updated.append(cloned)
    return updated


def _initial_cycle_state_map(report_metrics: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        "2021Q1": infer_company_cycle_state(
            source_period="2020FY",
            target_period="2021Q1",
            report_payload=report_metrics["2020FY"],
        )
    }


def _next_quarter_period(period: str) -> str | None:
    order = [
        "2021Q1",
        "2021H1",
        "2021Q3",
        "2021FY",
        "2022Q1",
        "2022H1",
        "2022Q3",
        "2022FY",
        "2023Q1",
        "2023H1",
        "2023Q3",
        "2023FY",
    ]
    if period not in order:
        return None
    index = order.index(period)
    if index + 1 >= len(order):
        return None
    return order[index + 1]


def _enrich_events_with_news_names(quarter_payload: dict[str, Any]) -> list[dict[str, Any]]:
    deduped_news = quarter_payload.get("deduped_news", [])
    duplicate_groups = quarter_payload.get("duplicate_groups", [])
    events = []
    for index, event in enumerate(quarter_payload.get("events", [])):
        enriched = copy.deepcopy(event)
        canonical = deduped_news[index] if index < len(deduped_news) else None
        if canonical:
            enriched.setdefault("canonical_news", str(canonical))
            enriched.setdefault("raw_news", str(canonical))
            enriched.setdefault("duplicate_news", _duplicates_for_canonical(str(canonical), duplicate_groups))
        enriched.setdefault("event_key", f"{enriched.get('scenario')}::{enriched.get('event_type')}::{enriched.get('direction')}")
        events.append(enriched)
    return events


def _duplicates_for_canonical(canonical: str, duplicate_groups: list[dict[str, Any]]) -> list[str]:
    for group in duplicate_groups:
        if group.get("canonical") == canonical:
            return [str(item) for item in group.get("duplicates", [])]
    return []


def _year_start_state(previous_fy_report: dict[str, Any]) -> dict[str, float]:
    previous_state = previous_fy_report["derived_state"]
    return {
        "profit_margin": 0.0,
        "asset_turnover": 0.0,
        "equity_multiplier": float(previous_state.get("equity_multiplier", 1.0)),
    }


def _predicted_state_from_student_record(student_record: dict[str, Any]) -> dict[str, float]:
    intermediate = student_record["intermediate_values"]
    return {
        "profit_margin": float(intermediate["updated_profit_margin"]),
        "asset_turnover": float(intermediate["updated_asset_turnover"]),
        "equity_multiplier": float(intermediate["updated_equity_multiplier"]),
        "reported_roa": float(student_record["final_prediction"]["roa"]),
    }


def _event_lag(event: dict[str, Any]) -> int:
    value = event.get("lag", 0)
    return min(int(value), 4) if isinstance(value, int) and value >= 0 else 0


def _event_duration(event: dict[str, Any]) -> int:
    value = event.get("duration", 1)
    if not isinstance(value, int) or value <= 0:
        return 1
    return min(value, 4)


def _merge_period_cycle_states(
    existing: list[dict[str, Any]],
    updates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_period = {item.get("period"): item for item in existing if item.get("period")}
    for item in updates:
        if item.get("period"):
            by_period[item["period"]] = item
    return list(by_period.values())


def _ensure_reported_roa(report: dict[str, Any]) -> None:
    derived = report.setdefault("derived_state", {})
    if "reported_roa" in derived:
        return
    profit_margin = float(derived.get("profit_margin", 0.0))
    asset_turnover = float(derived.get("asset_turnover", 0.0))
    derived["reported_roa"] = profit_margin * asset_turnover
    report.setdefault("key_indicators", {})["reported_roa"] = round(float(derived["reported_roa"]), 6)


def _operating_state(state: dict[str, Any] | None) -> dict[str, float] | None:
    if not isinstance(state, dict):
        return None
    return {
        "profit_margin": _safe_float(state.get("profit_margin")),
        "asset_turnover": _safe_float(state.get("asset_turnover")),
        "reported_roa": _safe_float(state.get("reported_roa")),
    }


def _safe_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _time_stage(label: str, fn, *args, **kwargs):
    started_at = time.perf_counter()
    print(f"[STAGE START] {label}", flush=True)
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - started_at
    print(f"[STAGE DONE] {label} | elapsed={elapsed:.2f}s", flush=True)
    return result


def _write_summary_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    rows = []
    for record in records:
        feedback = record.get("teacher_feedback") or {}
        comparison = feedback.get("comparison_summary", {})
        rows.append(
            {
                "t": record["t"],
                "period": record["period"],
                "kind": record["kind"],
                "report_period_used": record["report_period_used"],
                "raw_news_count": record.get("raw_news_count", 0),
                "deduped_news_count": record.get("deduped_news_count", 0),
                "non_noise_event_count": record.get("non_noise_event_count", 0),
                "predicted_roa": record["predicted_roa"],
                "actual_roa": record["actual_roa"],
                "predicted_delta": comparison.get("predicted_delta"),
                "supervision_delta": comparison.get("supervision_delta"),
                "teacher_label": feedback.get("evaluation_label"),
                "error_type": feedback.get("error_type"),
                "primary_error_driver": feedback.get("component_feedback", {}).get("primary_error_driver"),
            }
        )
    _write_csv(output_path, rows)


def _write_component_metrics(records: list[dict[str, Any]], output_path: Path) -> None:
    rows = []
    for record in records:
        if record["kind"] != "quarter":
            continue
        feedback = record["teacher_feedback"]
        comparison = feedback["comparison_summary"]
        roa_metrics = feedback["evaluation_metrics"]["roa"]
        rows.append(_metric_row(record, "roa", comparison, roa_metrics, feedback["evaluation_label"], feedback))
        for metric, values in feedback.get("component_feedback", {}).get("components", {}).items():
            rows.append(
                {
                    "t": record["t"],
                    "period": record["period"],
                    "metric": metric,
                    "baseline": values["baseline"],
                    "predicted": values["predicted"],
                    "actual": values["actual"],
                    "predicted_delta": values["predicted_delta"],
                    "actual_delta": values["actual_delta"],
                    "level_error": values["level_error"],
                    "delta_error": values["delta_error"],
                    "squared_error": values["squared_error"],
                    "teacher_label": values["evaluation_label"],
                    "primary_error_driver": feedback["component_feedback"].get("primary_error_driver"),
                }
            )
    _write_csv(output_path, rows)


def _metric_row(
    record: dict[str, Any],
    metric: str,
    comparison: dict[str, Any],
    values: dict[str, Any],
    label: str,
    feedback: dict[str, Any],
) -> dict[str, Any]:
    return {
        "t": record["t"],
        "period": record["period"],
        "metric": metric,
        "baseline": comparison["baseline_roa"],
        "predicted": comparison["predicted_roa"],
        "actual": comparison["actual_roa"],
        "predicted_delta": comparison["predicted_delta"],
        "actual_delta": comparison["supervision_delta"],
        "level_error": values.get("level_error"),
        "delta_error": values.get("delta_error"),
        "squared_error": values.get("squared_error"),
        "teacher_label": label,
        "primary_error_driver": feedback["component_feedback"].get("primary_error_driver"),
    }


def _write_agent_attribution(records: list[dict[str, Any]], output_path: Path) -> None:
    _write_csv(output_path, _build_rule_attribution_table([record for record in records if record["kind"] == "quarter"]))


def _write_event_frequency(counter: Counter[str], output_path: Path) -> None:
    rows = []
    for key, count in counter.most_common():
        scenario, event_type, direction = key.split("::")
        rows.append({"scenario": scenario, "event_type": event_type, "direction": direction, "count": count})
    _write_csv(output_path, rows)


def _write_rule_log(records: list[dict[str, Any]], output_path: Path) -> None:
    if not records:
        output_path.write_text("No rule updates.\n", encoding="utf-8")
        return
    lines = [
        f"{item['rule_id']} | {item['field']} | {item['previous_value']} -> {item['new_value']} | {item['reason']}"
        for item in records
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_global_report(
    reasoning_response: dict[str, Any],
    rule_update_records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    lines = [
        "# Global Reasoning Report",
        "",
        "## Analysis Summary",
        reasoning_response.get("analysis_summary", ""),
        "",
        "## Global Error Diagnosis",
        reasoning_response.get("global_error_diagnosis", ""),
        "",
        "## Rule Updates",
    ]
    if rule_update_records:
        for item in rule_update_records:
            lines.append(
                f"- `{item['rule_id']}` `{item['field']}`: `{item['previous_value']}` -> `{item['new_value']}`. {item['reason']}"
            )
    else:
        lines.append("- No rule updates.")
    lines.extend(["", "## Experimental Design Recommendations"])
    recommendations = reasoning_response.get("experimental_design_recommendations", [])
    if recommendations:
        lines.extend(f"- {item}" for item in recommendations)
    else:
        lines.append("- None.")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_pass_summary_csv(pass_summaries: list[dict[str, Any]], output_path: Path) -> None:
    rows = []
    for item in pass_summaries:
        mse = item.get("mse", {})
        rows.append(
            {
                "pass_id": item["pass_id"],
                "quarter_count": item["quarter_count"],
                "reasonable_count": item["reasonable_count"],
                "too_optimistic_count": item["too_optimistic_count"],
                "too_pessimistic_count": item["too_pessimistic_count"],
                "mean_abs_error": item["mean_abs_error"],
                "roa_mse": mse.get("roa", {}).get("mse"),
                "profit_margin_mse": mse.get("profit_margin", {}).get("mse"),
                "asset_turnover_mse": mse.get("asset_turnover", {}).get("mse"),
                "rule_update_count": item["rule_update_count"],
            }
        )
    _write_csv(output_path, rows)


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
