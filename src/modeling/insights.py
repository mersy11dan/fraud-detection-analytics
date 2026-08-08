"""Executive fraud detection insights from model outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import MODELING_OUTPUTS_DIR, MODELING_REPORTS_DIR, REPORTS_DIR
from src.modeling.interpretation import describe_feature
from src.utils.paths import ensure_dir

INSIGHTS_FILENAME = "fraud_insights.md"


def _resolve_modeling_dir() -> Path:
    for candidate in (MODELING_OUTPUTS_DIR, MODELING_REPORTS_DIR, REPORTS_DIR / "modeling"):
        if (candidate / "model_comparison_metrics.csv").exists():
            return candidate
    return MODELING_OUTPUTS_DIR


def _load_modeling_tables(modeling_dir: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    metrics = pd.read_csv(modeling_dir / "model_comparison_metrics.csv")
    best = pd.read_csv(modeling_dir / "best_model_summary.csv").iloc[0]
    features = pd.read_csv(modeling_dir / "top_features_best_model.csv")
    return metrics, best, features


def _feature_category(feature: str) -> str:
    if feature in {"time_since_signup_hours", "hour_of_day", "day_of_week", "age"}:
        return "behavioral_timing"
    if feature.startswith("country_"):
        return "geography"
    if feature.startswith("source_"):
        return "acquisition_channel"
    if feature.startswith("browser_"):
        return "device_browser"
    if feature == "purchase_value":
        return "transaction_value"
    return "other"


def build_fraud_insights_markdown(
    *,
    modeling_dir: Path | None = None,
) -> str:
    """Build executive fraud insights from saved modeling outputs."""
    report_dir = modeling_dir or _resolve_modeling_dir()
    metrics, best, features = _load_modeling_tables(report_dir)

    best_name = best["model_name"]
    pr_auc = float(best.get("pr_auc", best.get("auc_pr", 0)))
    precision = float(best["precision"])
    recall = float(best["recall"])
    f1 = float(best["f1"])
    tp = int(best["true_positives"])
    fp = int(best["false_positives"])
    fn = int(best["false_negatives"])
    tn = int(best["true_negatives"])

    top10 = features.head(10)
    top_feature = top10.iloc[0]["feature"]
    top_importance = float(top10.iloc[0]["importance"])

    timing_features = top10[top10["feature"].apply(_feature_category) == "behavioral_timing"]
    geo_features = top10[top10["feature"].str.startswith("country_")]
    channel_features = top10[top10["feature"].str.startswith("source_")]
    browser_features = top10[top10["feature"].str.startswith("browser_")]

    lines = [
        "# Executive Fraud Detection Insights — Fraud_Data",
        "",
        "## Executive summary",
        "",
        (
            f"The tuned **{best_name}** model is the current best performer on the e-commerce fraud "
            f"dataset (PR-AUC **{pr_auc:.3f}**). It flags fraud with **{precision:.1%} precision** "
            f"and **{recall:.1%} recall** on the holdout set — meaning most alerts are real fraud, "
            f"but roughly half of fraudulent transactions still evade detection at the default threshold."
        ),
        "",
        (
            f"On **{tp + fn + fp + tn:,}** holdout transactions, the model caught **{tp:,}** fraud cases, "
            f"missed **{fn:,}**, and generated only **{fp:,}** false alarms. The dominant risk signal is "
            f"**{describe_feature(top_feature)}**, which accounts for the majority of the model's "
            f"feature-importance weight."
        ),
        "",
        "---",
        "",
        "## Top fraud indicators",
        "",
        "Ranked by the best model's feature importance (Random Forest):",
        "",
        "| Rank | Feature | Importance | What it signals |",
        "| --- | --- | --- | --- |",
    ]

    for idx, row in top10.iterrows():
        lines.append(
            f"| {idx + 1} | `{row['feature']}` | {row['importance']:.4f} | {describe_feature(row['feature'])} |"
        )

    lines.extend(
        [
            "",
            "**Key takeaway:** `time_since_signup_hours` dominates (~"
            f"{top_importance:.0%} of importance), consistent with EDA showing fraudsters purchase almost "
            "immediately after signup while legitimate customers wait weeks on average.",
            "",
            "---",
            "",
            "## High-risk customer behaviors",
            "",
            "Patterns the model and EDA associate with elevated fraud risk:",
            "",
            "1. **Immediate post-signup purchases** — Median signup-to-purchase time is near **0 hours** "
            "for fraud vs **~1,443 hours** for legitimate users. First-purchase velocity is the strongest "
            "behavioral red flag.",
            "2. **New-account activity without relationship history** — Fraud cases cluster around accounts "
            "that transact before normal trust patterns develop.",
            "3. **Channel-specific behavior** — Direct traffic showed the highest fraud rate (~10.5%) in EDA; "
            f"{'`source_Direct` appears in model drivers' if 'source_Direct' in top10['feature'].values else 'channel features contribute secondary signal'}.",
            "4. **Geographic context** — Country labels (e.g. United States, China, unknown IP mapping) add "
            f"secondary risk context; **{len(geo_features)}** geographic features rank in the top 10.",
            "",
            "---",
            "",
            "## High-risk transaction characteristics",
            "",
            "Transaction-level patterns linked to higher model scores:",
            "",
        ]
    )

    if not timing_features.empty:
        lines.append(
            "- **Timing:** Purchases at certain hours/days of week contribute measurable risk "
            f"(`hour_of_day`, `day_of_week` in top drivers)."
        )
    lines.extend(
        [
            "- **Value:** `purchase_value` ranks in top features but EDA shows similar medians across classes — "
            "amount alone is a weak rule, but adds context in combination with other signals.",
            "- **Device/browser:** Browser one-hot features (Chrome, IE, Safari, etc.) appear in the top 20, "
            "suggesting device fingerprint patterns differ between fraud and legitimate traffic.",
            "- **Geolocation:** Transactions from specific countries or with unmapped IPs (`country_unknown`) "
            "can elevate scores — treat as review signals, not automatic blocks.",
            "",
            "---",
            "",
            "## Recommendations",
            "",
            "### Fraud prevention",
            "",
            "1. **Cooldown on new accounts** — Apply stepped purchase limits or extra verification when "
            "`time_since_signup_hours` is below a business-defined threshold (e.g. 24–72 hours).",
            "2. **First-transaction rules** — Combine signup timing with purchase value and channel for "
            "a lightweight rules layer ahead of the ML model.",
            "3. **Channel monitoring** — Review Direct-traffic conversion funnels; higher fraud rates may "
            "indicate acquisition fraud or weak signup controls on that path.",
            "4. **Do not block by country alone** — Geographic features inform prioritization, not blanket denial.",
            "",
            "### Customer verification",
            "",
            "1. **Step-up verification for high-score new users** — Email, phone, or payment 3-D Secure "
            "when signup-to-purchase time is very short and the model score exceeds threshold.",
            "2. **Manual review queue** — With **99%+ precision**, model-flagged cases are strong review "
            "candidates; analysts can focus on the highest-scored transactions first.",
            "3. **Unknown geography handling** — Transactions with unmapped IP/country should route to "
            "verification rather than auto-approval.",
            "",
            "### Operational monitoring",
            "",
            "1. **Track recall, not just precision** — Current recall (~53%) means ~47% of fraud is missed; "
            "monitor `false_negatives` weekly and tune thresholds if missed fraud is too costly.",
            "2. **Alert volume dashboard** — With only **13** false positives on holdout, the model is "
            "operationally efficient; watch for drift if alert volume spikes.",
            "3. **Feature drift alerts** — Monitor distributions of `time_since_signup_hours`, top countries, "
            "and channel mix; fraud tactics shift over time.",
            "4. **Retrain cadence** — Re-run tuning and SHAP analysis quarterly or after major fraud incidents.",
            "",
            "---",
            "",
            "## For data scientists",
            "",
            f"- **Best model:** `{best_name}` selected by PR-AUC ({pr_auc:.4f}) vs logistic regression "
            f"({float(metrics[metrics['model_name'] == 'logistic_regression']['pr_auc'].iloc[0]) if 'pr_auc' in metrics.columns else float(metrics[metrics['model_name'] == 'logistic_regression']['auc_pr'].iloc[0]):.4f}).",
            f"- **Metrics:** precision={precision:.4f}, recall={recall:.4f}, F1={f1:.4f}, "
            f"ROC-AUC={float(best['roc_auc']):.4f}.",
            "- **Imbalance:** SMOTE on train only; holdout retains ~9.4% fraud prevalence.",
            "- **Feature dominance:** A single engineered feature (`time_since_signup_hours`) drives most "
            "importance — validate with SHAP and test threshold policies for recall improvement.",
            "- **Velocity features:** Uniform in current data (one transaction per user); exclude or deprioritize "
            "until repeat-purchase data is available.",
            "- **Next steps:** Threshold tuning on PR curve, SHAP case review, and stream-specific "
            "ops playbooks for both Fraud_Data and creditcard.",
            "",
            "## Unified solution (both streams)",
            "",
            "- **E-commerce (`Fraud_Data`):** best model `random_forest_tuned` — high precision (~99%), "
            "dominant signal `time_since_signup_hours`.",
            "- **Banking (`creditcard.csv`):** best model `random_forest` — PR-AUC ~0.81, recall ~76%, "
            "dominant PCA drivers `V14` / `V17` / `V10` (see `reports/outputs/creditcard/`).",
            "- **Shared stack:** stratified split → SMOTE on train → tune LR/RF/XGBoost on PR-AUC → "
            "holdout evaluate → SHAP (`run_unified_modeling.py`).",
            "",
            "## For fraud analysts",
            "",
            "- **Prioritize:** New accounts buying within hours of signup — this is the clearest pattern.",
            "- **Review queue:** Model alerts are highly precise; investigate flagged orders before release.",
            "- **Context matters:** Combine model score with payment history, customer contact verification, "
            "and shipping/billing mismatches.",
            "- **Geography:** Elevated country signals mean \"review harder,\" not \"decline automatically.\"",
            f"- **Volume expectation:** On a similar holdout, expect ~{fp} false alarms per "
            f"{tp + fp + fn + tn:,} transactions at the current threshold — very low analyst noise.",
            "",
            "## For executives",
            "",
            "- **Business problem:** ~9.4% of e-commerce transactions are fraudulent; undetected fraud "
            "has direct revenue and chargeback cost.",
            "- **Model readiness:** Strong precision (99%+) supports a **review-first** deployment — the model "
            "reliably tells you which cases to investigate, not which to auto-block without human oversight.",
            "- **Gap to close:** Recall near 53% means roughly half of fraud still slips through at default "
            "settings; investment in threshold tuning and verification workflows will improve capture rate.",
            "- **Highest-ROI control:** Policies around **new-account purchase timing** address the single "
            "strongest signal and do not require model inference alone.",
            "- **Responsible AI:** Use geographic and channel signals for triage, not discriminatory blocking; "
            "document decisions and audit flagged cases regularly.",
        ]
    )

    report_dir_resolved = report_dir.resolve()
    try:
        artifacts_base = report_dir_resolved.relative_to(REPORTS_DIR.resolve().parent)
    except ValueError:
        artifacts_base = report_dir_resolved

    lines.extend(
        [
            "",
            "---",
            "",
            "## Source artifacts",
            "",
            f"- Model metrics: `{artifacts_base / 'model_comparison_metrics.csv'}`",
            f"- Best model summary: `{artifacts_base / 'best_model_summary.csv'}`",
            f"- Feature importance: `{artifacts_base / 'top_features_best_model.csv'}`",
            "- EDA reference: `notebooks/eda-fraud-data.ipynb`",
            "- Class imbalance: `reports/class_imbalance_summary.md`",
        ]
    )

    return "\n".join(lines)


def save_fraud_insights(
    *,
    output_path: Path | None = None,
    modeling_dir: Path | None = None,
) -> Path:
    """Write executive fraud insights markdown to reports/."""
    path = output_path or (REPORTS_DIR / INSIGHTS_FILENAME)
    ensure_dir(path.parent)
    path.write_text(build_fraud_insights_markdown(modeling_dir=modeling_dir), encoding="utf-8")
    return path
