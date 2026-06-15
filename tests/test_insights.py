"""Tests for executive fraud insights generation."""

from pathlib import Path

import pandas as pd
import pytest

from src.modeling.insights import build_fraud_insights_markdown, save_fraud_insights


def _write_sample_modeling_outputs(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            {
                "model_name": "random_forest_tuned",
                "precision": 0.99,
                "recall": 0.53,
                "f1": 0.69,
                "roc_auc": 0.77,
                "pr_auc": 0.62,
                "true_positives": 100,
                "false_positives": 2,
                "false_negatives": 90,
                "true_negatives": 1000,
            },
            {
                "model_name": "logistic_regression",
                "precision": 0.17,
                "recall": 0.69,
                "f1": 0.27,
                "roc_auc": 0.74,
                "pr_auc": 0.39,
                "true_positives": 120,
                "false_positives": 500,
                "false_negatives": 70,
                "true_negatives": 502,
            },
        ]
    ).to_csv(tmp_path / "model_comparison_metrics.csv", index=False)

    pd.DataFrame(
        [
            {
                "model_name": "random_forest_tuned",
                "precision": 0.99,
                "recall": 0.53,
                "f1": 0.69,
                "roc_auc": 0.77,
                "pr_auc": 0.62,
                "true_positives": 100,
                "false_positives": 2,
                "false_negatives": 90,
                "true_negatives": 1000,
            }
        ]
    ).to_csv(tmp_path / "best_model_summary.csv", index=False)

    pd.DataFrame(
        {
            "feature": ["time_since_signup_hours", "country_United States", "hour_of_day"],
            "importance": [0.68, 0.05, 0.03],
            "coefficient": [None, None, None],
            "direction": ["n/a", "n/a", "n/a"],
            "importance_type": ["feature_importance"] * 3,
        }
    ).to_csv(tmp_path / "top_features_best_model.csv", index=False)


def test_build_fraud_insights_markdown_contains_audience_sections(tmp_path: Path) -> None:
    _write_sample_modeling_outputs(tmp_path)
    markdown = build_fraud_insights_markdown(modeling_dir=tmp_path)

    assert "Executive summary" in markdown
    assert "For data scientists" in markdown
    assert "For fraud analysts" in markdown
    assert "For executives" in markdown
    assert "time_since_signup_hours" in markdown


def test_save_fraud_insights_writes_file(tmp_path: Path) -> None:
    _write_sample_modeling_outputs(tmp_path)
    output_path = save_fraud_insights(
        output_path=tmp_path / "fraud_insights.md",
        modeling_dir=tmp_path,
    )

    assert output_path.exists()
    assert "Fraud prevention" in output_path.read_text(encoding="utf-8")
