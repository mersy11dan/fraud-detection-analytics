"""
Fraud Detection Analytics — Streamlit dashboard.

Run from project root:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MODELING_DIR_CANDIDATES = (
    REPORTS_DIR / "outputs",
    REPORTS_DIR / "modeling",
)
PLOT_DIR_CANDIDATES = (
    REPORTS_DIR / "outputs" / "plots",
    REPORTS_DIR / "modeling",
    REPORTS_DIR / "figures",
)

PAGE_TITLE = "Fraud Detection Analytics"
PAGE_ICON = "🛡️"


# ---------------------------------------------------------------------------
# Data loaders (cached)
# ---------------------------------------------------------------------------


def _resolve_modeling_dir() -> Path | None:
    for candidate in MODELING_DIR_CANDIDATES:
        if (candidate / "model_comparison_metrics.csv").exists():
            return candidate
    return None


def _read_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def _read_text(path: Path) -> str | None:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def _normalize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "pr_auc" not in out.columns and "auc_pr" in out.columns:
        out = out.rename(columns={"auc_pr": "pr_auc"})
    return out


@st.cache_data(show_spinner=False)
def load_model_metrics() -> pd.DataFrame | None:
    modeling_dir = _resolve_modeling_dir()
    if modeling_dir is None:
        return None
    df = _read_csv(modeling_dir / "model_comparison_metrics.csv")
    return _normalize_metrics(df) if df is not None else None


@st.cache_data(show_spinner=False)
def load_best_model_summary() -> pd.Series | None:
    modeling_dir = _resolve_modeling_dir()
    if modeling_dir is None:
        return None
    df = _read_csv(modeling_dir / "best_model_summary.csv")
    if df is None or df.empty:
        return None
    row = df.iloc[0].copy()
    if "pr_auc" not in row.index and "auc_pr" in row.index:
        row["pr_auc"] = row["auc_pr"]
    return row


@st.cache_data(show_spinner=False)
def load_feature_importance() -> pd.DataFrame | None:
    modeling_dir = _resolve_modeling_dir()
    if modeling_dir is None:
        return None
    return _read_csv(modeling_dir / "top_features_best_model.csv")


@st.cache_data(show_spinner=False)
def load_class_imbalance() -> pd.DataFrame | None:
    return _read_csv(REPORTS_DIR / "class_imbalance_comparison.csv")


@st.cache_data(show_spinner=False)
def load_shap_features() -> pd.DataFrame | None:
    return _read_csv(REPORTS_DIR / "shap" / "shap_top_features.csv")


@st.cache_data(show_spinner=False)
def load_fraud_insights() -> str | None:
    return _read_text(REPORTS_DIR / "fraud_insights.md")


@st.cache_data(show_spinner=False)
def load_engineered_data() -> pd.DataFrame:
    path = DATA_PROCESSED_DIR / "fraud_data_engineered.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_feature_matrix() -> pd.DataFrame:
    path = DATA_PROCESSED_DIR / "fraud_data_features.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def find_plot_filenames() -> dict[str, Path]:
    names = {
        "pr_curve": "model_comparison_pr_curve.png",
        "roc_curve": "model_comparison_roc_curve.png",
        "feature_importance": "best_model_feature_importance.png",
        "shap_summary": "shap_summary_plot.png",
        "shap_bar": "shap_bar_plot.png",
    }
    found: dict[str, Path] = {}
    for plot_dir in PLOT_DIR_CANDIDATES:
        if not plot_dir.exists():
            continue
        for key, filename in names.items():
            if key in found:
                continue
            candidate = plot_dir / filename
            if candidate.exists():
                found[key] = candidate
    shap_dir = REPORTS_DIR / "figures"
    if shap_dir.exists():
        for path in shap_dir.glob("*.png"):
            stem = path.stem.lower()
            if "summary" in stem and "shap_summary" not in found:
                found["shap_summary"] = path
            elif "bar" in stem and "shap_bar" not in found:
                found["shap_bar"] = path
    return found


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def apply_page_style() -> None:
    st.markdown(
        """
        <style>
            .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px; }
            div[data-testid="stMetric"] {
                background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
                border: 1px solid #e2e8f0;
                border-radius: 0.75rem;
                padding: 0.75rem 1rem;
            }
            div[data-testid="stMetricLabel"] { font-size: 0.85rem; color: #64748b; }
            div[data-testid="stMetricValue"] { font-size: 1.6rem; color: #0f172a; }
            h1 { color: #0f172a; font-weight: 700; }
            h2, h3 { color: #1e293b; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, delta: str | None = None, help_text: str | None = None) -> None:
    st.metric(label=label, value=value, delta=delta, help=help_text)


def missing_data_message(artifact: str) -> None:
    st.warning(
        f"**{artifact}** not found. Run the modeling pipeline to generate reports under `reports/`."
    )


def plotly_template() -> str:
    return "plotly_white"


def metric_bar_chart(metrics: pd.DataFrame, metric: str, title: str) -> go.Figure:
    chart_df = metrics.sort_values(metric, ascending=True)
    fig = px.bar(
        chart_df,
        x=metric,
        y="model_name",
        orientation="h",
        title=title,
        text=chart_df[metric].map(lambda v: f"{v:.3f}"),
        color=metric,
        color_continuous_scale="Blues",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        template=plotly_template(),
        height=320,
        coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title=metric.replace("_", " ").title(),
        yaxis_title="",
    )
    return fig


def confusion_matrix_heatmap(tp: int, fp: int, fn: int, tn: int) -> go.Figure:
    matrix = [[tn, fp], [fn, tp]]
    labels = [["TN", "FP"], ["FN", "TP"]]
    text = [[f"{labels[i][j]}<br>{matrix[i][j]:,}" for j in range(2)] for i in range(2)]
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=["Predicted Legitimate", "Predicted Fraud"],
            y=["Actual Legitimate", "Actual Fraud"],
            text=text,
            texttemplate="%{text}",
            colorscale=[[0, "#e2e8f0"], [1, "#1d4ed8"]],
            showscale=False,
        )
    )
    fig.update_layout(
        template=plotly_template(),
        title="Best Model — Holdout Confusion Matrix",
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def page_overview() -> None:
    st.title("Overview")
    st.caption("Executive snapshot of fraud prevalence, model performance, and key risk signals.")

    engineered = load_engineered_data()
    best = load_best_model_summary()
    metrics = load_model_metrics()
    features = load_feature_importance()

    if engineered.empty:
        missing_data_message("Processed dataset (`data/processed/fraud_data_engineered.csv`)")
        return

    fraud_rate = engineered["class"].mean()
    total_txns = len(engineered)
    fraud_count = int(engineered["class"].sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi_card("Total Transactions", f"{total_txns:,}")
    with c2:
        kpi_card("Fraud Rate", f"{fraud_rate:.2%}", help_text="Share of transactions labeled fraudulent.")
    with c3:
        kpi_card("Fraud Cases", f"{fraud_count:,}")
    with c4:
        if best is not None:
            kpi_card("Best Model PR-AUC", f"{float(best['pr_auc']):.3f}")
        else:
            kpi_card("Best Model PR-AUC", "—")
    with c5:
        if best is not None:
            kpi_card("Precision", f"{float(best['precision']):.1%}")
        else:
            kpi_card("Precision", "—")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Class distribution")
        class_df = pd.DataFrame(
            {
                "class": ["Legitimate", "Fraud"],
                "count": [total_txns - fraud_count, fraud_count],
            }
        )
        fig = px.pie(
            class_df,
            names="class",
            values="count",
            hole=0.45,
            color="class",
            color_discrete_map={"Legitimate": "#94a3b8", "Fraud": "#dc2626"},
        )
        fig.update_layout(template=plotly_template(), height=360, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Model comparison")
        if metrics is not None and not metrics.empty:
            fig = metric_bar_chart(metrics, "pr_auc", "PR-AUC by model")
            st.plotly_chart(fig, use_container_width=True)
        else:
            missing_data_message("Model comparison metrics")

    if features is not None and not features.empty:
        st.subheader("Top fraud indicators")
        top = features.head(8)
        fig = px.bar(
            top,
            x="importance",
            y="feature",
            orientation="h",
            title="Best model — top feature importance",
            color="importance",
            color_continuous_scale="Reds",
        )
        fig.update_layout(
            template=plotly_template(),
            height=360,
            yaxis=dict(categoryorder="total ascending"),
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    if best is not None:
        st.info(
            f"**{best['model_name']}** is the current best model "
            f"(recall **{float(best['recall']):.1%}**, F1 **{float(best['f1']):.3f}**). "
            "See **Model Performance** and **Executive Recommendations** for deployment guidance."
        )


def page_dataset_summary() -> None:
    st.title("Dataset Summary")
    st.caption("Fraud_Data profile, feature matrix shape, and class-imbalance handling.")

    engineered = load_engineered_data()
    feature_matrix = load_feature_matrix()
    imbalance = load_class_imbalance()

    if engineered.empty:
        missing_data_message("Engineered dataset")
        return

    fraud_rate = engineered["class"].mean()
    n_users = engineered["user_id"].nunique()
    n_features = len(feature_matrix.columns) - 1 if not feature_matrix.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Observations", f"{len(engineered):,}")
    with c2:
        kpi_card("Unique Users", f"{n_users:,}")
    with c3:
        kpi_card("Fraud Rate", f"{fraud_rate:.2%}")
    with c4:
        kpi_card("Model Features", f"{n_features:,}")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Dataset columns (engineered)")
        profile = pd.DataFrame(
            {
                "column": engineered.columns,
                "dtype": engineered.dtypes.astype(str).values,
                "non_null_pct": (1 - engineered.isna().mean()).mul(100).round(1).values,
            }
        )
        st.dataframe(profile, use_container_width=True, height=320)

    with col_b:
        st.subheader("Numeric feature summary")
        numeric_cols = [
            "purchase_value",
            "age",
            "time_since_signup_hours",
            "hour_of_day",
            "day_of_week",
        ]
        available = [c for c in numeric_cols if c in engineered.columns]
        if available:
            st.dataframe(engineered[available].describe().T.round(2), use_container_width=True)

    if imbalance is not None and not imbalance.empty:
        st.subheader("Train / test class distribution")
        chart_df = imbalance.copy()
        chart_df["class_label"] = chart_df["class"].map({0: "Legitimate", 1: "Fraud"})
        fig = px.bar(
            chart_df,
            x="stage",
            y="pct",
            color="class_label",
            barmode="group",
            title="Class balance across pipeline stages (%)",
            labels={"stage": "Stage", "pct": "Percentage", "class_label": "Class"},
            color_discrete_map={"Legitimate": "#64748b", "Fraud": "#dc2626"},
        )
        fig.update_layout(template=plotly_template(), height=400, xaxis_tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            imbalance.assign(
                class_label=imbalance["class"].map({0: "Legitimate", 1: "Fraud"})
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        missing_data_message("Class imbalance report (`reports/class_imbalance_comparison.csv`)")

    summary_md = _read_text(REPORTS_DIR / "class_imbalance_summary.md")
    if summary_md:
        with st.expander("Class imbalance methodology"):
            st.markdown(summary_md)


def page_fraud_eda() -> None:
    st.title("Fraud EDA")
    st.caption("Interactive exploration of fraud patterns in the engineered e-commerce dataset.")

    df = load_engineered_data()
    if df.empty:
        missing_data_message("Engineered dataset")
        return

    fraud_rate = df["class"].mean()
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Overall Fraud Rate", f"{fraud_rate:.2%}")
    with c2:
        kpi_card("Median Purchase Value", f"${df['purchase_value'].median():,.0f}")
    with c3:
        kpi_card(
            "Median Signup-to-Purchase (hrs)",
            f"{df.loc[df['class'] == 0, 'time_since_signup_hours'].median():,.0f}",
            help_text="Legitimate customers (class 0).",
        )

    st.divider()

    tab_source, tab_country, tab_timing, tab_value = st.tabs(
        ["Acquisition channel", "Geography", "Signup timing", "Purchase value"]
    )

    with tab_source:
        if "source" in df.columns:
            by_source = (
                df.groupby("source", as_index=False)
                .agg(transactions=("class", "size"), fraud_rate=("class", "mean"))
                .sort_values("fraud_rate", ascending=False)
            )
            fig = px.bar(
                by_source,
                x="source",
                y="fraud_rate",
                text=by_source["fraud_rate"].map(lambda v: f"{v:.1%}"),
                title="Fraud rate by acquisition channel",
                color="fraud_rate",
                color_continuous_scale="Reds",
            )
            fig.update_layout(template=plotly_template(), height=380, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(
                by_source.assign(fraud_rate=by_source["fraud_rate"].map(lambda v: f"{v:.2%}")),
                use_container_width=True,
                hide_index=True,
            )

    with tab_country:
        if "country" in df.columns:
            min_txns = st.slider("Minimum transactions per country", 500, 5000, 1000, 250)
            by_country = (
                df.groupby("country", as_index=False)
                .agg(transactions=("class", "size"), fraud_rate=("class", "mean"))
                .query("transactions >= @min_txns")
                .sort_values("fraud_rate", ascending=False)
                .head(15)
            )
            fig = px.bar(
                by_country,
                x="fraud_rate",
                y="country",
                orientation="h",
                title=f"Fraud rate by country (≥ {min_txns:,} transactions)",
                color="fraud_rate",
                color_continuous_scale="Oranges",
            )
            fig.update_layout(
                template=plotly_template(),
                height=420,
                yaxis=dict(categoryorder="total ascending"),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab_timing:
        sample = df.sample(min(25_000, len(df)), random_state=42)
        sample["class_label"] = sample["class"].map({0: "Legitimate", 1: "Fraud"})
        cap_hours = st.number_input("Cap signup-to-purchase hours (chart)", 24, 2000, 720, 24)
        sample["signup_hours_capped"] = sample["time_since_signup_hours"].clip(upper=cap_hours)
        fig = px.histogram(
            sample,
            x="signup_hours_capped",
            color="class_label",
            barmode="overlay",
            nbins=40,
            opacity=0.65,
            title="Signup-to-purchase time distribution",
            labels={"signup_hours_capped": "Hours since signup"},
            color_discrete_map={"Legitimate": "#64748b", "Fraud": "#dc2626"},
        )
        fig.update_layout(template=plotly_template(), height=400)
        st.plotly_chart(fig, use_container_width=True)

        by_hour = (
            df.groupby("hour_of_day", as_index=False)
            .agg(fraud_rate=("class", "mean"), transactions=("class", "size"))
        )
        fig2 = px.line(
            by_hour,
            x="hour_of_day",
            y="fraud_rate",
            markers=True,
            title="Fraud rate by hour of day",
        )
        fig2.update_layout(template=plotly_template(), height=320)
        st.plotly_chart(fig2, use_container_width=True)

    with tab_value:
        sample = df.sample(min(20_000, len(df)), random_state=42)
        sample["class_label"] = sample["class"].map({0: "Legitimate", 1: "Fraud"})
        fig = px.box(
            sample,
            x="class_label",
            y="purchase_value",
            color="class_label",
            title="Purchase value by class",
            color_discrete_map={"Legitimate": "#64748b", "Fraud": "#dc2626"},
        )
        fig.update_layout(template=plotly_template(), height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def page_model_performance() -> None:
    st.title("Model Performance")
    st.caption("Holdout evaluation for tuned classifiers (SMOTE on train only).")

    metrics = load_model_metrics()
    best = load_best_model_summary()
    features = load_feature_importance()
    plots = find_plot_filenames()

    if metrics is None or metrics.empty:
        missing_data_message("Model comparison metrics")
        return

    best_name = str(best["model_name"]) if best is not None else metrics.iloc[0]["model_name"]

    c1, c2, c3, c4, c5 = st.columns(5)
    best_row = metrics.loc[metrics["model_name"] == best_name].iloc[0]
    with c1:
        kpi_card("Best Model", best_name)
    with c2:
        kpi_card("PR-AUC", f"{float(best_row['pr_auc']):.3f}")
    with c3:
        kpi_card("Precision", f"{float(best_row['precision']):.1%}")
    with c4:
        kpi_card("Recall", f"{float(best_row['recall']):.1%}")
    with c5:
        kpi_card("F1", f"{float(best_row['f1']):.3f}")

    st.divider()

    metric_choice = st.multiselect(
        "Metrics to compare",
        ["precision", "recall", "f1", "roc_auc", "pr_auc"],
        default=["precision", "recall", "f1", "pr_auc"],
    )

    if metric_choice:
        melted = metrics.melt(
            id_vars="model_name",
            value_vars=metric_choice,
            var_name="metric",
            value_name="score",
        )
        fig = px.bar(
            melted,
            x="model_name",
            y="score",
            color="metric",
            barmode="group",
            title="Model metrics comparison",
            text=melted["score"].map(lambda v: f"{v:.3f}"),
        )
        fig.update_layout(template=plotly_template(), height=420)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detailed metrics")
    display_metrics = metrics.copy()
    for col in ["precision", "recall", "f1", "roc_auc", "pr_auc", "accuracy"]:
        if col in display_metrics.columns:
            display_metrics[col] = display_metrics[col].map(lambda v: f"{v:.4f}")
    st.dataframe(display_metrics, use_container_width=True, hide_index=True)

    col_left, col_right = st.columns(2)

    with col_left:
        if best is not None:
            tp = int(best["true_positives"])
            fp = int(best["false_positives"])
            fn = int(best["false_negatives"])
            tn = int(best["true_negatives"])
            st.plotly_chart(confusion_matrix_heatmap(tp, fp, fn, tn), use_container_width=True)

    with col_right:
        if features is not None and not features.empty:
            top = features.head(12)
            fig = px.bar(
                top,
                x="importance",
                y="feature",
                orientation="h",
                title="Feature importance (best model)",
                color="importance",
                color_continuous_scale="Blues",
            )
            fig.update_layout(
                template=plotly_template(),
                height=380,
                yaxis=dict(categoryorder="total ascending"),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Saved evaluation plots")
    plot_cols = st.columns(2)
    static_plots = [
        ("PR curve", plots.get("pr_curve")),
        ("ROC curve", plots.get("roc_curve")),
        ("Feature importance", plots.get("feature_importance")),
    ]
    for idx, (label, path) in enumerate(static_plots):
        with plot_cols[idx % 2]:
            if path is not None:
                st.image(str(path), caption=label, use_container_width=True)
            else:
                st.caption(f"{label} — not generated yet (`reports/outputs/plots/`).")


def page_shap_explainability() -> None:
    st.title("SHAP Explainability")
    st.caption("Model-agnostic explanations for fraud prediction drivers.")

    shap_df = load_shap_features()
    shap_md = _read_text(REPORTS_DIR / "shap" / "shap_feature_summary.md")
    plots = find_plot_filenames()

    if shap_df is None or shap_df.empty:
        missing_data_message("SHAP feature summary (`reports/shap/shap_top_features.csv`)")
        st.code("python scripts/run_shap_analysis.py", language="bash")
        return

    top_feature = shap_df.iloc[0]["feature"]
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Features Explained", f"{len(shap_df)}")
    with c2:
        kpi_card("Top SHAP Driver", str(top_feature))
    with c3:
        kpi_card("Top |SHAP|", f"{float(shap_df.iloc[0]['mean_abs_shap']):.4f}")

    st.divider()

    top_n = st.slider("Number of features to display", 5, min(20, len(shap_df)), 10)

    chart_df = shap_df.head(top_n).copy()
    chart_df["direction_label"] = chart_df["impact_direction"].str.replace("_", " ").str.title()

    fig = px.bar(
        chart_df,
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        color="impact_direction",
        title="Mean |SHAP| — top fraud predictors",
        labels={"mean_abs_shap": "Mean |SHAP value|", "impact_direction": "Impact"},
        color_discrete_map={
            "increases_fraud_risk": "#dc2626",
            "decreases_fraud_risk": "#2563eb",
        },
    )
    fig.update_layout(
        template=plotly_template(),
        height=420,
        yaxis=dict(categoryorder="total ascending"),
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.scatter(
        chart_df,
        x="mean_shap",
        y="feature",
        size="mean_abs_shap",
        color="impact_direction",
        title="SHAP direction vs magnitude",
        labels={"mean_shap": "Mean SHAP value", "impact_direction": "Impact"},
        color_discrete_map={
            "increases_fraud_risk": "#dc2626",
            "decreases_fraud_risk": "#2563eb",
        },
    )
    fig2.update_layout(template=plotly_template(), height=380, yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(shap_df, use_container_width=True, hide_index=True)

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        if "shap_summary" in plots:
            st.image(str(plots["shap_summary"]), caption="SHAP summary plot", use_container_width=True)
        else:
            st.caption("SHAP summary plot not found in `reports/figures/`.")
    with img_col2:
        if "shap_bar" in plots:
            st.image(str(plots["shap_bar"]), caption="SHAP bar plot", use_container_width=True)
        else:
            st.caption("SHAP bar plot not found in `reports/figures/`.")

    if shap_md:
        with st.expander("Analyst interpretation (full SHAP summary)"):
            st.markdown(shap_md)


def page_executive_recommendations() -> None:
    st.title("Executive Recommendations")
    st.caption("Actionable fraud prevention, verification, and monitoring guidance.")

    insights = load_fraud_insights()
    best = load_best_model_summary()

    if best is not None:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Deployment Model", str(best["model_name"]))
        with c2:
            kpi_card("Precision", f"{float(best['precision']):.1%}")
        with c3:
            kpi_card("Recall", f"{float(best['recall']):.1%}")
        with c4:
            kpi_card("False Positives (holdout)", f"{int(best['false_positives']):,}")

    if insights is None:
        missing_data_message("Executive insights (`reports/fraud_insights.md`)")
        st.code("python scripts/run_fraud_insights.py", language="bash")
        return

    st.divider()

    tab_exec, tab_prevent, tab_verify, tab_monitor, tab_full = st.tabs(
        [
            "Executive summary",
            "Fraud prevention",
            "Verification",
            "Monitoring",
            "Full report",
        ]
    )

    sections = _split_markdown_sections(insights)

    with tab_exec:
        st.markdown(sections.get("executive summary", "_No executive summary found._"))
        st.markdown(sections.get("top fraud indicators", ""))
        st.markdown(sections.get("high-risk customer behaviors", ""))
        st.markdown(sections.get("high-risk transaction characteristics", ""))

    with tab_prevent:
        st.markdown(sections.get("fraud prevention", "_See full report for prevention recommendations._"))

    with tab_verify:
        st.markdown(
            sections.get("customer verification", "_See full report for verification recommendations._")
        )

    with tab_monitor:
        st.markdown(
            sections.get(
                "operational monitoring",
                "_See full report for operational monitoring recommendations._",
            )
        )

    with tab_full:
        st.markdown(insights)

    st.download_button(
        label="Download fraud insights (Markdown)",
        data=insights,
        file_name="fraud_insights.md",
        mime="text/markdown",
    )


def _split_markdown_sections(markdown: str) -> dict[str, str]:
    """Split markdown by ## and ### headings into lowercase keys."""
    sections: dict[str, str] = {}
    current_key: str | None = None
    buffer: list[str] = []

    for line in markdown.splitlines():
        if line.startswith("## "):
            if current_key is not None:
                sections[current_key] = "\n".join(buffer).strip()
            current_key = line[3:].strip().lower()
            buffer = []
        elif line.startswith("### "):
            if current_key is not None:
                sections[current_key] = "\n".join(buffer).strip()
            current_key = line[4:].strip().lower()
            buffer = []
        else:
            buffer.append(line)

    if current_key is not None:
        sections[current_key] = "\n".join(buffer).strip()

    return sections


# ---------------------------------------------------------------------------
# App entry
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_page_style()

    with st.sidebar:
        st.markdown(f"## {PAGE_ICON} Fraud Analytics")
        st.caption("E-commerce fraud detection — modeling & insights")
        st.divider()
        modeling_dir = _resolve_modeling_dir()
        if modeling_dir:
            st.success("Reports loaded", icon="✅")
            st.caption(f"`{modeling_dir.relative_to(PROJECT_ROOT)}`")
        else:
            st.warning("Some model reports missing", icon="⚠️")
        st.caption("Regenerate: `python scripts/run_modeling_reports.py`")

    pages = [
        st.Page(page_overview, title="Overview", icon="📊", default=True),
        st.Page(page_dataset_summary, title="Dataset Summary", icon="📁"),
        st.Page(page_fraud_eda, title="Fraud EDA", icon="🔍"),
        st.Page(page_model_performance, title="Model Performance", icon="📈"),
        st.Page(page_shap_explainability, title="SHAP Explainability", icon="🧠"),
        st.Page(page_executive_recommendations, title="Executive Recommendations", icon="💼"),
    ]

    navigation = st.navigation(pages)
    navigation.run()

    with st.sidebar:
        st.divider()
        st.caption("Fraud Detection Analytics · 10 Academy")


if __name__ == "__main__":
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    main()
