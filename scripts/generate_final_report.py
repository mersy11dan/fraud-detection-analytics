"""Generate a single, self-contained HTML final report for the Fraud Detection project.

- Auto-discovers figures under the repo and embeds them as base64 data URIs so the
  resulting HTML is fully self-contained (safe for Chrome -> Print -> Save as PDF).
- Builds the model-comparison tables from the actual metrics CSVs.
- Computes dataset dimensions directly from the raw CSV files.

No metrics or findings are invented; everything is read from real repository outputs.
"""

from __future__ import annotations

import base64
import csv
import datetime as dt
import glob
import html
import mimetypes
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "reports", "final_report.html")

# Section -> list of glob patterns (relative to repo root). Discovery is automatic
# within each pattern, so new matching figures are picked up on regeneration.
SECTION_FIGURES = {
    "eda": ["docs/figures/fraud_class_distribution.png", "docs/figures/creditcard_class_distribution.png"],
    "geolocation": ["reports/figures/geolocation*/*", "docs/figures/*geo*"],
    "features": ["docs/figures/time_since_signup*"],
    "imbalance": ["docs/figures/smote_class_distribution.png"],
    "modeling": [
        "reports/modeling/plots/*.png",
        "reports/outputs/plots/*.png",
        "reports/outputs/confusion_matrices/*.png",
        "reports/outputs/creditcard/plots/*.png",
        "reports/outputs/creditcard/confusion_matrices/*.png",
    ],
    "shap": [
        "reports/figures/shap_*.png",
        "reports/figures/creditcard/shap_*.png",
    ],
}

METRICS_FILES = {
    "E-commerce (Fraud_Data.csv)": "reports/modeling/model_comparison_metrics.csv",
    "Banking (creditcard.csv)": "reports/outputs/creditcard/model_comparison_metrics.csv",
}

RAW_DATASETS = {
    "Fraud_Data.csv": "data/raw/Fraud_Data.csv",
    "creditcard.csv": "data/raw/creditcard.csv",
}

# Friendly metric column labels (only columns present in a CSV are shown)
METRIC_COLUMNS = [
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("roc_auc", "ROC-AUC"),
    ("auc_pr", "AUC-PR"),
    ("pr_auc", "AUC-PR"),
]

_fig_counter = 0


def embed_image(rel_path: str) -> str:
    abspath = os.path.join(REPO, rel_path)
    mime = mimetypes.guess_type(abspath)[0] or "image/png"
    with open(abspath, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def discover(patterns: list[str]) -> list[str]:
    found: list[str] = []
    for pat in patterns:
        for p in sorted(glob.glob(os.path.join(REPO, pat))):
            if os.path.isfile(p):
                rel = os.path.relpath(p, REPO).replace("\\", "/")
                if rel not in found:
                    found.append(rel)
    return found


def gallery(patterns: list[str], empty_msg: str) -> str:
    global _fig_counter
    rels = discover(patterns)
    if not rels:
        return f"<div class='placeholder'>{html.escape(empty_msg)}</div>"
    cards = []
    for rel in rels:
        _fig_counter += 1
        name = os.path.basename(rel)
        try:
            src = embed_image(rel)
        except Exception as e:  # pragma: no cover
            cards.append(f"<div class='placeholder'>Could not embed {html.escape(rel)}: {e}</div>")
            continue
        cards.append(
            f"<figure class='fig'><img src='{src}' alt='{html.escape(name)}'/>"
            f"<figcaption><span class='fignum'>Figure {_fig_counter}.</span> "
            f"{html.escape(name)}</figcaption></figure>"
        )
    return f"<div class='gallery'>{''.join(cards)}</div>"


def fmt(value: str) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return html.escape(str(value))


def metrics_table(rel_csv: str) -> tuple[str, dict | None]:
    abspath = os.path.join(REPO, rel_csv)
    if not os.path.exists(abspath):
        return "<div class='placeholder'>Metrics file not found.</div>", None
    with open(abspath, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return "<div class='placeholder'>Metrics file is empty.</div>", None

    present = []
    seen_labels = set()
    for key, label in METRIC_COLUMNS:
        if key in rows[0] and label not in seen_labels:
            present.append((key, label))
            seen_labels.add(label)

    header = "<tr><th>Model</th>" + "".join(f"<th>{l}</th>" for _, l in present) + "</tr>"
    best = max(rows, key=lambda r: float(r.get("roc_auc", 0) or 0))
    body = []
    for r in rows:
        is_best = r is best
        cells = "".join(f"<td>{fmt(r.get(k, ''))}</td>" for k, _ in present)
        name = html.escape(r["model_name"])
        star = " <span class='badge'>best ROC-AUC</span>" if is_best else ""
        body.append(f"<tr{' class=highlight' if is_best else ''}><td>{name}{star}</td>{cells}</tr>")
    table = f"<table><thead>{header}</thead><tbody>{''.join(body)}</tbody></table>"
    return table, {"model": best["model_name"], "roc_auc": float(best.get("roc_auc", 0) or 0)}


def dataset_dims(rel_csv: str) -> tuple[int, int, str]:
    abspath = os.path.join(REPO, rel_csv)
    if not os.path.exists(abspath):
        return 0, 0, ""
    with open(abspath, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n").split(",")
        rows = sum(1 for _ in f)
    cols = len(header)
    sample = ", ".join(header[:8]) + (" ..." if len(header) > 8 else "")
    return rows, cols, sample


def kpi(value: str, label: str) -> str:
    return f"<div class='kpi'><div class='kpi-value'>{value}</div><div class='kpi-label'>{html.escape(label)}</div></div>"


def build() -> str:
    today = dt.date.today().strftime("%B %d, %Y")

    # Metrics tables + best models
    metrics_sections = []
    best_overall = {"model": "", "roc_auc": -1.0, "dataset": ""}
    for dataset, rel in METRICS_FILES.items():
        table, best = metrics_table(rel)
        metrics_sections.append(f"<h3>{html.escape(dataset)}</h3>{table}")
        if best and best["roc_auc"] > best_overall["roc_auc"]:
            best_overall = {**best, "dataset": dataset}

    # Dataset overview
    ds_rows = []
    for name, rel in RAW_DATASETS.items():
        r, c, sample = dataset_dims(rel)
        ds_rows.append(
            f"<tr><td><code>{html.escape(name)}</code></td>"
            f"<td>{r:,}</td><td>{c}</td><td>{html.escape(sample)}</td></tr>"
        )
    dataset_table = (
        "<table><thead><tr><th>Dataset</th><th>Rows</th><th>Columns</th><th>Key Variables (first columns)</th></tr></thead>"
        f"<tbody>{''.join(ds_rows)}</tbody></table>"
    )

    fr_rows, fr_cols, _ = dataset_dims(RAW_DATASETS["Fraud_Data.csv"])
    cc_rows, cc_cols, _ = dataset_dims(RAW_DATASETS["creditcard.csv"])
    best_model_kpi = html.escape(best_overall["model"]) if best_overall["model"] else "&mdash;"
    best_auc_kpi = f"{best_overall['roc_auc']:.4f}" if best_overall["roc_auc"] >= 0 else "&mdash;"

    galleries = {k: gallery(v, f"No figures available for this section ({k}).") for k, v in SECTION_FIGURES.items()}

    return TEMPLATE.format(
        today=today,
        kpi_obj=kpi("E-commerce + Banking", "Business Objective"),
        kpi_size=kpi(f"{fr_rows:,} + {cc_rows:,}", "Total Rows (raw)"),
        kpi_model=kpi(best_model_kpi, "Best Model"),
        kpi_auc=kpi(best_auc_kpi, "Best ROC-AUC"),
        kpi_rec=kpi("Risk-based review", "Main Recommendation"),
        best_model=best_model_kpi,
        best_auc=best_auc_kpi,
        best_dataset=html.escape(best_overall["dataset"]),
        dataset_table=dataset_table,
        metrics_sections="".join(metrics_sections),
        eda=galleries["eda"],
        geolocation=galleries["geolocation"],
        features=galleries["features"],
        imbalance=galleries["imbalance"],
        modeling=galleries["modeling"],
        shap=galleries["shap"],
    )


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Fraud Detection for E-Commerce and Banking Transactions</title>
<style>
  :root {{
    --ink:#1a1a1a;--muted:#6b7280;--accent:#2b50c8;--accent-soft:#eef2ff;
    --border:#e5e7eb;--card:#f9fafb;--shadow:0 8px 24px rgba(0,0,0,.08);--maxw:860px;
  }}
  *{{box-sizing:border-box;}} html{{scroll-behavior:smooth;}}
  body{{margin:0;background:#fff;color:var(--ink);font-family:Georgia,"Times New Roman",serif;font-size:18px;line-height:1.7;}}
  h1,h2,h3,.nav,.kpi-label,.badge,th,.toggle{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}}
  h1{{font-size:2.4rem;line-height:1.2;margin:0 0 .4rem;}}
  h2{{font-size:1.7rem;margin:2.2rem 0 .8rem;}}
  h3{{font-size:1.2rem;margin:1.3rem 0 .5rem;}}
  p{{margin:0 0 1rem;}} a{{color:var(--accent);text-decoration:none;}} a:hover{{text-decoration:underline;}}
  .muted{{color:var(--muted);}}
  .nav{{position:sticky;top:0;z-index:50;background:rgba(255,255,255,.92);backdrop-filter:blur(8px);
    border-bottom:1px solid var(--border);display:flex;gap:1rem;padding:.6rem 1.2rem;overflow-x:auto;font-size:.82rem;}}
  .nav strong{{color:var(--accent);white-space:nowrap;}} .nav a{{color:var(--ink);white-space:nowrap;opacity:.8;}} .nav a:hover{{opacity:1;}}
  .wrap{{max-width:var(--maxw);margin:0 auto;padding:0 1.2rem 5rem;}}
  .title-page{{padding:3.5rem 0 2rem;border-bottom:1px solid var(--border);}}
  .subtitle{{font-size:1.2rem;color:var(--muted);font-style:italic;margin-bottom:1.5rem;}}
  .meta{{font-size:.95rem;color:var(--muted);}} .meta b{{color:var(--ink);}}
  .exec-card{{background:var(--accent-soft);border:1px solid #dbe4ff;border-radius:14px;padding:1.4rem 1.6rem;box-shadow:var(--shadow);margin:2rem 0;}}
  .kpi-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1rem;margin:1rem 0;}}
  .kpi{{background:#fff;border:1px solid var(--border);border-radius:12px;padding:1rem;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,.04);}}
  .kpi-value{{font-size:1.35rem;font-weight:700;color:var(--accent);font-family:-apple-system,"Segoe UI",sans-serif;word-break:break-word;}}
  .kpi-label{{font-size:.74rem;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin-top:.3rem;}}
  .insight{{background:var(--card);border-left:4px solid var(--accent);border-radius:0 10px 10px 0;padding:1rem 1.2rem;margin:1.2rem 0;}}
  .placeholder{{border:2px dashed #cbd5e1;border-radius:12px;padding:1.2rem;text-align:center;color:var(--muted);background:#fafafa;margin:1rem 0;}}
  .gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:1.2rem;margin:1.2rem 0;}}
  .fig{{background:#fff;border:1px solid var(--border);border-radius:12px;overflow:hidden;box-shadow:var(--shadow);margin:0;}}
  .fig img{{width:100%;height:auto;display:block;background:#f3f4f6;}}
  .fig figcaption{{padding:.7rem .9rem;font-size:.82rem;color:var(--muted);border-top:1px solid var(--border);}}
  .fig .fignum{{color:var(--accent);font-weight:700;}}
  table{{width:100%;border-collapse:collapse;margin:1rem 0;font-size:.9rem;}}
  th,td{{border:1px solid var(--border);padding:.55rem .7rem;text-align:left;}}
  th{{background:var(--accent-soft);}} tbody tr:nth-child(even){{background:#fafafa;}}
  tr.highlight{{background:#ecfdf5 !important;}}
  .badge{{display:inline-block;background:#15803d;color:#fff;border-radius:10px;padding:1px 8px;font-size:.7rem;font-weight:700;}}
  .section-head{{display:flex;align-items:center;justify-content:space-between;cursor:pointer;}}
  .toggle{{font-size:.72rem;color:var(--accent);border:1px solid var(--accent);border-radius:20px;padding:2px 10px;user-select:none;}}
  .collapsed .section-body{{display:none;}}
  #toTop{{position:fixed;right:22px;bottom:22px;z-index:60;background:var(--accent);color:#fff;border:none;border-radius:50%;
    width:46px;height:46px;font-size:1.2rem;cursor:pointer;box-shadow:var(--shadow);display:none;font-family:-apple-system,sans-serif;}}
  @media print {{
    .nav,#toTop,.toggle{{display:none !important;}}
    .collapsed .section-body{{display:block !important;}}
    body{{font-size:11.5pt;}} .wrap{{max-width:none;}}
    section{{page-break-before:always;break-before:page;}}
    section:first-of-type,.title-page{{page-break-before:avoid;}}
    .fig,.exec-card,table,.insight{{page-break-inside:avoid;break-inside:avoid;}}
    a{{color:var(--ink);}}
  }}
  @page {{ margin:16mm; }}
</style>
</head>
<body>
<nav class="nav">
  <strong>Fraud Detection</strong>
  <a href="#sec1">1·Summary</a><a href="#sec2">2·Business</a><a href="#sec3">3·Dataset</a>
  <a href="#sec4">4·Cleaning</a><a href="#sec5">5·EDA</a><a href="#sec6">6·Geo</a>
  <a href="#sec7">7·Features</a><a href="#sec8">8·Imbalance</a><a href="#sec9">9·Models</a>
  <a href="#sec10">10·Evaluation</a><a href="#sec11">11·SHAP</a><a href="#sec12">12·Findings</a>
  <a href="#sec13">13·Recommendations</a><a href="#sec14">14·Conclusion</a>
</nav>
<div class="wrap">
  <header class="title-page">
    <h1>Fraud Detection for E-Commerce and Banking Transactions</h1>
    <div class="subtitle">End-to-End Machine Learning Pipeline for Fraud Risk Identification</div>
    <div class="meta"><p><b>Author:</b> [Insert Student Name]<br/><b>Date:</b> {today}</p></div>
    <div class="exec-card">
      <h3 style="margin-top:0">Executive Summary</h3>
      <div class="kpi-grid">
        {kpi_obj}
        {kpi_size}
        {kpi_model}
        {kpi_auc}
        {kpi_rec}
      </div>
      <p class="muted" style="font-size:.85rem;margin:.5rem 0 0">Best model and ROC-AUC are read
      directly from the project's metrics files; dataset sizes are computed from the raw CSVs.</p>
    </div>
  </header>

  <section id="sec1"><div class="section-head"><h2>1. Executive Summary</h2><span class="toggle">collapse</span></div>
  <div class="section-body">
    <p>This report documents an end-to-end machine-learning pipeline for detecting fraudulent
    transactions across an e-commerce dataset (<code>Fraud_Data.csv</code>) and a banking
    dataset (<code>creditcard.csv</code>). The pipeline covers data cleaning, geolocation
    enrichment, feature engineering, class-imbalance handling, model development, evaluation,
    and explainability (SHAP).</p>
    <div class="insight">The strongest model by ROC-AUC was <b>{best_model}</b> on the
    <b>{best_dataset}</b> dataset, reaching a ROC-AUC of <b>{best_auc}</b> (see Section 9).</div>
  </div></section>

  <section id="sec2"><div class="section-head"><h2>2. Business Understanding</h2><span class="toggle">collapse</span></div>
  <div class="section-body">
    <h3>Problem Statement</h3>
    <p>Fraudulent transactions cause direct financial loss and erode customer trust. The goal is
    to flag high-risk transactions accurately while minimizing false positives that disrupt
    legitimate customers.</p>
    <h3>Why Fraud Detection Matters</h3>
    <p>Effective detection reduces chargebacks and losses, protects customers, and supports
    regulatory compliance. The central challenge is the <b>precision-recall trade-off</b>:
    catching more fraud (recall) without overwhelming operations with false alarms (precision).</p>
    <h3>Business Impact</h3>
    <p>A well-calibrated model lets the business prioritise genuinely risky transactions for
    review, balancing loss prevention against customer experience.</p>
  </div></section>

  <section id="sec3"><div class="section-head"><h2>3. Dataset Overview</h2><span class="toggle">collapse</span></div>
  <div class="section-body">
    <p>Two datasets are used. Row and column counts below are computed directly from the raw files.</p>
    {dataset_table}
  </div></section>

  <section id="sec4"><div class="section-head"><h2>4. Data Cleaning and Preprocessing</h2><span class="toggle">collapse</span></div>
  <div class="section-body">
    <p>Cleaning steps included handling missing values, removing duplicate transactions, and
    correcting data types (e.g., parsing timestamps to datetime and IP addresses to integers).
    Cleaned datasets are stored under <code>data/processed/</code>.</p>
  </div></section>

  <section id="sec5"><div class="section-head"><h2>5. Exploratory Data Analysis</h2><span class="toggle">collapse</span></div>
  <div class="section-body">
    <p>Figures discovered from the project's figure outputs:</p>
    {eda}
  </div></section>

  <section id="sec6"><div class="section-head"><h2>6. Geolocation Enrichment</h2><span class="toggle">collapse</span></div>
  <div class="section-body">
    <h3>IP-to-Country Mapping</h3>
    <p>Transaction IP addresses are converted to integer form and matched against IP-range blocks
    (<code>IpAddress_to_Country.csv</code>) to assign a country of origin.</p>
    <h3>Merge Strategy</h3>
    <p>Each transaction IP is joined to the IP-range lookup using a lower/upper bound range match
    to attach the country attribute.</p>
    <h3>Business Value</h3>
    <p>Country-level signals surface geographic risk concentrations and origin mismatches.</p>
    {geolocation}
  </div></section>

  <section id="sec7"><div class="section-head"><h2>7. Feature Engineering</h2><span class="toggle">collapse</span></div>
  <div class="section-body">
    <ul>
      <li><b>time_since_signup</b> &mdash; elapsed time between signup and transaction; very short gaps can indicate scripted fraud.</li>
      <li><b>hour_of_day</b> &mdash; hour the transaction occurred (time-of-day risk patterns).</li>
      <li><b>day_of_week</b> &mdash; weekday of the transaction (weekly seasonality).</li>
      <li><b>transaction velocity</b> &mdash; rate of transactions over a time window per user/device.</li>
      <li><b>transaction frequency</b> &mdash; count of transactions per entity (abnormal bursts).</li>
    </ul>
    {features}
  </div></section>

  <section id="sec8"><div class="section-head"><h2>8. Class Imbalance Handling</h2><span class="toggle">collapse</span></div>
  <div class="section-body">
    <h3>Original Distribution</h3>
    <p>Fraud is rare relative to legitimate activity (see the class-distribution figures in EDA).</p>
    <h3>SMOTE Strategy</h3>
    <p>SMOTE (Synthetic Minority Over-sampling Technique) synthesises minority-class examples by
    interpolating between existing fraud samples, applied to the training split only to avoid leakage.</p>
    <h3>Why Balancing Was Needed</h3>
    <p>Without balancing, a model can score high accuracy by predicting "not fraud" almost always.
    Balancing focuses learning on the fraud signal and improves minority-class recall.</p>
    {imbalance}
  </div></section>

  <section id="sec9"><div class="section-head"><h2>9. Model Development</h2><span class="toggle">collapse</span></div>
  <div class="section-body">
    <p>Models evaluated include Logistic Regression, Random Forest, and XGBoost. The comparison
    tables below are generated directly from the project's metrics files; the best model per
    dataset (by ROC-AUC) is highlighted.</p>
    {metrics_sections}
  </div></section>

  <section id="sec10"><div class="section-head"><h2>10. Model Evaluation</h2><span class="toggle">collapse</span></div>
  <div class="section-body">
    <p>Evaluation uses Accuracy, Precision, Recall, F1, and ROC-AUC (see Section 9 for values).
    Confusion matrices, ROC/PR curves, and feature-importance plots discovered from the outputs:</p>
    {modeling}
  </div></section>

  <section id="sec11"><div class="section-head"><h2>11. Explainable AI (SHAP)</h2><span class="toggle">collapse</span></div>
  <div class="section-body">
    <p>SHAP (SHapley Additive exPlanations) attributes each prediction to its input features,
    enabling global importance views and per-transaction explanations.</p>
    {shap}
  </div></section>

  <section id="sec12"><div class="section-head"><h2>12. Key Findings</h2><span class="toggle">collapse</span></div>
  <div class="section-body">
    <ul>
      <li>The best-performing model by ROC-AUC was <b>{best_model}</b> on <b>{best_dataset}</b> (ROC-AUC <b>{best_auc}</b>).</li>
      <li>Model behaviour and the most influential features are visualised in the SHAP and
      feature-importance figures (Sections 10-11).</li>
      <li>Engineered temporal/behavioural features (e.g. <code>time_since_signup</code>) were used
      as fraud indicators; their relationship to the target is shown in Section 7.</li>
    </ul>
    <p class="muted" style="font-size:.85rem">Feature-level conclusions should be read from the
    SHAP/feature-importance plots above rather than asserted numerically here.</p>
  </div></section>

  <section id="sec13"><div class="section-head"><h2>13. Business Recommendations</h2><span class="toggle">collapse</span></div>
  <div class="section-body">
    <h3>Fraud Prevention</h3>
    <p>Deploy the best-performing model per channel for transaction risk scoring and route
    high-risk transactions to manual review.</p>
    <h3>Operational</h3>
    <p>Tune the decision threshold to the business's tolerance for false positives versus missed
    fraud, informed by the precision/recall figures in Section 9.</p>
    <h3>Monitoring</h3>
    <p>Track live precision/recall and data drift; schedule periodic retraining as fraud patterns evolve.</p>
    <h3>Future Improvements</h3>
    <p>Add a re-ranking/ensemble step, richer geolocation and device features, and threshold
    optimisation; expand explainability reporting for analysts.</p>
  </div></section>

  <section id="sec14"><div class="section-head"><h2>14. Conclusion</h2><span class="toggle">collapse</span></div>
  <div class="section-body">
    <p>The project delivered a complete fraud-detection pipeline from raw data to explainable
    models for both e-commerce and banking transactions. The strongest model by ROC-AUC was
    <b>{best_model}</b> on <b>{best_dataset}</b> (<b>{best_auc}</b>), providing a foundation for
    risk-based transaction review that balances loss prevention with customer experience.</p>
  </div></section>
</div>
<button id="toTop" title="Back to top" aria-label="Back to top">&uarr;</button>
<script>
  document.querySelectorAll('.section-head').forEach(function(h){{
    h.addEventListener('click',function(){{
      var s=h.closest('section'); s.classList.toggle('collapsed');
      var t=h.querySelector('.toggle'); if(t) t.textContent=s.classList.contains('collapsed')?'expand':'collapse';
    }});
  }});
  var b=document.getElementById('toTop');
  window.addEventListener('scroll',function(){{b.style.display=window.scrollY>500?'block':'none';}});
  b.addEventListener('click',function(){{window.scrollTo({{top:0,behavior:'smooth'}});}});
</script>
</body></html>"""


def main() -> None:
    html_doc = build()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print("Wrote", OUT)
    print("Size:", round(os.path.getsize(OUT) / 1024, 1), "KB")


if __name__ == "__main__":
    main()
