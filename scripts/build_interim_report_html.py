"""Build a self-contained HTML interim report with embedded figures."""

import base64
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "week-5-6-interim-report.html"


def embed_image(filename: str, alt: str) -> str:
    path = FIGURES_DIR / filename
    if not path.exists():
        return f'<p><em>Figure missing: {filename}</em></p>'
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        f'<figure><img src="data:image/png;base64,{encoded}" alt="{alt}">'
        f'<figcaption>{alt}</figcaption></figure>'
    )


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Fraud Detection Analytics — Week 5–6 Interim Report</title>
  <style>
    :root {{
      --text: #1a1a1a;
      --muted: #5c5c5c;
      --border: #e6e6e6;
      --accent: #0f766e;
      --bg: #ffffff;
      --code-bg: #f6f8fa;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: Georgia, "Times New Roman", serif;
      color: var(--text);
      background: var(--bg);
      line-height: 1.7;
      margin: 0;
      padding: 0;
    }}
    .container {{
      max-width: 760px;
      margin: 0 auto;
      padding: 48px 24px 80px;
    }}
    header {{
      border-bottom: 1px solid var(--border);
      margin-bottom: 40px;
      padding-bottom: 24px;
    }}
    h1 {{
      font-size: 2rem;
      line-height: 1.25;
      margin: 0 0 16px;
      font-weight: 700;
    }}
    .meta {{
      color: var(--muted);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 0.95rem;
    }}
    h2 {{
      font-size: 1.45rem;
      margin-top: 48px;
      margin-bottom: 16px;
      padding-top: 8px;
      border-top: 1px solid var(--border);
    }}
    h3 {{
      font-size: 1.15rem;
      margin-top: 28px;
      margin-bottom: 12px;
    }}
    p, li {{
      font-size: 1.05rem;
    }}
    ul, ol {{
      padding-left: 1.4rem;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 20px 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 0.92rem;
    }}
    th, td {{
      border: 1px solid var(--border);
      padding: 10px 12px;
      text-align: left;
    }}
    th {{
      background: #f9fafb;
      font-weight: 600;
    }}
    figure {{
      margin: 28px 0;
      text-align: center;
    }}
    img {{
      max-width: 100%;
      height: auto;
      border: 1px solid var(--border);
      border-radius: 4px;
    }}
    figcaption {{
      color: var(--muted);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 0.9rem;
      margin-top: 8px;
    }}
    code {{
      font-family: Consolas, Monaco, monospace;
      background: var(--code-bg);
      padding: 2px 6px;
      border-radius: 3px;
      font-size: 0.9em;
    }}
    pre {{
      background: var(--code-bg);
      padding: 16px;
      overflow-x: auto;
      border-radius: 6px;
      font-size: 0.85rem;
      line-height: 1.5;
      border: 1px solid var(--border);
    }}
    .callout {{
      border-left: 4px solid var(--accent);
      padding: 12px 16px;
      margin: 24px 0;
      background: #f0fdfa;
      font-size: 0.98rem;
    }}
    footer {{
      margin-top: 48px;
      padding-top: 24px;
      border-top: 1px solid var(--border);
      color: var(--muted);
      font-size: 0.92rem;
      font-style: italic;
    }}
    strong {{ font-weight: 700; }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Fraud Detection Analytics — Week 5–6 Interim Report (Task 1)</h1>
      <div class="meta">
        <div><strong>Project:</strong> fraud-detection-analytics</div>
        <div><strong>Scope:</strong> Data understanding, preprocessing, EDA, geolocation enrichment, feature engineering, and class imbalance handling</div>
        <div><strong>Date:</strong> June 2026</div>
      </div>
    </header>

    <h2>1. Project Overview</h2>
    <p>This interim deliverable establishes the foundation for a fraud detection analytics pipeline using two complementary datasets: e-commerce transaction logs (<code>Fraud_Data.csv</code>) and credit card transactions (<code>creditcard.csv</code>). A third reference table (<code>IpAddress_to_Country.csv</code>) supports geolocation enrichment.</p>
    <p>Task 1 focuses on making the work <strong>reproducible and modular</strong>. The repository now includes:</p>
    <ul>
      <li>A structured Python package under <code>src/</code> for loading, cleaning, enrichment, and feature engineering</li>
      <li>An EDA notebook for <code>Fraud_Data</code> (<code>notebooks/eda-fraud-data.ipynb</code>)</li>
      <li>CLI scripts for preprocessing, geolocation, feature engineering, and imbalance handling</li>
      <li>A GitHub Actions workflow that installs dependencies and runs <strong>43 pytest tests</strong></li>
      <li>Processed outputs saved to <code>data/processed/</code> and diagnostic reports to <code>reports/</code></li>
    </ul>
    <p>The immediate goal is not final model deployment, but a reliable path from raw data to model-ready features with documented data quality and class imbalance decisions.</p>

    <h2>2. Data Understanding</h2>
    <h3>Fraud_Data.csv (e-commerce)</h3>
    <table>
      <tr><th>Attribute</th><th>Value</th></tr>
      <tr><td>Rows</td><td>151,112</td></tr>
      <tr><td>Columns</td><td>11</td></tr>
      <tr><td>Target</td><td><code>class</code> (0 = legitimate, 1 = fraud)</td></tr>
      <tr><td>Key fields</td><td><code>signup_time</code>, <code>purchase_time</code>, <code>purchase_value</code>, <code>device_id</code>, <code>source</code>, <code>browser</code>, <code>sex</code>, <code>age</code>, <code>ip_address</code></td></tr>
    </table>
    <p>After loading, the raw file contains <strong>no missing values</strong> and <strong>no duplicate rows</strong>. Fraud accounts for <strong>9.36%</strong> of transactions (14,151 fraud / 136,961 legitimate), yielding an imbalance ratio of approximately <strong>9.7:1</strong>.</p>
    <p>Each <code>user_id</code> appears exactly once in this dataset, meaning every row represents a single user transaction.</p>

    <h3>creditcard.csv (PCA-transformed features)</h3>
    <table>
      <tr><th>Attribute</th><th>Value</th></tr>
      <tr><td>Raw rows</td><td>284,807</td></tr>
      <tr><td>Columns</td><td>31 (<code>Time</code>, <code>V1</code>–<code>V28</code>, <code>Amount</code>, <code>Class</code>)</td></tr>
      <tr><td>Clean rows (after preprocessing)</td><td>283,726</td></tr>
    </table>
    <p>This dataset is <strong>severely imbalanced</strong>: only <strong>0.17%</strong> of transactions are fraudulent (473 fraud / 283,253 legitimate), with a ratio near <strong>599:1</strong>. Features are already PCA components, so direct feature interpretation is limited. <code>Time</code> records seconds elapsed since the first transaction in the dataset.</p>

    <h3>IpAddress_to_Country.csv</h3>
    <table>
      <tr><th>Attribute</th><th>Value</th></tr>
      <tr><td>Rows</td><td>138,846</td></tr>
      <tr><td>Columns</td><td><code>lower_bound_ip_address</code>, <code>upper_bound_ip_address</code>, <code>country</code></td></tr>
    </table>
    <p>This file maps IPv4 ranges to country names and is used for range-based IP enrichment on <code>Fraud_Data</code>.</p>

    <h2>3. Cleaning and Preprocessing</h2>
    <p>Preprocessing is implemented in reusable modules (<code>src/preprocessing/</code>) and applied consistently across scripts and notebooks.</p>
    <p><strong>Fraud_Data pipeline</strong></p>
    <ol>
      <li>Standardize column names to lowercase snake_case</li>
      <li>Parse <code>signup_time</code> and <code>purchase_time</code> as datetimes</li>
      <li>Remove duplicate rows (none found in raw data)</li>
      <li>Handle missing values (none found; pipeline still enforces required columns)</li>
      <li>Coerce numeric and string dtypes for modeling</li>
    </ol>
    <p><strong>creditcard.csv pipeline</strong></p>
    <ol>
      <li>Standardize column names</li>
      <li>Remove <strong>1,081 duplicate rows</strong></li>
      <li>Cast <code>class</code>, <code>amount</code>, <code>time</code>, and PCA features to numeric types</li>
    </ol>
    <p>All cleaning steps log actions to stdout, which supports auditability during notebook and script runs. Unit tests verify missing-value handling, duplicate removal, and timestamp conversion on synthetic data.</p>

    <h2>4. EDA Findings</h2>
    <h3>Fraud_Data</h3>
    <p>Exploratory analysis in <code>notebooks/eda-fraud-data.ipynb</code> highlights behavioral and channel-level patterns.</p>
    <p><strong>Class imbalance.</strong> Fraud is the minority class but not negligible at ~9.4%. Accuracy alone would be a poor evaluation metric; precision, recall, F1, and PR-AUC are more appropriate.</p>
    {fig_fraud_class}
    <p><strong>Signup-to-purchase timing is the strongest signal.</strong> Legitimate users have a median time from signup to purchase of approximately <strong>1,443 hours</strong> (~60 days). Fraudulent users have a median near <strong>0 hours</strong> — essentially immediate purchases after signup. This motivates <code>time_since_signup_hours</code> as a primary engineered feature.</p>
    {fig_time_signup}
    <p><em>Note: The boxplot clips at the 95th percentile for readability; fraud remains near zero even without clipping.</em></p>
    <p><strong>Channel and demographic patterns (modest but useful)</strong></p>
    <table>
      <tr><th>Segment</th><th>Fraud rate</th></tr>
      <tr><td>Direct traffic</td><td>10.54%</td></tr>
      <tr><td>Ads</td><td>9.21%</td></tr>
      <tr><td>SEO</td><td>8.93%</td></tr>
      <tr><td>Chrome</td><td>9.88%</td></tr>
      <tr><td>Firefox</td><td>9.52%</td></tr>
      <tr><td>Safari</td><td>9.02%</td></tr>
      <tr><td>Male</td><td>9.55%</td></tr>
      <tr><td>Female</td><td>9.10%</td></tr>
    </table>
    <p>Median <code>purchase_value</code> (~35) and median <code>age</code> (~33) are similar across classes, so amount and age alone do not separate fraud well.</p>

    <h3>creditcard.csv</h3>
    <p>Preprocessing and descriptive review show a fundamentally different problem shape:</p>
    <table>
      <tr><th>Metric</th><th>Legitimate (0)</th><th>Fraud (1)</th></tr>
      <tr><td>Share of rows</td><td>99.83%</td><td>0.17%</td></tr>
      <tr><td>Median amount</td><td>€22.00</td><td>€9.82</td></tr>
      <tr><td>Median time (seconds)</td><td>84,711</td><td>73,408</td></tr>
    </table>
    <p>Fraud transactions tend to be slightly lower in amount and occur earlier in the dataset timeline. The extreme rarity of fraud cases (under 500 rows) will require careful resampling and metric selection in later modeling work.</p>
    {fig_creditcard}

    <h2>5. Geolocation Enrichment</h2>
    <p>IP addresses in <code>Fraud_Data</code> are stored as numeric values. The enrichment pipeline (<code>src/preprocessing/geolocation.py</code>):</p>
    <ol>
      <li>Converts IPs to 32-bit integers (<code>ip_address_int</code>)</li>
      <li>Performs a <strong>range-based lookup</strong> against <code>IpAddress_to_Country.csv</code> using <code>pd.merge_asof</code></li>
      <li>Validates that each IP falls within the matched range upper bound</li>
      <li>Assigns <code>country</code>, or <code>unknown</code> when no range matches</li>
    </ol>
    <table>
      <tr><th>Metric</th><th>Value</th></tr>
      <tr><td>Transactions enriched</td><td>151,112</td></tr>
      <tr><td>IPs matched to a country</td><td>129,146 (85.46%)</td></tr>
      <tr><td>Unmatched (unknown)</td><td>21,966</td></tr>
      <tr><td>Distinct countries</td><td>182</td></tr>
    </table>
    <p>Among countries with at least 100 transactions, higher observed fraud rates included Ecuador (26.4%), Tunisia (26.3%), and Peru (26.1%). These segment-level patterns support including <code>country</code> as a categorical feature, while recognizing that high rates in smaller geographies may reflect limited sample size.</p>
    <p>Output saved to: <code>data/processed/fraud_data_geolocated.csv</code></p>

    <h2>6. Feature Engineering</h2>
    <p>Feature engineering is implemented in <code>src/features/</code> and executed via <code>scripts/run_feature_engineering.py</code>.</p>
    <h3>Temporal features</h3>
    <table>
      <tr><th>Feature</th><th>Description</th></tr>
      <tr><td><code>time_since_signup_hours</code></td><td>Hours between <code>signup_time</code> and <code>purchase_time</code></td></tr>
      <tr><td><code>hour_of_day</code></td><td>Purchase hour (0–23)</td></tr>
      <tr><td><code>day_of_week</code></td><td>Purchase weekday (0 = Monday)</td></tr>
    </table>
    <div class="callout"><code>time_since_signup_hours</code> is the most important engineered feature given EDA: median <strong>0.0003 h</strong> for fraud vs <strong>1,443 h</strong> for legitimate users in the engineered dataset.</div>
    <h3>Velocity features</h3>
    <p>The pipeline also computes per-user rolling transaction counts (<code>txn_count_last_1h</code>, <code>txn_count_last_24h</code>, <code>txn_count_last_168h</code>), <code>hours_since_last_txn</code>, <code>user_cumulative_txn_count</code>, and <code>user_txn_velocity_per_day</code>.</p>
    <p>Because each <code>user_id</code> in <code>Fraud_Data</code> has only one transaction, these velocity features are uniformly <strong>1</strong> (or <strong>0</strong> hours since last transaction) in the current dataset. The code remains valuable for datasets with repeat purchasers.</p>
    <h3>Encoding and scaling</h3>
    <ul>
      <li>Numeric features are standardized with <code>StandardScaler</code></li>
      <li>Categorical features (<code>source</code>, <code>browser</code>, <code>sex</code>, <code>country</code>) are one-hot encoded</li>
      <li>High-cardinality <code>device_id</code> is excluded from one-hot encoding to avoid sparse, unstable columns</li>
    </ul>
    <table>
      <tr><th>File</th><th>Shape / size</th></tr>
      <tr><td><code>data/processed/fraud_data_engineered.csv</code></td><td>151,112 rows × 22 columns</td></tr>
      <tr><td><code>data/processed/fraud_data_features.csv</code></td><td>151,112 rows × 204 columns (203 features + class)</td></tr>
    </table>

    <h2>7. Class Imbalance Handling</h2>
    <p>Fraud detection models trained on raw class proportions tend to favor the majority class. We address this with <strong>SMOTE on the training split only</strong>, implemented in <code>src/modeling/imbalance.py</code>.</p>
    <p><strong>Why SMOTE over undersampling</strong></p>
    <ul>
      <li><code>Fraud_Data</code> has ~9.4% fraud — imbalanced, but not so rare that majority rows should be discarded</li>
      <li>SMOTE keeps all 109,568 legitimate training rows and synthesizes additional fraud examples</li>
      <li>Random undersampling would remove most legitimate transactions and waste information</li>
    </ul>
    <p><strong>Safeguards</strong></p>
    <ol>
      <li>Stratified <code>train_test_split</code> (80/20) before any resampling</li>
      <li>SMOTE applied only to <code>(X_train, y_train)</code></li>
      <li>Test set left untouched to preserve real-world prevalence</li>
    </ol>
    <table>
      <tr><th>Stage</th><th>Class 0</th><th>Class 1</th><th>Fraud %</th></tr>
      <tr><td>Train (before SMOTE)</td><td>109,568</td><td>11,321</td><td>9.36%</td></tr>
      <tr><td>Train (after SMOTE)</td><td>109,568</td><td>109,568</td><td>50.0%</td></tr>
      <tr><td>Test (holdout)</td><td>27,393</td><td>2,830</td><td>9.36%</td></tr>
    </table>
    {fig_smote}

    <h2>8. Challenges and Next Steps</h2>
    <h3>Challenges encountered</h3>
    <ol>
      <li><strong>Different imbalance profiles</strong> — <code>Fraud_Data</code> (~9% fraud) and <code>creditcard.csv</code> (~0.17% fraud) require dataset-specific resampling strategies.</li>
      <li><strong>IP geolocation coverage</strong> — 14.5% of IPs did not match any country range and were labeled <code>unknown</code>.</li>
      <li><strong>Single-transaction users</strong> — velocity features do not vary in the current <code>Fraud_Data</code>.</li>
      <li><strong>creditcard interpretability</strong> — PCA features limit direct business explanations.</li>
    </ol>
    <h3>Next steps (beyond Task 1)</h3>
    <ol>
      <li>Train baseline classifiers on the SMOTE-balanced training set</li>
      <li>Evaluate on the untouched test set using precision, recall, F1, and PR-AUC</li>
      <li>Extend preprocessing and feature engineering to <code>creditcard.csv</code></li>
      <li>Apply SHAP for interpretability on the best-performing model</li>
      <li>Compare SMOTE against class-weighted models</li>
    </ol>

    <h2>Appendix: Reproducibility</h2>
    <pre><code># Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Execute pipelines
python scripts/run_preprocess.py
python scripts/run_geolocation_enrichment.py
python scripts/run_feature_engineering.py
python scripts/run_imbalance_resampling.py</code></pre>
    <table>
      <tr><th>Artifact</th><th>Location</th></tr>
      <tr><td>EDA notebook</td><td><code>notebooks/eda-fraud-data.ipynb</code></td></tr>
      <tr><td>Geolocated data</td><td><code>data/processed/fraud_data_geolocated.csv</code></td></tr>
      <tr><td>Engineered features</td><td><code>data/processed/fraud_data_engineered.csv</code></td></tr>
      <tr><td>Model-ready matrix</td><td><code>data/processed/fraud_data_features.csv</code></td></tr>
      <tr><td>Imbalance report</td><td><code>reports/class_imbalance_summary.md</code></td></tr>
    </table>

    <footer>
      This report reflects outputs produced by the project codebase as of the interim submission.
      No modeling results are included, as Task 1 focuses on data preparation and analysis.
    </footer>
  </div>
</body>
</html>
"""


def main() -> None:
    html = HTML.format(
        fig_fraud_class=embed_image(
            "fraud_class_distribution.png", "Fraud_Data class distribution"
        ),
        fig_time_signup=embed_image(
            "time_since_signup_by_class.png",
            "Signup-to-purchase time by class (95th percentile clip)",
        ),
        fig_creditcard=embed_image(
            "creditcard_class_distribution.png",
            "Credit card fraud class distribution (log scale)",
        ),
        fig_smote=embed_image(
            "smote_class_distribution.png",
            "Training set class distribution before and after SMOTE",
        ),
    )
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
