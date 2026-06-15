"""Build a self-contained HTML final report with embedded figures for PDF export."""

from __future__ import annotations

import base64
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
FIGURES_DIR = DOCS_DIR / "figures"
REPORTS_FIGURES = PROJECT_ROOT / "reports" / "figures"
REPORTS_PLOTS = PROJECT_ROOT / "reports" / "outputs" / "plots"
REPORTS_CONFUSION = PROJECT_ROOT / "reports" / "outputs" / "confusion_matrices"
MARKDOWN_PATH = DOCS_DIR / "final-report.md"
OUTPUT_PATH = DOCS_DIR / "final-report.html"


def _resolve_image_path(relative: str) -> Path | None:
    """Resolve markdown image paths relative to docs/final-report.md."""
    rel = relative.replace("\\", "/")
    candidates = [
        DOCS_DIR / rel,
        PROJECT_ROOT / rel.lstrip("./"),
        DOCS_DIR / "figures" / Path(rel).name,
    ]
    if rel.startswith("../reports/figures/"):
        candidates.insert(0, REPORTS_FIGURES / Path(rel).name)
    if rel.startswith("../reports/outputs/plots/"):
        candidates.insert(0, REPORTS_PLOTS / Path(rel).name)
    if rel.startswith("../reports/outputs/confusion_matrices/"):
        candidates.insert(0, REPORTS_CONFUSION / Path(rel).name)
    for path in candidates:
        if path.exists():
            return path
    return None


def embed_image(relative: str, alt: str) -> str:
    path = _resolve_image_path(relative)
    if path is None:
        return f'<p><em>Figure missing: {relative}</em></p>'
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        f'<figure><img src="data:image/png;base64,{encoded}" alt="{alt}">'
        f'<figcaption>{alt}</figcaption></figure>'
    )


def _inline_format(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return text


def _parse_table(lines: list[str]) -> tuple[str, int]:
    rows: list[str] = []
    i = 0
    while i < len(lines) and lines[i].strip().startswith("|"):
        row = lines[i].strip()
        if not re.match(r"^\|[-| :]+\|$", row):
            cells = [_inline_format(c.strip()) for c in row.strip("|").split("|")]
            tag = "th" if not rows else "td"
            rows.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>")
        i += 1
    return "<table>" + "".join(rows) + "</table>", i


def markdown_to_html(md: str) -> str:
    lines = md.splitlines()
    html_parts: list[str] = []
    i = 0
    in_code = False
    code_buffer: list[str] = []

    while i < len(lines):
        line = lines[i]

        if line.strip().startswith("```"):
            if in_code:
                html_parts.append(
                    "<pre><code>" + "\n".join(code_buffer) + "</code></pre>"
                )
                code_buffer = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue

        if in_code:
            code_buffer.append(line.replace("<", "&lt;").replace(">", "&gt;"))
            i += 1
            continue

        img_match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line.strip())
        if img_match:
            alt, src = img_match.groups()
            html_parts.append(embed_image(src, alt))
            i += 1
            if i < len(lines) and lines[i].strip().startswith("*"):
                html_parts.append(f"<p class='caption'>{_inline_format(lines[i].strip())}</p>")
                i += 1
            continue

        if line.strip().startswith("|"):
            table_html, consumed = _parse_table(lines[i:])
            html_parts.append(table_html)
            i += consumed
            continue

        if line.startswith("# "):
            html_parts.append(f"<h1>{_inline_format(line[2:].strip())}</h1>")
        elif line.startswith("## "):
            html_parts.append(f"<h2>{_inline_format(line[3:].strip())}</h2>")
        elif line.startswith("### "):
            html_parts.append(f"<h3>{_inline_format(line[4:].strip())}</h3>")
        elif line.strip() == "---":
            html_parts.append("<hr>")
        elif line.strip().startswith("- "):
            items = [line.strip()[2:]]
            i += 1
            while i < len(lines) and lines[i].strip().startswith("- "):
                items.append(lines[i].strip()[2:])
                i += 1
            html_parts.append(
                "<ul>" + "".join(f"<li>{_inline_format(item)}</li>" for item in items) + "</ul>"
            )
            continue
        elif line.strip():
            html_parts.append(f"<p>{_inline_format(line.strip())}</p>")

        i += 1

    return "\n".join(html_parts)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Fraud Detection Analytics — Final Report</title>
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
      max-width: 720px;
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
      font-size: 1.4rem;
      margin-top: 44px;
      margin-bottom: 14px;
      padding-top: 8px;
      border-top: 1px solid var(--border);
    }}
    h3 {{
      font-size: 1.1rem;
      margin-top: 24px;
      margin-bottom: 10px;
    }}
    p, li {{
      font-size: 1.05rem;
      margin: 0 0 14px;
    }}
    p.caption {{
      font-size: 0.9rem;
      color: var(--muted);
      font-style: italic;
      margin-top: -8px;
      margin-bottom: 24px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 16px 0 24px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 0.92rem;
    }}
    th, td {{
      border: 1px solid var(--border);
      padding: 8px 12px;
      text-align: left;
    }}
    th {{
      background: #f8fafc;
      font-weight: 600;
    }}
    figure {{
      margin: 28px 0;
      text-align: center;
    }}
    figure img {{
      max-width: 100%;
      height: auto;
      border: 1px solid var(--border);
      border-radius: 4px;
    }}
    figcaption {{
      font-size: 0.9rem;
      color: var(--muted);
      margin-top: 8px;
      font-style: italic;
    }}
    code {{
      font-family: "SF Mono", Consolas, monospace;
      font-size: 0.88em;
      background: var(--code-bg);
      padding: 2px 6px;
      border-radius: 3px;
    }}
    pre {{
      background: var(--code-bg);
      padding: 16px;
      border-radius: 6px;
      overflow-x: auto;
      font-size: 0.85rem;
      line-height: 1.5;
    }}
    pre code {{
      background: none;
      padding: 0;
    }}
    hr {{
      border: none;
      border-top: 1px solid var(--border);
      margin: 32px 0;
    }}
    @media print {{
      body {{ font-size: 11pt; }}
      .container {{ max-width: 100%; padding: 0; }}
      h2 {{ page-break-before: auto; }}
      figure {{ page-break-inside: avoid; }}
    }}
  </style>
</head>
<body>
  <div class="container">
    {body}
  </div>
</body>
</html>
"""


def main() -> None:
    md = MARKDOWN_PATH.read_text(encoding="utf-8")
    body = markdown_to_html(md)
    OUTPUT_PATH.write_text(HTML_TEMPLATE.format(body=body), encoding="utf-8")
    print(f"Final report written to: {OUTPUT_PATH}")
    print("Open in a browser and use Print -> Save as PDF to export.")


if __name__ == "__main__":
    main()
