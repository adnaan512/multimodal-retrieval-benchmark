"""
Generate a self-contained dark-themed HTML report summarizing a full
benchmark run: recall metrics, direction comparison, per-category
breakdown, failure examples, and hard-negative reranking impact.
"""
from __future__ import annotations

import html
import os
from typing import Dict, List, Optional

from src.models import FailureCase, RecallMetrics

_CSS = """
:root {
  --bg: #0d1117; --panel: #161b22; --border: #30363d;
  --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
  --good: #3fb950; --bad: #f85149; --warn: #d29922;
}
* { box-sizing: border-box; }
body {
  background: var(--bg); color: var(--text); margin: 0; padding: 32px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
}
h1 { font-size: 26px; margin-bottom: 4px; }
h2 { font-size: 18px; margin-top: 40px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }
.subtitle { color: var(--muted); margin-bottom: 24px; }
.cards { display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0; }
.card {
  background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
  padding: 18px 22px; min-width: 140px;
}
.card .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
.card .value { font-size: 28px; font-weight: 600; color: var(--accent); margin-top: 6px; }
table { width: 100%; border-collapse: collapse; margin: 12px 0 8px 0; font-size: 14px; }
th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--border); }
th { color: var(--muted); font-weight: 500; text-transform: uppercase; font-size: 11px; letter-spacing: .04em; }
tr:hover { background: #1c2129; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 6px; font-size: 12px; }
.badge.bad { background: rgba(248,81,73,0.15); color: var(--bad); }
.badge.warn { background: rgba(210,153,34,0.15); color: var(--warn); }
.finding {
  background: rgba(210,153,34,0.08); border: 1px solid var(--warn); border-radius: 8px;
  padding: 14px 18px; margin: 16px 0; color: var(--text);
}
.failure-case {
  background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
  padding: 14px 18px; margin-bottom: 10px;
}
.failure-case .qtext { color: var(--accent); }
.failure-case .meta { color: var(--muted); font-size: 12px; margin-top: 6px; }
footer { color: var(--muted); font-size: 12px; margin-top: 48px; border-top: 1px solid var(--border); padding-top: 16px; }
code { background: #010409; padding: 2px 6px; border-radius: 4px; font-size: 13px; }
"""


def _card(label: str, value: str) -> str:
    return f'<div class="card"><div class="label">{html.escape(label)}</div><div class="value">{html.escape(str(value))}</div></div>'


def _metrics_table(rows: List[dict]) -> str:
    if not rows:
        return "<p class='subtitle'>No metrics available.</p>"
    headers = list(rows[0].keys())
    thead = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    trs = []
    for row in rows:
        tds = "".join(f"<td>{html.escape(str(row[h]))}</td>" for h in headers)
        trs.append(f"<tr>{tds}</tr>")
    return f"<table><thead><tr>{thead}</tr></thead><tbody>{''.join(trs)}</tbody></table>"


def _category_table(per_category: Dict[str, dict]) -> str:
    rows = []
    for cat, stats in sorted(per_category.items(), key=lambda kv: kv[1].get("failure_rate", 1 - kv[1].get("r@1", 0)), reverse=True):
        fr = stats.get("failure_rate", 1 - stats.get("r@1", 0))
        n = stats.get("total", stats.get("n", 0))
        badge = ""
        if cat == "spatial_relation":
            badge = ' <span class="badge warn">hardest category</span>'
        elif fr > 0.4:
            badge = ' <span class="badge bad">high failure</span>'
        rows.append(f"<tr><td>{html.escape(cat)}{badge}</td><td>{n}</td><td>{fr*100:.1f}%</td></tr>")
    return (
        "<table><thead><tr><th>Query Category</th><th>N</th><th>Failure Rate</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _failure_case_html(case: FailureCase) -> str:
    query = case.query_text if case.query_text else f"[image #{case.query_image_index}]"
    return (
        '<div class="failure-case">'
        f'<div class="qtext">Query: {html.escape(str(query))}</div>'
        f'<div class="meta">category={html.escape(case.category)} &nbsp;|&nbsp; '
        f'top-1 retrieved index={case.top1_index} (score={case.top1_score:.3f}) &nbsp;|&nbsp; '
        f'ground truth index(es)={case.ground_truth_indices} &nbsp;|&nbsp; direction={case.direction}</div>'
        "</div>"
    )


def generate_report(
    output_path: str,
    metrics_rows: List[dict],
    per_category: Dict[str, dict],
    failure_summary: str,
    top_failures: List[FailureCase],
    hard_negative_before: Optional[dict] = None,
    hard_negative_after: Optional[dict] = None,
    run_config: Optional[dict] = None,
) -> str:
    run_config = run_config or {}

    r1_values = [row.get("R@1", 0) for row in metrics_rows if "R@1" in row]
    r5_values = [row.get("R@5", 0) for row in metrics_rows if "R@5" in row]
    r10_values = [row.get("R@10", 0) for row in metrics_rows if "R@10" in row]
    cards = "".join([
        _card("Recall@1", f"{(max(r1_values) if r1_values else 0)*100:.1f}%"),
        _card("Recall@5", f"{(max(r5_values) if r5_values else 0)*100:.1f}%"),
        _card("Recall@10", f"{(max(r10_values) if r10_values else 0)*100:.1f}%"),
        _card("Queries Evaluated", str(sum(row.get("n", 0) for row in metrics_rows) if metrics_rows else 0)),
    ])

    hn_section = ""
    if hard_negative_before and hard_negative_after:
        hn_rows = [
            {"stage": "before reranking", **hard_negative_before},
            {"stage": "after hard-negative reranking", **hard_negative_after},
        ]
        hn_section = f"<h2>Hard Negative Reranking Impact (RQ3)</h2>{_metrics_table(hn_rows)}"

    config_items = "".join(f"<li><code>{html.escape(str(k))}</code>: {html.escape(str(v))}</li>" for k, v in run_config.items())

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Multimodal Retrieval Benchmark Report</title>
<style>{_CSS}</style>
</head>
<body>
  <h1>Multimodal Retrieval Benchmark</h1>
  <div class="subtitle">Zero-shot CLIP cross-modal retrieval &mdash; Flickr30K</div>

  <div class="cards">{cards}</div>

  <h2>Direction &amp; Backbone Comparison</h2>
  {_metrics_table(metrics_rows)}

  <h2>Query Category Breakdown (RQ2)</h2>
  {_category_table(per_category)}
  <div class="finding"><strong>Finding:</strong> {html.escape(failure_summary)}</div>

  <h2>Top Failure Examples (RQ3)</h2>
  {''.join(_failure_case_html(c) for c in top_failures) if top_failures else "<p class='subtitle'>No failures recorded.</p>"}

  {hn_section}

  <h2>Run Configuration</h2>
  <ul>{config_items}</ul>

  <footer>Generated by multimodal-retrieval-benchmark &middot; report_generator.py</footer>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(doc)
    return output_path
