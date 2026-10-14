"""Responsive HTML report for deterministic statistical comparison insights.

This module is presentation-only: it consumes the comparison data already
stored on ComparisonReport and does not recalculate comparison statistics.
"""
from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any


def _safe(value: Any) -> str:
    """Escape values before placing them in HTML."""
    if value is None:
        return "—"
    try:
        # Numpy scalar values expose item().
        if hasattr(value, "item"):
            value = value.item()
    except (ValueError, TypeError):
        pass
    if isinstance(value, float):
        if not math.isfinite(value):
            return "—"
        return f"{value:,.6g}"
    # Escape text before embedding it in HTML to prevent values from being interpreted as markup.
    return html.escape(str(value))


def _percent(value: Any) -> str:
    """Format a finite numeric percentage or show an unavailable marker."""
    try:
        number = float(value)
        return f"{number:.2f}%" if math.isfinite(number) else "—"
    except (TypeError, ValueError):
        return "—"


def _unwrap_comparison(comparison: Any) -> Any:
    """Unwrap a comparison result container to obtain its payload."""
    return getattr(comparison, "result", comparison)


def _items(comparison: Any) -> list[dict]:
    """Extract the list of comparison items from a report result."""
    if comparison is None:
        return []
    result = _unwrap_comparison(comparison)
    return list(getattr(result, "metadata_comparisons", []) or [])


def _get_stats(item: dict) -> dict:
    """Return the statistics dictionary associated with a comparison item."""
    stats = item.get("statistics")
    return stats if isinstance(stats, dict) else {}


def _metric_name(item: dict) -> str:
    """Select a display label for a metric from its metadata."""
    return str(item.get("display_name") or item.get("parameter_name") or item.get("parameter_id") or "Unknown metric")


def _asset_name(item: dict) -> str:
    """Select a display label for the asset type associated with a metric."""
    return str(item.get("asset_name") or "asset")


def _field(value: Any, key: str, default: Any = None) -> Any:
    """Read a field from either a mapping or an object."""
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _canonical_asset_name(value: Any) -> str:
    """Normalize asset labels so summary keys and metric labels can be matched."""
    return "".join(ch.casefold() for ch in str(value or "") if ch.isalnum())


def _find_asset_summary(summaries: Any, asset_name: str) -> Any:
    """Find an asset summary by exact or normalized asset-type name."""
    if summaries is None:
        return None
    if isinstance(summaries, dict):
        if asset_name in summaries:
            return summaries[asset_name]
        target = _canonical_asset_name(asset_name)
        for key, value in summaries.items():
            if _canonical_asset_name(key) == target:
                return value
        return None

    # Some result containers expose asset summaries as attributes.
    direct = _field(summaries, asset_name, None)
    if direct is not None:
        return direct
    target = _canonical_asset_name(asset_name)
    for key in dir(summaries):
        if key.startswith("_") or _canonical_asset_name(key) != target:
            continue
        return getattr(summaries, key, None)
    return None


def _model_asset_summary(report: Any, asset_name: str) -> Any:
    """Get the per-asset model summary without recalculating statistics."""
    comparison = _unwrap_comparison(_field(report, "model_comparison", None))
    return _find_asset_summary(comparison, asset_name)


def _state_asset_summary(report: Any, asset_name: str) -> Any:
    """Get the per-asset power-flow summary without recalculating statistics."""
    comparison = _unwrap_comparison(_field(report, "power_flow_comparison", None))
    summaries = _field(comparison, "asset_summaries", None)
    if summaries is None:
        # Support result containers where summaries are one level deeper.
        nested = _field(comparison, "result", None)
        summaries = _field(nested, "asset_summaries", None)
    return _find_asset_summary(summaries, asset_name)


def _asset_summary_badges(summary: Any, scope: str) -> str:
    """Format precomputed asset-level mismatch counts and percentages."""
    if summary is None:
        return "<span class='badge info'>Asset mismatch summary unavailable</span>"

    mismatch_count = _field(summary, "mismatch_asset_count", None)
    mismatch_pct = _field(summary, "mismatch_asset_percentage", None)
    badges = []

    if scope == "model":
        count_a = _field(summary, "asset_count_a", None)
        count_b = _field(summary, "asset_count_b", None)
        if count_a is not None:
            badges.append(
                f"<span class='badge info'>Case A assets: {_safe(count_a)}</span>"
            )
        if count_b is not None:
            badges.append(
                f"<span class='badge info'>Case B assets: {_safe(count_b)}</span>"
            )

    if mismatch_count is not None:
        badges.append(
            f"<span class='badge mid'>Assets mismatched: {_safe(mismatch_count)}</span>"
        )
    if mismatch_pct is not None:
        badges.append(
            f"<span class='badge high'>Asset mismatch rate: {_percent(mismatch_pct)}</span>"
        )
    if not badges:
        badges.append("<span class='badge info'>Asset mismatch summary unavailable</span>")
    return "".join(badges)


def _asset_group_card(
    asset_name: str,
    items: list[dict],
    findings_for_item: Any,
    scope: str,
    report: Any,
    start_index: int,
) -> str:
    """Render one expandable asset-type block containing collapsed metric cards."""
    summary = (
        _model_asset_summary(report, asset_name)
        if scope == "model"
        else _state_asset_summary(report, asset_name)
    )
    metric_html = "".join(
        _metric_card(item, findings_for_item(item), start_index + offset)
        for offset, item in enumerate(items)
    )
    summary_badges = _asset_summary_badges(summary, scope)
    group_id = f"{scope}-asset-{start_index}"
    parameter_label = "parameter" if len(items) == 1 else "parameters"

    return f"""
    <details class="asset-group" id="{html.escape(group_id, quote=True)}"
             data-asset="{html.escape(asset_name.casefold(), quote=True)}">
      <summary>
        <div class="asset-heading">
          <div class="asset-title">{html.escape(asset_name)}</div>
          <div class="asset-subtitle">{len(items)} {parameter_label} · Expand to inspect individual metrics</div>
        </div>
        <div class="badges">{summary_badges}</div>
      </summary>
      <div class="asset-body">{metric_html}</div>
    </details>
    """


def _render_asset_groups(
    items: list[dict],
    findings_for_item: Any,
    scope: str,
    report: Any,
    start_index: int = 1,
) -> str:
    """Group metric cards by asset type while preserving their original order."""
    grouped: dict[str, list[dict]] = {}
    for item in items:
        asset_name = _asset_name(item)
        grouped.setdefault(asset_name, []).append(item)

    rendered = []
    index = start_index
    for asset_name, group_items in grouped.items():
        rendered.append(
            _asset_group_card(
                asset_name=asset_name,
                items=group_items,
                findings_for_item=findings_for_item,
                scope=scope,
                report=report,
                start_index=index,
            )
        )
        index += len(group_items)
    return "".join(rendered)


def _asset_count_label(stats: dict) -> str:
    """Format affected-asset and total-asset counts with the available percentage."""
    affected = stats.get("affected_asset_count")
    total = stats.get("asset_count")
    pct = stats.get("affected_asset_percent")
    if affected is None or total is None:
        return "Not available"
    label = f"{_safe(affected)} / {_safe(total)}"
    return f"{label} ({_percent(pct)})" if pct is not None else label


def _summary_rows(item: dict) -> list[tuple[str, Any]]:
    """Build the statistics rows displayed for one metric in the HTML report."""
    s = _get_stats(item)
    unit = str(item.get("unit") or "")

    def with_unit(key: str) -> str:
        """Format a statistic together with the metric unit when the value is available."""
        return f"{_safe(s.get(key))} {html.escape(unit)}".strip() if s.get(key) is not None else "—"

    rows = [
        ("Values compared", s.get("comparable_count")),
        ("Mismatches", f"{_safe(s.get('mismatch_count'))} ({_percent(s.get('mismatch_percent'))})"),
        ("Affected assets", _asset_count_label(s)),
        ("Mean absolute delta", with_unit("mean_delta")),
        ("Median absolute delta", with_unit("median_delta")),
        ("Median absolute deviation (MAD)", with_unit("mad_delta")),
        ("P95 absolute delta", with_unit("p95_delta")),
        ("P99 absolute delta", with_unit("p99_delta")),
        ("Maximum absolute delta", with_unit("max_delta")),
        ("Configured tolerance", with_unit_value(item.get("tolerance"), unit)),
        ("Severe-threshold exceedances", s.get("severe_mismatch_count")),
    ]
    # Include relative statistics only when present; this avoids cluttering metrics
    # for which relative deltas are not meaningful or were not calculated.
    for key, label in (("median_relative_delta", "Median relative delta"),
                       ("p95_relative_delta", "P95 relative delta"),
                       ("p99_relative_delta", "P99 relative delta"),
                       ("max_relative_delta", "Maximum relative delta"),
                       ("mad_relative_delta", "Relative-delta MAD")):
        if s.get(key) is not None:
            rows.append((label, f"{_safe(s.get(key))}%"))
    return [(label, value) for label, value in rows if value is not None]


def with_unit_value(value: Any, unit: str) -> str:
    """Format a value and its unit for display, or show an unavailable marker."""
    return f"{_safe(value)} {html.escape(unit)}".strip() if value is not None else "—"


def _largest_differences(item: dict, limit: int = 5) -> list[tuple[str, str]]:
    """Rank asset deltas from the existing delta array; no deltas are recalculated."""
    delta = item.get("delta")
    if delta is None:
        return []
    try:
        values = list(delta.reshape(-1) if hasattr(delta, "reshape") else delta)
    except TypeError:
        return []
    ids = item.get("asset_ids")
    if ids is None:
        ids = list(range(len(values)))
    else:
        try:
            ids = list(ids.reshape(-1) if hasattr(ids, "reshape") else ids)
        except TypeError:
            ids = list(range(len(values)))
    mismatch_mask = item.get("mismatch_mask")
    try:
        mask = list(mismatch_mask.reshape(-1) if hasattr(mismatch_mask, "reshape") else mismatch_mask) if mismatch_mask is not None else [True] * len(values)
    except TypeError:
        mask = [True] * len(values)
    candidates = []
    for i, value in enumerate(values):
        try:
            magnitude = abs(float(value))
            if not math.isfinite(magnitude) or (i < len(mask) and not bool(mask[i])):
                continue
        except (TypeError, ValueError):
            continue
        asset_id = ids[i] if i < len(ids) else i
        candidates.append((magnitude, asset_id))
    candidates.sort(key=lambda row: row[0], reverse=True)
    unit = str(item.get("unit") or "")
    return [(f"{_asset_name(item)} {_safe(asset_id)}", f"{_safe(magnitude)} {html.escape(unit)}".strip()) for magnitude, asset_id in candidates[:limit]]


def _direction_rows(item: dict) -> list[tuple[str, Any]]:
    """Build summary rows for positive and negative signed-delta proportions."""
    s = _get_stats(item)
    positive = s.get("positive_delta_percent")
    negative = s.get("negative_delta_percent")
    rows = []
    if positive is not None:
        rows.append(("Case B > Case A", _percent(positive)))
    if negative is not None:
        rows.append(("Case B < Case A", _percent(negative)))
    return rows


def _findings_by_metric(findings):
    """Group generated findings by the metric they describe."""
    model_by_metric: dict[str, list[dict]] = {}
    flow_by_metric: dict[str, list[dict]] = {}
    overall = []
    targets = {
        "model": model_by_metric,
        "power_flow": flow_by_metric,
    }
    for scope, target in targets.items():
        for finding in findings.get(scope, []) or []:
            if not isinstance(finding, dict):
                finding = {"message": str(finding), "category": "general"}
            key = str(finding.get("parameter_name") or "")
            if key:
                target.setdefault(key.casefold(), []).append(finding)
            else:
                overall.append(finding)
    overall.extend(f for f in (findings.get("overall", []) or []) if isinstance(f, dict))
    return model_by_metric, flow_by_metric, overall


def _category_label(category: Any) -> tuple[str, str]:
    """Convert an internal finding category to its display label."""
    category = str(category or "general")
    labels = {
        "isolated_mismatches": ("Concentrated mismatches", "mid"),
        "widespread_mismatches": ("Widespread mismatches", "high"),
        "percentile_outlier_pattern": ("Extreme outlier pattern", "high"),
        "percentile_distribution": ("Distribution pattern", "mid"),
        "tolerance_exceedance": ("Tolerance exceeded", "high"),
        "severe_threshold_exceedance": ("Severe threshold", "high"),
        "directional_bias": ("Directional pattern", "mid"),
        "relative_delta_outlier": ("Relative outlier pattern", "mid"),
        "dominant_asset_group": ("Asset-group pattern", "mid"),
        "parameter_specific_pattern": ("Metric-specific pattern", "mid"),
        "input_discrepancy_signal": ("Input discrepancy signal", "mid"),
        "overall_comparison_pattern": ("Scope summary", "info"),
    }
    if category in labels:
        return labels[category]
    return (category.replace("_", " ").capitalize(), "info")


def _metric_card(item: dict, findings: list[dict], index: int) -> str:
    """Render the metric details, statistics, and findings as an HTML card."""
    name = _metric_name(item)
    s = _get_stats(item)
    count = s.get("comparable_count", 0) or 0
    mismatches = s.get("mismatch_count", 0) or 0
    pct = s.get("mismatch_percent")
    try:
        pct_num = float(pct or 0)
    except (TypeError, ValueError):
        pct_num = 0
    badge_class = "high" if pct_num >= 50 else "mid" if pct_num > 5 else "low"
    unit = str(item.get("unit") or "")
    metric_id = f"metric-{index}"

    stat_rows = "".join(f"<tr><th scope='row'>{label}</th><td>{value}</td></tr>" for label, value in _summary_rows(item))
    top = _largest_differences(item)
    if top:
        top_rows = "".join(f"<tr><td>{rank}</td><td>{asset}</td><td>{delta}</td></tr>" for rank, (asset, delta) in enumerate(top, 1))
        top_html = f"<div class='table-wrap'><table><thead><tr><th>Rank</th><th>Asset</th><th>Absolute delta</th></tr></thead><tbody>{top_rows}</tbody></table></div>"
    else:
        top_html = "<p class='muted'>Asset-level delta ranking is unavailable for this metric.</p>"

    direction = _direction_rows(item)
    direction_html = ""
    if direction:
        direction_rows = "".join(f"<tr><th scope='row'>{label}</th><td>{value}</td></tr>" for label, value in direction)
        direction_html = f"<h3>Signed direction among mismatches</h3><div class='table-wrap'><table><tbody>{direction_rows}</tbody></table></div>"

    if findings:
        finding_html = "".join(
            f"<li><span class='badge {_category_label(f.get('category'))[1]}'>{html.escape(_category_label(f.get('category'))[0])}</span> {html.escape(str(f.get('message') or f.get('description') or ''))}</li>"
            for f in findings
        )
        findings_html = f"<div class='finding-list'><h3>Statistical interpretations</h3><ul>{finding_html}</ul></div>"
    else:
        findings_html = ""

    return f"""<details class='metric' id='{metric_id}' data-search='{html.escape((name + ' ' + _asset_name(item)).casefold(), quote=True)}'>
      <summary><div><div class='metric-title'>{html.escape(name)}</div><div class='metric-subtitle'>{html.escape(_asset_name(item))} · { _safe(mismatches) } / {_safe(count)} mismatches</div></div><div class='badges'><span class='badge {badge_class}'>{_percent(pct)} mismatch rate</span><span class='badge info'>{html.escape(unit or 'unit not specified')}</span></div></summary>
      <div class='metric-body'><div class='metric-grid'>
        <section><h3>Statistical summary</h3><div class='table-wrap'><table class='stats'><tbody>{stat_rows}</tbody></table></div></section>
        <section><h3>Largest differences (Outliners)</h3>{top_html}{direction_html}</section>
      </div>{findings_html}</div></details>"""


def _overall_cards(items: list[dict], label: str) -> str:
    """Render the headline overview cards for the report."""
    comparable = sum(int(_get_stats(i).get("comparable_count", 0) or 0) for i in items)
    mismatches = sum(int(_get_stats(i).get("mismatch_count", 0) or 0) for i in items)
    rate = 100.0 * mismatches / comparable if comparable else 0.0
    affected_metrics = sum(1 for i in items if int(_get_stats(i).get("mismatch_count", 0) or 0) > 0)
    rate_class = "high" if rate >= 50 else "mid" if rate > 5 else "low"
    return f"<div class='card'><div class='label'>{html.escape(label)}</div><div class='value'>{_percent(rate)}</div><div class='sub'>{_safe(mismatches)} mismatches / {_safe(comparable)} metric-value comparisons</div><div class='sub'>{_safe(affected_metrics)} metrics with mismatches</div><span class='badge {rate_class}'>Aggregated mismatch rate</span></div>"


def _legend_html() -> str:
    """Render the explanation of status labels and report statistics."""
    entries = [
        ("Extreme outlier pattern", "The maximum is much larger than P95 (in this engine, often at least 3× P95). Example: P95 = 2 units and max = 10 units. The largest difference may be concentrated in a few values; inspect the top assets and verify source values/units."),
        ("Concentrated mismatches", "A small share of compared values mismatch (the generator classifies ≤10% as isolated). Example: 10 mismatches out of 411 = 2.43%. Check the specific affected assets first."),
        ("Widespread mismatches", "A large share of values mismatch (the generator classifies ≥50% as widespread). Example: 69 of 69 values = 100%. Check shared assumptions, case setup, conventions, or broad operating-state differences before treating each item separately."),
        ("P95 exceeds tolerance", "The 95th-percentile absolute delta is above the configured mismatch tolerance. Differences are not limited to just one maximum; inspect mismatch rate and affected assets, then compare the configured tolerance and units."),
        ("Heavy upper tail", "P95 is much larger than the median, or P99 is much larger than P95. Example: median = 1, P95 = 8, P99 = 30. A subset of values has much larger differences; inspect the distribution and ranked assets."),
        ("Mean much larger than median", "A few large values may pull the mean upward. Example: median = 2 but mean = 15. Review P95, maximum, and top assets; do not use the mean alone to describe a typical difference."),
        ("Directional pattern", "Most signed deltas share a sign. Here signed delta means Case B − Case A (except where a metric-specific comparison convention applies). Example: 95% positive means Case B is higher for most mismatches. Check whether this aligns with expected model or operating changes."),
        ("Relative outlier pattern", "Relative deltas are large compared with typical relative differences. Small baseline values can inflate percentages, so inspect absolute and relative deltas together and confirm denominator/baseline values."),
        ("MAD", "Median absolute deviation: the median distance from the median delta magnitude. A zero MAD can occur when at least half the values sit at the same magnitude (often zero); it does not mean every value matches."),
        ("Affected assets", "Distinct assets with at least one mismatch, divided by distinct compared assets when asset IDs are available. This is not always the same as the number of mismatching metric values."),
        ("Severe-threshold exceedance", "Count of values exceeding the configured severe threshold. Interpret it using the metric's unit and configuration; a threshold is not automatically a universal engineering limit."),
    ]
    cards = "".join(f"<article class='legend-item'><h3>{html.escape(title)}</h3><p>{html.escape(body)}</p></article>" for title, body in entries)
    return f"<section id='legend' class='section'><div class='section-heading'><div><h2>Legend: insight terms and how to investigate</h2><p>These labels describe statistical patterns, not confirmed root causes.</p></div></div><div class='legend-grid'>{cards}</div><div class='callout'><strong>Recommended investigation sequence</strong><ol><li>Confirm the metric, unit, tolerance, severe threshold, and comparison convention.</li><li>Check mismatch rate and affected-asset coverage to determine whether the issue is localized or broad.</li><li>Compare median, MAD, P95, P99, and maximum to understand typical versus tail behavior.</li><li>Inspect the largest-delta assets and compare their Case A, Case B, and signed-delta values.</li><li>For relative outliers, inspect baseline values because small denominators can inflate percentages.</li><li>Compare related metrics and nearby assets, then record a root cause only after checking the underlying inputs and operating assumptions.</li></ol></div></section>"


def _styles() -> str:
    """Return the embedded CSS used to style the self-contained HTML report."""
    return r"""
:root{--ink:#182230;--muted:#64748b;--line:#e2e8f0;--canvas:#f4f7fb;--blue:#2563eb;--blue-soft:#eff6ff;--red:#b42318;--red-soft:#fef3f2;--amber:#92400e;--amber-soft:#fffaeb;--green:#166534;--green-soft:#f0fdf4}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--canvas);color:var(--ink);font:15px/1.55 Inter,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.hero{background:linear-gradient(120deg,#10283f,#1e4970);color:white;padding:34px max(22px,calc((100vw - 1180px)/2))}.eyebrow{text-transform:uppercase;letter-spacing:.13em;font-size:11px;font-weight:800;color:#b9d8f5}h1{font-size:clamp(28px,4vw,40px);line-height:1.12;margin:8px 0;letter-spacing:-.035em}.hero p{margin:0;color:#e2edf8;max-width:800px}.layout{max-width:1180px;margin:24px auto 60px;padding:0 20px;display:grid;grid-template-columns:205px minmax(0,1fr);gap:22px;align-items:start}nav{position:sticky;top:18px;background:white;border:1px solid var(--line);border-radius:14px;padding:15px}nav h2{font-size:12px;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);margin:0 0 9px}nav a{display:block;padding:8px;border-radius:8px;color:#334155;text-decoration:none;font-size:13px}nav a:hover{background:var(--blue-soft)}main{min-width:0}.section{margin-bottom:28px;scroll-margin-top:20px}.section-heading{margin-bottom:13px}.section-heading h2{font-size:23px;letter-spacing:-.025em;margin:0}.section-heading p,.muted{color:var(--muted);font-size:13px;margin:3px 0 0}.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.card,.metric,.panel{background:white;border:1px solid var(--line);border-radius:14px;box-shadow:0 2px 7px #0f172a08}.card{padding:17px}.label{color:var(--muted);font-size:12px;font-weight:650}.value{font-size:28px;font-weight:800;letter-spacing:-.04em;margin:4px 0}.sub{color:var(--muted);font-size:12px;margin:3px 0}.callout{margin-top:14px;padding:16px 18px;border-radius:12px;background:white;border:1px solid var(--line);border-left:4px solid var(--blue)}.callout p{margin:0;color:#475569}.toolbar{display:flex;gap:9px;flex-wrap:wrap;margin:0 0 12px}input{flex:1;min-width:180px;border:1px solid #cbd5e1;border-radius:9px;padding:10px 12px;font:inherit}button{cursor:pointer;border:1px solid #cbd5e1;background:white;border-radius:9px;padding:9px 12px;color:#334155;font-weight:650}.metric{margin-bottom:13px;overflow:hidden}.metric summary{cursor:pointer;list-style:none;display:flex;align-items:center;justify-content:space-between;gap:14px;padding:16px 18px}.metric summary::-webkit-details-marker{display:none}.metric summary:after{content:"＋";font-size:20px;color:var(--muted)}.metric[open] summary:after{content:"−"}.metric-title{font-size:16px;font-weight:780}.metric-subtitle{color:var(--muted);font-size:12px;margin-top:2px}.badges{display:flex;gap:6px;flex-wrap:wrap;align-items:center}.badge{display:inline-block;font-size:11px;font-weight:750;border-radius:999px;padding:4px 8px;white-space:nowrap;margin:1px}.badge.high{color:var(--red);background:var(--red-soft)}.badge.mid{color:var(--amber);background:var(--amber-soft)}.badge.low{color:var(--green);background:var(--green-soft)}.badge.info{color:#1e40af;background:var(--blue-soft)}.metric-body{padding:0 18px 18px;border-top:1px solid #edf1f5}.metric-grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:18px;margin-top:15px}h3{margin:0 0 9px;font-size:13px;text-transform:uppercase;letter-spacing:.05em;color:#475569}table{border-collapse:collapse;width:100%;font-size:13px}th{color:#64748b;font-size:11px;text-transform:uppercase;letter-spacing:.04em;background:#f8fafc}th,td{text-align:left;padding:9px 10px;border-bottom:1px solid #edf1f5;vertical-align:top}td:last-child{text-align:right;font-variant-numeric:tabular-nums}tbody tr:last-child th,tbody tr:last-child td{border-bottom:0}.table-wrap{border:1px solid var(--line);border-radius:10px;overflow:auto}.stats th{width:58%;text-transform:none;letter-spacing:0;font-size:12px;font-weight:550;background:white}.stats td{font-weight:650;color:#172b42}.finding-list{margin-top:18px;padding:14px 15px;background:#f8fafc;border-radius:10px}.finding-list ul{margin:0;padding-left:20px}.finding-list li{margin:10px 0;color:#475569}.legend-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.legend-item{background:white;border:1px solid var(--line);border-radius:12px;padding:15px}.legend-item h3{color:var(--ink);margin-bottom:5px}.legend-item p{margin:0;color:#475569;font-size:13px}.footer{border-top:1px solid var(--line);padding:18px 0;color:var(--muted);font-size:12px}@media(max-width:850px){.layout{grid-template-columns:1fr}nav{position:static}nav div{display:grid;grid-template-columns:repeat(2,1fr)}.cards,.metric-grid,.legend-grid{grid-template-columns:1fr}}@media print{body{background:white}.hero{padding:22px;print-color-adjust:exact}.layout{display:block;max-width:none;margin:12px 0;padding:0}nav,.toolbar,button{display:none!important}.metric{break-inside:avoid;box-shadow:none}.metric:not([open]) .metric-body{display:block}.metric summary:after{display:none}.cards{grid-template-columns:repeat(2,1fr)}.legend-grid{grid-template-columns:repeat(2,1fr)}a{color:inherit}}

/* Expandable asset-type parent blocks */
.asset-group{margin-bottom:14px;background:white;border:1px solid var(--line);border-radius:14px;box-shadow:0 2px 7px #0f172a08;overflow:hidden}
.asset-group:last-child{margin-bottom:0}
.asset-group>summary{cursor:pointer;list-style:none;display:flex;align-items:center;justify-content:space-between;gap:14px;padding:16px 18px}
.asset-group>summary::-webkit-details-marker{display:none}
.asset-group>summary:after{content:"＋";font-size:20px;color:var(--muted);flex-shrink:0}
.asset-group[open]>summary:after{content:"−"}
.asset-heading{min-width:0;flex:1}
.asset-title{font-size:16px;font-weight:800;overflow-wrap:anywhere}
.asset-subtitle{color:var(--muted);font-size:12px;margin-top:3px}
.asset-group>summary .badges{flex-shrink:0;justify-content:flex-end}
.asset-body{padding:14px 18px 5px;border-top:1px solid #edf1f5;background:#fcfdff}
.asset-body>.metric{margin-bottom:10px}
.asset-body>.metric:last-child{margin-bottom:8px}
@media(max-width:650px){
.asset-group>summary{align-items:flex-start;flex-direction:column;gap:8px}
.asset-group>summary:after{align-self:flex-end;margin-top:-4px}
.asset-group>summary .badges{flex-shrink:1;justify-content:flex-start}
.asset-body{padding:12px 14px 5px}
}
@media print{
.asset-group{break-inside:avoid;box-shadow:none}
.asset-group:not([open])>.asset-body{display:block}
.asset-group>summary:after{display:none}
.asset-body>.metric:not([open])>.metric-body{display:block}
}
"""


def generate_statistical_insights_html(report: Any, output_path: str | Path) -> Path:
    """Write an interactive HTML report for ``report.statistical_insights``.

    Args:
        report: ComparisonReport returned by ``build_report``.
        output_path: Destination .html path.

    Returns:
        The output path.
    """
    findings = getattr(report, "statistical_insights", None) or {}
    model_items = _items(getattr(report, "model_comparison", None))
    flow_items = _items(getattr(report, "power_flow_comparison", None))
    model_by_metric, state_by_metric, overall_findings = _findings_by_metric(findings)

    # Findings carry parameter_name; match them against the metric display/name.
    def relevant(item: dict) -> list[dict]:
        """Determine whether a finding references the metric represented by the current card."""
        names = {
            _metric_name(item).casefold(),
            str(item.get("parameter_name") or "").casefold(),
            str(item.get("parameter_id") or "").casefold(),
        }
        matched = []
        by_metric = model_by_metric if item['scope'] == 'model' else state_by_metric
        for key, group in by_metric.items():
            if key in names:
                matched.extend(group)
        return matched

    model_cards = _render_asset_groups(
        model_items, relevant, "model", report, start_index=1
    )
    flow_cards = _render_asset_groups(
        flow_items, relevant, "power_flow", report, start_index=1 + len(model_items)
    )
    overall_html = "".join(
        f"<li><span class='badge info'>{html.escape(_category_label(f.get('category'))[0])}</span> {html.escape(str(f.get('message') or f.get('description') or ''))}</li>"
        for f in overall_findings
    ) or "<li>No overall pattern findings were generated.</li>"

    model_cards_html = model_cards or "<p class='muted'>No model comparison metadata is available.</p>"
    flow_cards_html = flow_cards or "<p class='muted'>No power-flow comparison metadata is available or the cases were not comparable.</p>"
    model_summary = _overall_cards(model_items, "Model comparison")
    flow_summary = _overall_cards(flow_items, "Power-flow comparison")

    page = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Statistical Insights — Power System Comparison</title><style>{_styles()}</style></head><body>
<header class='hero'><div class='eyebrow'>Engineering comparison · Statistical analysis</div><h1>Statistical Insights</h1><p>Case A vs. Case B — mismatch prevalence, delta distributions, signed direction, and asset-level exceptions.</p></header>
<div class='layout'><nav aria-label='Report navigation'><h2>On this page</h2><div><a href='#overview'>01 · Overview</a><a href='#overall'>02 · Overall patterns</a><a href='#model'>03 · Model comparison</a><a href='#power-flow'>04 · Power-flow comparison</a><a href='#legend'>05 · Legend & investigation tips</a></div></nav><main>
<section id='overview' class='section'><div class='section-heading'><h2>Overall comparison</h2><p>Aggregated counts across the available metric-value comparisons.</p></div><div class='cards'>{model_summary}{flow_summary}</div><div class='callout'><strong>How to use this report</strong><p>Start with mismatch rates, then examine each metric's distribution and largest affected assets. A statistical pattern is a signal to investigate, not by itself proof of a root cause.</p></div></section>
<section id='overall' class='section'><div class='section-heading'><h2>Overall comparison patterns</h2><p>Cross-metric observations generated by the statistical insight engine.</p></div><div class='panel' style='padding:16px 18px'><ul>{overall_html}</ul></div></section>
<section id='model' class='section'><div class='section-heading'><h2>Model comparison</h2><p>Parameters are grouped by asset type. Expand an asset type to inspect its individual parameter details.</p></div><div class='toolbar'><input id='metricSearch' type='search' placeholder='Filter metrics or asset types…' aria-label='Filter metrics'><button id='expandAll' type='button'>Expand all</button><button id='collapseAll' type='button'>Collapse all</button></div>{model_cards_html}</section>
<section id='power-flow' class='section'><div class='section-heading'><h2>Power-flow comparison</h2><p>Results are grouped by asset type, with mismatch counts and percentages plus expandable parameter details.</p></div>{flow_cards_html}</section>
{_legend_html()}
<div class='footer'>Generated from the existing ComparisonReport object. Comparison values and statistics are taken from the comparison engine; this module formats the results for HTML and does not change comparison logic.</div>
</main></div><script>
const search=document.getElementById('metricSearch');const modelSection=document.getElementById('model');const modelGroups=Array.from(modelSection.querySelectorAll('details.asset-group'));const modelCards=Array.from(modelSection.querySelectorAll('details.metric'));search.addEventListener('input',()=>{{const q=search.value.trim().toLowerCase();modelCards.forEach(c=>{{c.hidden=!!q&&!c.innerText.toLowerCase().includes(q)&&!c.dataset.search.includes(q)}});modelGroups.forEach(g=>{{const children=Array.from(g.querySelectorAll('details.metric'));const visible=children.some(c=>!c.hidden);g.hidden=!!q&&!visible;if(q&&visible)g.open=true}})}});document.getElementById('expandAll').addEventListener('click',()=>modelSection.querySelectorAll('details').forEach(d=>{{if(!d.hidden)d.open=true}}));document.getElementById('collapseAll').addEventListener('click',()=>modelSection.querySelectorAll('details').forEach(d=>d.open=false));
</script></body></html>"""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(page, encoding="utf-8")
    return destination
