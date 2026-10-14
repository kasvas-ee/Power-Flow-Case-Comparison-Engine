"""Evidence-grounded AI engineering insights for Pandapower case comparisons.

The numerical comparison engine remains authoritative. This module packages its
existing outputs, requests structured interpretations from a local OpenAI-compatible
LLM server, validates evidence references, and renders a self-contained HTML report.
"""
from __future__ import annotations

import html
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

MAX_EVIDENCE_ITEMS = 60
MAX_ASSET_EXAMPLES = 4

JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "executive_assessment": {"type": "string"},
        "diagnostic_narratives": {
            "type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "title": {"type": "string"},
                    "hypothesis": {"type": "string"},
                    "supporting_evidence_ids": {"type": "array", "items": {"type": "string"}},
                    "counter_evidence_or_unknowns": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "recommended_checks": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["title", "hypothesis", "supporting_evidence_ids", "counter_evidence_or_unknowns", "confidence", "recommended_checks"],
            },
        },
        "consistency_questions": {"type": "array", "items": {"type": "string"}},
        "evidence_gaps": {"type": "array", "items": {"type": "string"}},
        "overall_assessment": {"type": "string", "enum": ["likely_model_related", "likely_solution_related", "multiple_plausible_explanations", "insufficient_evidence", "no_material_issue_identified"]},
    },
    "required": ["executive_assessment", "diagnostic_narratives", "consistency_questions", "evidence_gaps", "overall_assessment"],
}

SYSTEM_PROMPT = """You are a cautious power-systems engineering diagnostic assistant reviewing two Pandapower case-comparison outputs.
Your task is NOT to repeat statistics. Interpret relationships across the supplied evidence, propose testable hypotheses, prioritize targeted engineering checks, and identify contradictions or missing evidence.
Rules:
- The supplied JSON is the only factual source. Never invent assets, values, limits, controls, topology, solver settings, or causes.
- Use only evidence IDs that exist in the supplied evidence list. Cite evidence IDs for every diagnostic narrative.
- Separate observations from hypotheses. A relationship is not proof of causation.
- Do not claim overload, voltage-limit violation, instability, or non-convergence unless the supplied evidence explicitly establishes it.
- Do not alter or reinterpret deterministic statuses. Treat NOT PERFORMED as unknown, not as a match.
- Prefer a few useful diagnostic narratives over a list of every metric. Avoid restating counts, maxima, deltas, and other statistics except where essential to explain a hypothesis.
- Give actionable checks, ordered from likely upstream causes to downstream symptoms. Every check must be relevant to the evidence.
- If evidence is weak, say so and lower confidence. Do not manufacture a high-confidence explanation.
- Output only the required JSON object matching the schema."""


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field from an object or mapping and return a default when it is absent."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _finite_number(value: Any) -> float | None:
    """Convert a value to a finite number, returning no value when conversion is unsafe."""
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _json_safe(value: Any) -> Any:
    """Convert nested values into JSON-safe representations, excluding non-finite numbers."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value[:MAX_ASSET_EXAMPLES]]
    return str(value)[:300]


def _as_flat_list(value: Any) -> list[Any] | None:
    """Normalize a value into a flat list for evidence packaging."""
    if value is None:
        return None
    try:
        import numpy as np
        return np.asarray(value, dtype=object).reshape(-1).tolist()
    except Exception:
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]


def _status_text(value: Any) -> str:
    """Normalize a comparison status for display in the evidence report."""
    return str(value if value is not None else "NOT PERFORMED").strip().upper()


def build_evidence_package(report: Any) -> dict[str, Any]:
    """Create a compact, bounded evidence package from existing report objects."""
    evidence: list[dict[str, Any]] = []
    counter = 1

    def add(kind: str, title: str, facts: dict[str, Any]) -> str:
        """Add a bounded evidence record and assign its reference identifier."""
        nonlocal counter
        evidence_id = f"E{counter:03d}"
        counter += 1
        evidence.append({"id": evidence_id, "kind": kind, "title": title, "facts": facts})
        return evidence_id

    ybus = _get(report, "ybus_comparison")
    if ybus is not None:
        add("ybus", "Primary Ybus comparison", {
            "status": _status_text(_get(ybus, "is_match", False) and "MATCH" or "MISMATCH"),
            "bus_count_match": _json_safe(_get(ybus, "bus_count_match")),
            "topology_match": _json_safe(_get(ybus, "topology_match")),
            "ybus_value_match": _json_safe(_get(ybus, "ybus_value_match")),
            "pattern_diff_count": _json_safe(_get(ybus, "pattern_diff_count")),
            "value_diff_count": _json_safe(_get(ybus, "value_diff_count")),
            "max_value_diff": _json_safe(_get(ybus, "max_value_diff")),
            "reason": _json_safe(_get(ybus, "reason")),
        })

    add("overall", "Comparison prerequisites and overall statuses", {
        "model_match": _json_safe(_get(report, "model_match")),
        "results_comparable": _json_safe(_get(report, "results_comparable")),
        "results_match": _json_safe(_get(report, "results_match")),
        "validation_is_valid": _json_safe(_get(_get(report, "validation"), "is_valid")),
        "validation_reason": _json_safe(_get(_get(report, "validation"), "reason")),
    })

    model = _get(report, "model_comparison")
    state = _get(report, "power_flow_comparison")
    groups = [("model", _get(model, "metadata_comparisons", []) or []),
              ("power_flow", _get(state, "metadata_comparisons", []) or [])]

    for scope, entries in groups:
        for item in entries:
            if len(evidence) >= MAX_EVIDENCE_ITEMS:
                break
            status = _status_text(item.get("status"))
            stats = item.get("statistics") or {}
            mismatch_mask = _as_flat_list(item.get("mismatch_mask")) or []
            severe_mask = _as_flat_list(item.get("severe_mask")) or []
            delta = _as_flat_list(item.get("delta"))
            case_a = _as_flat_list(item.get("case_a"))
            case_b = _as_flat_list(item.get("case_b"))
            asset_ids = _as_flat_list(item.get("asset_ids"))
            mismatch_indices = [i for i, flag in enumerate(mismatch_mask) if bool(flag)]
            if not mismatch_indices and status not in {"NOT PERFORMED", "YES", "MATCH"}:
                mismatch_indices = list(range(min(1, len(delta or []))))
            # For matching metrics, keep only status if they help establish context.
            if status in {"YES", "MATCH"} and not mismatch_indices:
                continue
            samples = []
            for i in mismatch_indices[:MAX_ASSET_EXAMPLES]:
                aid = asset_ids[i] if asset_ids is not None and i < len(asset_ids) else i
                samples.append({
                    "asset_id": _json_safe(aid),
                    "case_a": _json_safe(case_a[i]) if case_a is not None and i < len(case_a) else None,
                    "case_b": _json_safe(case_b[i]) if case_b is not None and i < len(case_b) else None,
                    "delta": _json_safe(delta[i]) if delta is not None and i < len(delta) else None,
                    "severe": bool(severe_mask[i]) if i < len(severe_mask) else False,
                })
            facts = {
                "scope": scope,
                "asset_name": _json_safe(item.get("asset_name")),
                "parameter_name": _json_safe(item.get("parameter_name")),
                "display_name": _json_safe(item.get("display_name")),
                "parameter_id": _json_safe(item.get("parameter_id")),
                "status": status,
                "unit": _json_safe(item.get("unit")),
                "tolerance": _json_safe(item.get("tolerance")),
                "severe_threshold": _json_safe(item.get("severe_threshold")),
                "reason": _json_safe(item.get("reason")),
                "mismatch_count": _json_safe(stats.get("mismatch_count")),
                "severe_mismatch_count": _json_safe(stats.get("severe_mismatch_count")),
                "comparable_count": _json_safe(stats.get("comparable_count")),
                "mismatch_examples": samples,
            }
            add(scope, f"{scope}: {item.get('asset_name', 'asset')} / {item.get('display_name') or item.get('parameter_name') or item.get('parameter_id') or 'metric'}", facts)

    # System summary is context only; no second calculation is performed.
    system = _get(state, "system") if state is not None else None
    if system is not None:
        system_facts = {}
        for name in getattr(system, "__dataclass_fields__", {}):
            system_facts[name] = _json_safe(_get(system, name))
        if not system_facts and isinstance(system, dict):
            system_facts = {str(k): _json_safe(v) for k, v in system.items()}
        add("system_summary", "Existing system-level summary", system_facts)

    limitations = []
    if state is None:
        limitations.append("Power-flow comparison output is unavailable; no solved-state interpretation is possible.")
    if not _get(report, "results_comparable", False):
        limitations.append("The comparison engine reports that solved results are not comparable.")
    if any(item.get("facts", {}).get("status") == "NOT PERFORMED" for item in evidence):
        limitations.append("At least one relevant metric was NOT PERFORMED; its status is unknown, not a match.")
    if not evidence:
        limitations.append("No usable comparison evidence was available to the AI layer.")

    return {
        "schema_version": "1.0",
        "purpose": "Engineering interpretation, testable hypotheses, targeted checks, and evidence gaps; not statistical reporting.",
        "deterministic_statuses_are_authoritative": True,
        "limitations": limitations,
        "evidence": evidence[:MAX_EVIDENCE_ITEMS],
    }


def _insight_schema() -> dict[str, Any]:
    """Build the JSON schema used to validate the structured insight response."""
    return JSON_SCHEMA


def _validate_insights(payload: Any, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate the response structure and discard evidence references not in the supplied package."""
    if not isinstance(payload, dict):
        raise ValueError("AI response is not a JSON object")
    required = set(JSON_SCHEMA["required"])
    if not required.issubset(payload):
        raise ValueError("AI response is missing required fields")
    # Evidence references are checked against this allow-list before model output is accepted.
    allowed_ids = {str(item["id"]) for item in evidence}
    allowed_assessments = {"likely_model_related", "likely_solution_related", "multiple_plausible_explanations", "insufficient_evidence", "no_material_issue_identified"}
    if payload["overall_assessment"] not in allowed_assessments:
        payload["overall_assessment"] = "insufficient_evidence"
    if not isinstance(payload["diagnostic_narratives"], list):
        payload["diagnostic_narratives"] = []
    cleaned = []
    for narrative in payload["diagnostic_narratives"][:6]:
        if not isinstance(narrative, dict):
            continue
        refs = [str(ref) for ref in narrative.get("supporting_evidence_ids", []) if str(ref) in allowed_ids]
        confidence = str(narrative.get("confidence", "low")).lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "low"
        # No evidence references means the hypothesis cannot be safely surfaced.
        if not refs:
            continue
        cleaned.append({
            "title": str(narrative.get("title", "Diagnostic hypothesis"))[:180],
            "hypothesis": str(narrative.get("hypothesis", ""))[:1800],
            "supporting_evidence_ids": refs,
            "counter_evidence_or_unknowns": str(narrative.get("counter_evidence_or_unknowns", "Not established by available evidence."))[:1200],
            "confidence": confidence,
            "recommended_checks": [str(x)[:500] for x in narrative.get("recommended_checks", [])[:6]],
        })
    payload["diagnostic_narratives"] = cleaned
    for key in ("consistency_questions", "evidence_gaps"):
        value = payload.get(key, [])
        payload[key] = [str(x)[:700] for x in value[:8]] if isinstance(value, list) else []
    payload["executive_assessment"] = str(payload.get("executive_assessment", "Insufficient evidence for a reliable interpretation."))[:1800]
    return payload


def _call_local_model(evidence_package: dict[str, Any], settings: Any) -> dict[str, Any]:
    """Request structured diagnostic insights from the configured local model endpoint."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Missing optional dependency 'openai'. Install requirements-ai-insights.txt.") from exc

    base_url = os.getenv("AI_INSIGHT_BASE_URL") or getattr(settings, "ai_insight_base_url", None) or "http://localhost:8000/v1"
    model = os.getenv("AI_INSIGHT_MODEL") or getattr(settings, "ai_insight_model", None) or "Qwen/Qwen3-14B"
    api_key = os.getenv("AI_INSIGHT_API_KEY", "EMPTY")
    timeout = float(getattr(settings, "ai_insight_timeout_seconds", 180) or 180)
    max_tokens = int(getattr(settings, "ai_insight_max_tokens", 2200) or 2200)
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=1)

    user_prompt = (
        "Analyze this evidence package. Produce a concise engineering diagnostic report, not a statistics summary. "
        "Use evidence IDs in each hypothesis and recommend checks that would discriminate between explanations.\n\n"
        + json.dumps(evidence_package, ensure_ascii=False, separators=(",", ":"))
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
        temperature=0.15,
        max_tokens=max_tokens,
        response_format={"type": "json_schema", "json_schema": {"name": "power_system_ai_insights", "strict": True, "schema": _insight_schema()}},
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Local model returned an empty response")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Local model returned invalid JSON: {exc}") from exc
    return _validate_insights(payload, evidence_package["evidence"])


def _esc(value: Any) -> str:
    """Escape a value for safe insertion into HTML."""
    return html.escape(str(value if value is not None else ""), quote=True)


def _evidence_index(evidence: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build a lookup from evidence identifiers to their records."""
    return {str(item["id"]): item for item in evidence}


def _render_html(insights: dict[str, Any], package: dict[str, Any], model_name: str) -> str:
    """Render the validated insight content as a standalone HTML report."""
    index = _evidence_index(package.get("evidence", []))
    assessment_labels = {
        "likely_model_related": "Likely model-related",
        "likely_solution_related": "Likely solution-related",
        "multiple_plausible_explanations": "Multiple plausible explanations",
        "insufficient_evidence": "Insufficient evidence",
        "no_material_issue_identified": "No material issue identified",
    }
    narrative_html = []
    for i, item in enumerate(insights.get("diagnostic_narratives", []), 1):
        refs = item.get("supporting_evidence_ids", [])
        cards = []
        for ref in refs:
            ev = index.get(ref)
            if not ev:
                continue
            facts = ev.get("facts", {})
            concise = {k: v for k, v in facts.items() if k in {"scope", "asset_name", "parameter_name", "display_name", "status", "unit", "mismatch_count", "severe_mismatch_count", "reason", "mismatch_examples", "is_match", "topology_match", "ybus_value_match"}}
            cards.append(f'<details class="evidence"><summary>{_esc(ref)} · {_esc(ev.get("title"))}</summary><pre>{_esc(json.dumps(concise, indent=2, ensure_ascii=False))}</pre></details>')
        checks = "".join(f"<li>{_esc(check)}</li>" for check in item.get("recommended_checks", [])) or "<li>No specific check supplied.</li>"
        narrative_html.append(f'''<article class="narrative">
          <div class="narrative-head"><span class="number">{i:02d}</span><h3>{_esc(item.get('title'))}</h3><span class="badge { _esc(item.get('confidence','low')) }">{_esc(item.get('confidence','low')).upper()} confidence</span></div>
          <p class="hypothesis">{_esc(item.get('hypothesis'))}</p>
          <h4>Evidence supporting this hypothesis</h4>{''.join(cards) if cards else '<p>No valid evidence references were returned.</p>'}
          <h4>What remains uncertain</h4><p>{_esc(item.get('counter_evidence_or_unknowns'))}</p>
          <h4>Recommended checks</h4><ol>{checks}</ol>
        </article>''')
    if not narrative_html:
        narrative_html = ['<div class="empty">The model did not identify a sufficiently evidence-backed diagnostic narrative. Review evidence gaps and comparison coverage below.</div>']

    consistency = "".join(f"<li>{_esc(x)}</li>" for x in insights.get("consistency_questions", [])) or '<li>No consistency questions were identified.</li>'
    gaps = list(dict.fromkeys((package.get("limitations", []) or []) + (insights.get("evidence_gaps", []) or [])))
    gaps_html = "".join(f"<li>{_esc(x)}</li>" for x in gaps) or '<li>No explicit evidence gaps were identified; this does not guarantee complete evidence.</li>'
    evidence_rows = []
    for ev in package.get("evidence", []):
        facts = ev.get("facts", {})
        status = facts.get("status", facts.get("model_match", ""))
        evidence_rows.append(f'<tr><td>{_esc(ev.get("id"))}</td><td>{_esc(ev.get("title"))}</td><td>{_esc(ev.get("kind"))}</td><td>{_esc(status)}</td></tr>')
    evidence_table = "".join(evidence_rows) or '<tr><td colspan="4">No evidence records available.</td></tr>'
    generated = __import__("datetime").datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")

    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Engineering Insights</title><style>
:root{{--ink:#17212b;--muted:#5d6b78;--line:#dce3e9;--panel:#fff;--bg:#f3f6f9;--accent:#174d6b;--soft:#eaf2f7;}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.58 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}.page{{max-width:1120px;margin:28px auto;padding:0 22px 44px}}header{{background:linear-gradient(120deg,#12364c,#1d617b);color:white;padding:30px 34px;border-radius:16px;box-shadow:0 8px 24px #193b4d20}}.eyebrow{{text-transform:uppercase;letter-spacing:.14em;font-size:12px;opacity:.8}}h1{{margin:6px 0 4px;font-size:30px;line-height:1.2}}header p{{margin:5px 0;color:#e4eff5}}.meta{{font-size:12px;opacity:.82}}.callout{{margin:18px 0;padding:18px 22px;border:1px solid #d4e3eb;background:#fff;border-left:5px solid var(--accent);border-radius:10px}}.callout h2{{margin:0 0 8px;font-size:15px;text-transform:uppercase;letter-spacing:.07em;color:var(--accent)}}.callout p{{margin:0}}.pill{{display:inline-block;border-radius:999px;padding:5px 11px;background:#e6eef3;color:#173f54;font-size:12px;font-weight:700;margin-top:10px}}section{{margin-top:28px}}h2{{font-size:22px;margin:0 0 12px}}h3{{font-size:18px;margin:0;flex:1}}h4{{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:18px 0 7px}}.narrative{{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:21px 23px;margin:14px 0;box-shadow:0 2px 7px #17212b08;break-inside:avoid}}.narrative-head{{display:flex;align-items:center;gap:12px;flex-wrap:wrap}}.number{{font-size:12px;font-weight:800;color:var(--accent);background:var(--soft);border-radius:8px;padding:7px 9px}}.badge{{font-size:11px;letter-spacing:.04em;border-radius:999px;padding:4px 8px;font-weight:800}}.badge.high{{background:#d9f0e3;color:#17613b}}.badge.medium{{background:#fff0ce;color:#795000}}.badge.low{{background:#fbe3df;color:#8a3027}}.hypothesis{{font-size:16px;margin:15px 0 8px}}.evidence{{border:1px solid var(--line);border-radius:8px;margin:7px 0;background:#fbfcfd}}.evidence summary{{cursor:pointer;padding:9px 12px;font-size:13px;font-weight:650}}pre{{margin:0;padding:12px;white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6f8;border-top:1px solid var(--line);font-size:12px}}ol,ul{{padding-left:22px}}li{{margin:5px 0}}.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}.panel{{background:white;border:1px solid var(--line);border-radius:12px;padding:20px;break-inside:avoid}}.panel h2{{font-size:18px}}.empty{{padding:20px;background:white;border:1px dashed #b9c8d2;border-radius:10px;color:var(--muted)}}table{{border-collapse:collapse;width:100%;background:white;font-size:13px}}th,td{{padding:10px 12px;text-align:left;border-bottom:1px solid var(--line)}}th{{background:#edf3f6;font-size:11px;text-transform:uppercase;letter-spacing:.06em}}.foot{{font-size:12px;color:var(--muted);margin-top:28px;border-top:1px solid var(--line);padding-top:14px}}@media(max-width:720px){{.two-col{{grid-template-columns:1fr}}header{{padding:23px}}h1{{font-size:25px}}}}@media print{{body{{background:white;font-size:11pt}}.page{{max-width:none;margin:0;padding:0}}header{{background:#174d6b!important;-webkit-print-color-adjust:exact;print-color-adjust:exact;box-shadow:none}}.narrative,.panel,.callout{{box-shadow:none;break-inside:avoid}}details.evidence>pre{{display:block}}section{{break-inside:auto}}.narrative{{page-break-inside:avoid}}}}
</style></head><body><main class="page"><header><div class="eyebrow">Power-system comparison · diagnostic layer</div><h1>AI Engineering Insights</h1><p>Evidence-linked hypotheses and targeted troubleshooting checks</p><div class="meta">Generated { _esc(generated) } · Local model: {_esc(model_name)} · Numerical comparison statuses remain authoritative</div></header>
<div class="callout"><h2>Executive assessment</h2><p>{_esc(insights.get('executive_assessment'))}</p><span class="pill">{_esc(assessment_labels.get(insights.get('overall_assessment'), 'Insufficient evidence'))}</span></div>
<section><h2>Diagnostic narratives</h2><p style="color:var(--muted);margin-top:-5px">Prioritized engineering hypotheses. Each narrative links to the existing comparison evidence that motivated it; hypotheses are not confirmed causes.</p>{''.join(narrative_html)}</section>
<section class="two-col"><div class="panel"><h2>Consistency questions</h2><ul>{consistency}</ul></div><div class="panel"><h2>Evidence gaps and limitations</h2><ul>{gaps_html}</ul></div></section>
<section><h2>Evidence index</h2><p style="color:var(--muted)">The index lists evidence records available to the AI. Expand evidence references in each narrative to inspect relevant details.</p><div style="overflow-x:auto"><table><thead><tr><th>ID</th><th>Evidence</th><th>Type</th><th>Status</th></tr></thead><tbody>{evidence_table}</tbody></table></div></section>
<div class="foot"><strong>Interpretation guardrail:</strong> This report proposes hypotheses and investigation steps; it does not certify power-system safety, prove causation, or replace engineering review. Do not infer operating-limit violations unless the relevant limits and results are explicitly available. For a PDF, use the browser's Print → Save as PDF.</div></main></body></html>'''


def generate_ai_insight_report(report: Any, settings: Any, output_dir: str | Path) -> tuple[Path, Path]:
    """Generate HTML insight report and machine-readable JSON sidecar.

    Raises on LLM/service failures so the caller can log the failure while preserving
    the deterministic comparison report and workbook.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    package = build_evidence_package(report)
    insights = _call_local_model(package, settings)
    model_name = os.getenv("AI_INSIGHT_MODEL") or getattr(settings, "ai_insight_model", None) or "Qwen/Qwen3-14B"
    html_path = output_dir / "ai_engineering_insights.html"
    json_path = output_dir / "ai_engineering_insights.json"
    html_path.write_text(_render_html(insights, package, model_name), encoding="utf-8")
    json_path.write_text(json.dumps({"insights": insights, "evidence_package": package}, indent=2, ensure_ascii=False), encoding="utf-8")
    return html_path, json_path
