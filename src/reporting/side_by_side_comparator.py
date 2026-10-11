"""Optional metadata-driven Excel export for Case A / Case B comparisons.

The workbook consumes the comparison results already produced by the engine.
It does not execute a second numerical comparison. Asset-sheet parameters are
restricted to enabled metrics represented in the existing metadata results.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openpyxl.styles import PatternFill


SHEET_ORDER = [
    "ybus_comparison",
    "bus",
    "branch",
    "gen",
    "load",
    "res_bus",
    "res_branch",
    "res_load",
    "res_gen",
    "system_summary",
    "mismatch_details",
]

ASSET_SHEETS = (
    "bus", "branch", "gen", "load",
    "res_bus", "res_branch", "res_load", "res_gen",
)

# Comparison output asset names -> workbook sheet names.
_ASSET_SHEET = {
    ("model", "bus"): "bus",
    ("model", "branch"): "branch",
    ("model", "load"): "load",
    ("model", "generation"): "gen",
    ("model", "gen"): "gen",
    ("model", "grid"): "gen",
    ("state", "bus"): "res_bus",
    ("state", "branch"): "res_branch",
    ("state", "load"): "res_load",
    ("state", "generation"): "res_gen",
    ("state", "gen"): "res_gen",
    ("state", "grid"): "res_gen",
}


def _metadata_entries(report: Any, scope: str) -> list[dict[str, Any]]:
    if scope == "model":
        owner = getattr(report, "model_comparison", None)
        # In the comparison pipeline, model_comparison is an ActionResult;
        # the metadata list lives on its .result payload.
        owner = getattr(owner, "result", owner)
    else:
        owner = getattr(report, "power_flow_comparison", None)
    return list(getattr(owner, "metadata_comparisons", []) or [])


def _ybus_result(report: Any) -> Any:
    """Support both a raw YbusComparisonResult and an ActionResult wrapper."""
    value = getattr(report, "ybus_comparison", None)
    return getattr(value, "result", value)


def _ybus_matches(report: Any) -> bool:
    result = _ybus_result(report)
    return bool(getattr(result, "is_match", False))


def _is_skipped_bus_branch_model(entry: dict[str, Any], report: Any) -> bool:
    """Exclude bus/branch model rows when Ybus equality supersedes them."""
    if not _ybus_matches(report) or str(entry.get("scope", "")).lower() != "model":
        return False
    asset = str(entry.get("asset_name", "")).strip().lower()
    # A matching Ybus is the primary model-equivalence test. Do not duplicate
    # bus/branch model checks in the workbook when those checks are skipped.
    return asset in {"bus", "branch"}


def _scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _equal(a: Any, b: Any) -> bool:
    try:
        if pd.isna(a) and pd.isna(b):
            return True
    except (TypeError, ValueError):
        pass
    try:
        result = a == b
        return bool(result) if np.ndim(result) == 0 else bool(np.all(result))
    except (TypeError, ValueError):
        return False


def _as_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        return np.asarray(value).reshape(-1)
    except (TypeError, ValueError):
        return None


def _entry_ids(entry: dict[str, Any], length: int) -> np.ndarray:
    ids = _as_array(entry.get("asset_ids"))
    if ids is None or len(ids) != length:
        return np.arange(length, dtype=object)
    return ids


def _entry_delta(entry: dict[str, Any], a: Any, b: Any, index: int) -> Any:
    """Use the engine's delta; exact metrics use an equality indicator."""
    delta = _as_array(entry.get("delta"))
    if delta is not None and index < len(delta):
        return _scalar(delta[index])
    if str(entry.get("comparison", "")).lower() == "exact":
        return 0 if _equal(a, b) else "DIFFERENT"
    try:
        return abs(float(a) - float(b))
    except (TypeError, ValueError):
        return None


def _entry_mismatch_masks(entry: dict[str, Any], length: int) -> tuple[np.ndarray, np.ndarray]:
    """Return mismatch/severe masks, deriving them from engine deltas if needed."""
    mismatch = _as_array(entry.get("mismatch_mask"))
    severe = _as_array(entry.get("severe_mask"))
    a_values = _as_array(entry.get("case_a"))
    b_values = _as_array(entry.get("case_b"))
    comparison = str(entry.get("comparison", "")).strip().lower()

    if mismatch is None or len(mismatch) != length:
        mismatch = np.zeros(length, dtype=bool)
        if a_values is not None and b_values is not None:
            tolerance = _number_or_none(entry.get("tolerance"))
            for i in range(min(length, len(a_values), len(b_values))):
                if comparison == "exact":
                    mismatch[i] = not _equal(a_values[i], b_values[i])
                else:
                    delta = _entry_delta(entry, a_values[i], b_values[i], i)
                    if isinstance(delta, (int, float, np.number)) and np.isfinite(delta):
                        mismatch[i] = tolerance is not None and float(delta) > tolerance
                    else:
                        mismatch[i] = not _equal(a_values[i], b_values[i])
    else:
        mismatch = mismatch.astype(bool)

    if severe is None or len(severe) != length:
        severe = np.zeros(length, dtype=bool)
        threshold = _number_or_none(entry.get("severe_threshold"))
        if threshold is not None and a_values is not None and b_values is not None:
            for i in range(min(length, len(a_values), len(b_values))):
                delta = _entry_delta(entry, a_values[i], b_values[i], i)
                severe[i] = (
                    isinstance(delta, (int, float, np.number))
                    and np.isfinite(delta)
                    and float(delta) > threshold
                )
    else:
        severe = severe.astype(bool)
    return mismatch, severe


def _number_or_none(value: Any) -> float | None:
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _delta_severity_lookup(report: Any) -> dict[tuple[str, str, str], str]:
    """Map (sheet, asset ID, delta header) to mismatch severity."""
    lookup: dict[tuple[str, str, str], str] = {}
    entries = _metadata_entries(report, "model") + _metadata_entries(report, "state")
    for entry in entries:
        if _is_skipped_bus_branch_model(entry, report):
            continue
        if str(entry.get("status", "")).strip().upper() == "NOT PERFORMED":
            continue
        scope = str(entry.get("scope", "")).strip().lower()
        asset = str(entry.get("asset_name", "")).strip().lower()
        sheet = _ASSET_SHEET.get((scope, asset))
        if sheet is None:
            continue
        a_values = _as_array(entry.get("case_a"))
        b_values = _as_array(entry.get("case_b"))
        if a_values is None or b_values is None:
            continue
        length = min(len(a_values), len(b_values))
        ids = _entry_ids(entry, length)
        mismatch, severe = _entry_mismatch_masks(entry, length)
        display = str(entry.get("display_name") or entry.get("parameter_name") or entry.get("parameter_id") or "Metric")
        unit = str(entry.get("unit") or "").strip()
        header = f"{display} ({unit})" if unit else display
        for i in range(length):
            if severe[i]:
                level = "severe"
            elif mismatch[i]:
                level = "mismatch"
            else:
                continue
            lookup[(sheet, str(_scalar(ids[i])), f"Δ {header}")] = level
    return lookup


def _asset_frames(report: Any) -> dict[str, pd.DataFrame]:
    """Build side-by-side sheets from enabled metadata comparison outputs only."""
    rows_by_sheet: dict[str, dict[str, dict[str, Any]]] = {
        name: {} for name in ASSET_SHEETS
    }
    entries = _metadata_entries(report, "model") + _metadata_entries(report, "state")

    for entry in entries:
        scope = str(entry.get("scope", "")).strip().lower()
        asset = str(entry.get("asset_name", "")).strip().lower()
        sheet = _ASSET_SHEET.get((scope, asset))
        if sheet is None or _is_skipped_bus_branch_model(entry, report):
            continue

        status = str(entry.get("status", "")).strip().upper()
        a_values, b_values = _as_array(entry.get("case_a")), _as_array(entry.get("case_b"))
        # Do not show parameters that were configured but never actually compared.
        if status == "NOT PERFORMED" or a_values is None or b_values is None:
            continue

        length = min(len(a_values), len(b_values))
        ids = _entry_ids(entry, length)
        display = str(entry.get("display_name") or entry.get("parameter_name") or entry.get("parameter_id") or "Metric")
        unit = str(entry.get("unit") or "").strip()
        header = f"{display} ({unit})" if unit else display

        for i in range(length):
            asset_id = _scalar(ids[i])
            key = str(asset_id)
            row = rows_by_sheet[sheet].setdefault(key, {"Asset ID": asset_id})
            row[f"{header} A"] = _scalar(a_values[i])
            row[f"{header} B"] = _scalar(b_values[i])
            row[f"Δ {header}"] = _entry_delta(entry, a_values[i], b_values[i], i)

    frames = {}
    for sheet, keyed_rows in rows_by_sheet.items():
        rows = list(keyed_rows.values())
        frames[sheet] = pd.DataFrame(rows)
        if frames[sheet].empty:
            frames[sheet] = pd.DataFrame(columns=["Asset ID"])
    return frames


def _mismatch_details(report: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    entries = _metadata_entries(report, "model") + _metadata_entries(report, "state")
    for entry in entries:
        if _is_skipped_bus_branch_model(entry, report):
            continue
        if str(entry.get("status", "")).strip().upper() == "NOT PERFORMED":
            continue
        a_values, b_values = _as_array(entry.get("case_a")), _as_array(entry.get("case_b"))
        if a_values is None or b_values is None:
            continue
        length = min(len(a_values), len(b_values))
        mismatch, severe = _entry_mismatch_masks(entry, length)
        ids = _entry_ids(entry, length)
        display = entry.get("display_name") or entry.get("parameter_name") or entry.get("parameter_id")
        for i in np.flatnonzero(mismatch[:length]):
            rows.append({
                "Scope": entry.get("scope"),
                "Asset Type": entry.get("asset_name"),
                "Asset ID": _scalar(ids[i]),
                "Parameter": display,
                "Parameter ID": entry.get("parameter_id"),
                "Case A": _scalar(a_values[i]),
                "Case B": _scalar(b_values[i]),
                "Delta": _entry_delta(entry, a_values[i], b_values[i], i),
                "Unit": entry.get("unit"),
                "Tolerance": entry.get("tolerance"),
                "Severe Threshold": entry.get("severe_threshold"),
                "Severity": "Severe" if severe[i] else "Mismatch",
            })
    columns = [
        "Scope", "Asset Type", "Asset ID", "Parameter", "Parameter ID",
        "Case A", "Case B", "Delta", "Unit", "Tolerance",
        "Severe Threshold", "Severity",
    ]
    return pd.DataFrame(rows, columns=columns)


def _ybus_summary(report: Any) -> pd.DataFrame:
    result = _ybus_result(report)
    if result is None:
        return pd.DataFrame([{"Check": "Ybus comparison", "Result": "NOT PERFORMED", "Details": "No Ybus comparison result available"}])

    def fmt_shape(value: Any) -> Any:
        if value is None:
            return None
        try:
            return " × ".join(str(int(part)) for part in value)
        except (TypeError, ValueError):
            return str(value)

    overall = bool(getattr(result, "is_match", False))
    rows = [
        {"Check": "Overall Ybus match", "Result": "MATCH" if overall else "MISMATCH", "Details": getattr(result, "reason", None)},
        {"Check": "Bus count / matrix shape match", "Result": "MATCH" if getattr(result, "bus_count_match", False) else "MISMATCH", "Details": f"Case A: {fmt_shape(getattr(result, 'shape_a', None))}; Case B: {fmt_shape(getattr(result, 'shape_b', None))}"},
        {"Check": "Topology / sparsity pattern match", "Result": "MATCH" if getattr(result, "topology_match", False) else "MISMATCH", "Details": f"Pattern differences: {getattr(result, 'pattern_diff_count', None)}"},
        {"Check": "Ybus numerical values match", "Result": "MATCH" if getattr(result, "ybus_value_match", False) else "MISMATCH", "Details": f"Value differences: {getattr(result, 'value_diff_count', None)}; max value difference: {getattr(result, 'max_value_diff', None)}"},
    ]
    return pd.DataFrame(rows, columns=["Check", "Result", "Details"])


def _system_summary(report: Any) -> pd.DataFrame:
    rows = [
        {"Metric": "Ybus match", "Case A": None, "Case B": None, "Delta": None,
         "Status": "MATCH" if _ybus_matches(report) else "MISMATCH"},
        {"Metric": "Model match", "Case A": None, "Case B": None, "Delta": None,
         "Status": "MATCH" if getattr(report, "model_match", False) else "MISMATCH"},
        {"Metric": "Results comparable", "Case A": None, "Case B": None, "Delta": None,
         "Status": str(getattr(report, "results_comparable", None))},
        {"Metric": "Results match", "Case A": None, "Case B": None, "Delta": None,
         "Status": str(getattr(report, "results_match", None))},
    ]
    summary = getattr(getattr(report, "power_flow_comparison", None), "system", None)
    if summary is not None:
        for field, value in vars(summary).items():
            if field.endswith("_a_mw") or field.endswith("_a_mvar"):
                suffix = "mw" if field.endswith("_a_mw") else "mvar"
                base = field[:-5] if suffix == "mw" else field[:-7]
                unit = "MW" if suffix == "mw" else "MVAr"
                case_b = getattr(summary, f"{base}_b_{suffix}", None)
                delta = getattr(summary, f"delta_{base}_{suffix}", None)
                rows.append({"Metric": f"{base.upper()} ({unit})", "Case A": value,
                             "Case B": case_b, "Delta": delta,
                             "Status": "AVAILABLE" if value is not None and case_b is not None else "NOT PERFORMED"})
    return pd.DataFrame(rows, columns=["Metric", "Case A", "Case B", "Delta", "Status"])


def _format_sheet(writer: pd.ExcelWriter, name: str, frame: pd.DataFrame) -> None:
    frame.to_excel(writer, sheet_name=name, index=False)
    ws = writer.sheets[name]
    ws.freeze_panes = "A2"
    if ws.max_row >= 1 and ws.max_column >= 1:
        ws.auto_filter.ref = ws.dimensions
    for cells in ws.columns:
        width = min(max(max(len(str(cell.value or "")) for cell in cells) + 2, 12), 42)
        ws.column_dimensions[cells[0].column_letter].width = width


def _apply_delta_highlights(writer: pd.ExcelWriter, frames: dict[str, pd.DataFrame], report: Any) -> None:
    """Color individual delta cells: yellow for mismatch, red for severe."""
    yellow = PatternFill(fill_type="solid", fgColor="FFFF00")
    red = PatternFill(fill_type="solid", fgColor="FF0000")
    severity_lookup = _delta_severity_lookup(report)

    # Side-by-side asset sheets: severity is derived from each enabled metadata metric.
    for sheet in ASSET_SHEETS:
        frame = frames.get(sheet)
        if frame is None or frame.empty or "Asset ID" not in frame.columns:
            continue
        ws = writer.sheets[sheet]
        header_to_col = {str(cell.value): cell.column for cell in ws[1]}
        for row_number, asset_id in enumerate(frame["Asset ID"].tolist(), start=2):
            asset_key = str(_scalar(asset_id))
            for header, col_number in header_to_col.items():
                if not header.startswith("Δ "):
                    continue
                level = severity_lookup.get((sheet, asset_key, header))
                if level == "severe":
                    ws.cell(row=row_number, column=col_number).fill = red
                elif level == "mismatch":
                    ws.cell(row=row_number, column=col_number).fill = yellow

    # Mismatch details has one row per mismatching metric; color its Delta cell too.
    details = frames.get("mismatch_details")
    if details is not None and not details.empty and "Severity" in details.columns:
        ws = writer.sheets["mismatch_details"]
        headers = {str(cell.value): cell.column for cell in ws[1]}
        delta_col = headers.get("Delta")
        severity_col = headers.get("Severity")
        if delta_col and severity_col:
            for row_number in range(2, ws.max_row + 1):
                severity = str(ws.cell(row=row_number, column=severity_col).value or "").strip().lower()
                if severity == "severe":
                    ws.cell(row=row_number, column=delta_col).fill = red
                elif severity == "mismatch":
                    ws.cell(row=row_number, column=delta_col).fill = yellow


def export_side_by_side_comparator(
    canonical_cases: tuple[Any, Any], report: Any, path: str | Path
) -> Path:
    """Export the asset sheets, Ybus summary, system summary, and mismatch details."""
    if len(canonical_cases) != 2:
        raise ValueError("Expected canonical Case A and Case B networks")

    frames = _asset_frames(report)
    frames["system_summary"] = _system_summary(report)
    frames["ybus_comparison"] = _ybus_summary(report)
    frames["mismatch_details"] = _mismatch_details(report)

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet in SHEET_ORDER:
            _format_sheet(writer, sheet, frames.get(sheet, pd.DataFrame()))
        _apply_delta_highlights(writer, frames, report)
    return output
