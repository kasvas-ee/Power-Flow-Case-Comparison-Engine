"""Source-specific extraction for the metadata-driven comparator.

The generic numerical comparator remains independent of PPC and pandapower.
This adapter intentionally does not perform row matching: canonicalization is
responsible for aligned ordering, and the existing Ybus gate is unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.comparison.metadata_comparator import MetricDefinition, SafeFormula, compare_scope_asset, group_definitions

from src.comparison.result_structures import TopSevereResult
from src.logger import logger

# MATPOWER/PYPOWER PPC column positions (kept local to avoid mutating PPC arrays).
PPC_COLUMNS = {
    "bus": {"BUS_I": 0, "BUS_TYPE": 1, "PD": 2, "QD": 3, "GS": 4,
            "BS": 5, "BUS_AREA": 6, "VM": 7, "VA": 8, "BASE_KV": 9,
            "ZONE": 10, "VMAX": 11, "VMIN": 12},
    "branch": {"F_BUS": 0, "T_BUS": 1, "BR_R": 2, "BR_X": 3,
               "BR_B": 4, "RATE_A": 5, "RATE_B": 6, "RATE_C": 7,
               "TAP": 8, "SHIFT": 9, "BR_STATUS": 10, "ANGMIN": 11,
               "ANGMAX": 12},
    "gen": {"GEN_BUS": 0, "PG": 1, "QG": 2, "QMAX": 3, "QMIN": 4,
            "VG": 5, "MBASE": 6, "GEN_STATUS": 7, "PMAX": 8, "PMIN": 9},
}

PPC_ASSET_TABLE = {
    "bus": "bus", "branch": "branch", "load": "bus",
    "generation": "gen", "grid": "gen",
}
NET_TABLES = {
    "bus": "res_bus", "load": "res_load", "generation": "res_gen",
    "branch": "res_impedance", "grid": "res_gen",
}

PARAMETER_ALIASES = {
    "vm_pu": "vm",
    "va_degree": "va",
    "p_mw": "p",
    "q_mvar": "q",
    "p_from_mw": "p_from",
    "q_from_mvar": "q_from",
    "p_to_mw": "p_to",
    "q_to_mvar": "q_to",
    "p_loss_mw": "p_loss",
    "q_loss_mvar": "q_loss",
    "p_loss_percent": "p_loss_pct",
    "q_loss_percent": "q_loss_pct",
    "s_from_mva": "s_from",
    "s_to_mva": "s_to",
}

FORMULA = SafeFormula()


def _net_table(net: Any, definition: MetricDefinition) -> pd.DataFrame:
    table_name = definition.asset_id
    # In state metadata, asset ID is the exact result-table identifier.
    table = net.get(table_name) if hasattr(net, "get") else None
    if not isinstance(table, pd.DataFrame):
        # Fallback by category for metadata where asset ID is a model table.
        table_name = NET_TABLES.get(definition.asset_name, table_name)
        table = net.get(table_name) if hasattr(net, "get") else None
    if not isinstance(table, pd.DataFrame):
        raise KeyError(f"pandapower table {table_name!r} is unavailable")
    return table


def _extract_direct(definition: MetricDefinition, case: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    source = definition.source
    asset = definition.asset_name
    if source == "ppc":
        ppc = case["ppc"]
        table_name = PPC_ASSET_TABLE.get(asset)
        if table_name is None:
            raise KeyError(f"No PPC table mapping for asset {asset!r}")
        table = ppc[table_name]
        column = PPC_COLUMNS.get(table_name, {}).get(definition.parameter_id)
        if column is None:
            raise KeyError(f"Unknown PPC {table_name} parameter {definition.parameter_id!r}")
        values = table[:, column]
        if asset == "load":
            # Load metrics are bus-level injections; include only buses with a load.
            mask = (ppc["bus"][:, PPC_COLUMNS["bus"]["PD"]] != 0) | (ppc["bus"][:, PPC_COLUMNS["bus"]["QD"]] != 0)
            values = values[mask]
        if asset == "grid":
            # A PPC gen row cannot be safely classified as ext_grid from PPC alone.
            # Avoid silently comparing all generators as grid assets.
            raise KeyError("PPC does not retain a reliable ext_grid-vs-gen row label")
        return np.asarray(values), np.asarray(values)
    if source == "net":
        table = _net_table(case["net"], definition)
        column = definition.parameter_id
        if column not in table.columns:
            raise KeyError(f"Column {column!r} is missing from table {definition.asset_id!r}")
        return table[column].to_numpy(copy=False), table.index.to_numpy(copy=False)
    raise KeyError(f"Unsupported direct source {source!r}")


def _asset_arrays(definitions, case):
    """Build formula namespace from columns in the corresponding asset table."""
    reqd_params = [definition.parameter_id for definition in definitions]
    first = definitions[0]
    if first.source in {"net", "derived"}:
        # Derived parameters share the asset table declared by asset ID.
        table = _net_table(case["net"], first)
        reqd_params = np.intersect1d(reqd_params, table.columns)  # to avoid derived params
        table = table[reqd_params]
        values = {str(c): table[c].to_numpy(copy=False) for c in table.columns}
        # DataFrame index is a stable reporting ID when canonicalization aligned rows.
        return values, np.asarray(list(range(len(table))))
    if first.source == "ppc":
        # a dict {k:v} of k: parameter (i.e.metric) and v: its values and a list of seq numbers as indices
        ppc = case["ppc"]
        asset_table_name = PPC_ASSET_TABLE.get(first.asset_name)
        if asset_table_name is None:
            raise KeyError(f"No PPC table mapping for asset {first.asset_name!r}")
        asset_table_contents = ppc[asset_table_name]
        ids = np.asarray(list(range(len(asset_table_contents))))  # TODO: Actual IDs to be implemented in the future
        reqd_params = [definition.parameter_id for definition in definitions]  # PPC_COLUMNS[asset_table_name]
        reqd_params = {k: PPC_COLUMNS[asset_table_name][k] for k in reqd_params if k in PPC_COLUMNS[asset_table_name]}
        values = {key: asset_table_contents[:, col] for key, col in reqd_params.items() if
                  col < asset_table_contents.shape[1]}
        return values, ids
    raise KeyError(f"Unsupported source {first.source!r}")


def rank_metadata(results, top_n):
    """Rank severe rows from metadata deltas; no comparison is repeated."""
    ranked = {}
    for item in results:
        delta = item.get("delta")
        severe_mask = item.get("severe_mask")
        if delta is None or severe_mask is None:
            continue
        indices = np.flatnonzero(severe_mask & np.isfinite(delta))
        if not len(indices):
            continue
        order = indices[np.argsort(np.abs(delta[indices]))[::-1][:top_n]]
        a = np.asarray(item.get("case_a")).reshape(-1)
        b = np.asarray(item.get("case_b")).reshape(-1)
        asset_name = str(item.get("asset_name", "Asset"))
        raw_parameter = str(item.get("parameter_id") or item.get("parameter_name")).lower()

        key = f"{asset_name}_{PARAMETER_ALIASES.get(raw_parameter, raw_parameter)}"
        ranked[key] = [TopSevereResult(rank=rank, asset=f"{asset_name}_{i}", value_a=float(a[i]), value_b=float(b[i]),
                                       delta=float(delta[i]), abs_delta=float(abs(delta[i]))
                                       ) for rank, i in enumerate(order, 1)]
    return ranked


def run_metadata_comparisons(
    cases,
    ppcs,
    metrics,
    scope,
    skip_assets={},
    report_all_mismatches=False,
    enable_multiprocessing=False
):
    """Run enabled metadata comparisons for one scope, preserving NOT PERFORMED rows."""
    definitions = [
        d for d in metrics
        if d.scope == scope
    ]

    # When reporting all mismatches, use tolerance as the severe threshold.
    if report_all_mismatches:
        from dataclasses import replace

        definitions = [
            replace(d, severe_threshold=d.tolerance)
            for d in definitions
        ]

    skipped_definitions = [
        d for d in definitions
        if d.asset_name.strip().lower() in skip_assets
    ]

    definitions = [
        d for d in definitions
        if d.asset_name.strip().lower() not in skip_assets
    ]

    # Set 'NOT PERFORMED' as status for Bus and Branch if skipped
    output = [
        {
            "scope": d.scope,
            "asset_name": d.asset_name,
            "asset_id": d.asset_id,
            "parameter_name": d.parameter_name,
            "parameter_id": d.parameter_id,
            "display_name": d.display_name,
            "unit": d.unit,
            "comparison": d.comparison,
            "tolerance": d.tolerance,
            "severe_threshold": d.severe_threshold,
            "status": "NOT PERFORMED",
            "reason": "Skipped because Ybus matched.",
            "case_a": None,
            "case_b": None,
            "delta": None,
            "statistics": {},
        }
        for d in skipped_definitions
    ]

    case_payloads = {
        "a": {"net": cases[0], "ppc": ppcs[0]},
        "b": {"net": cases[1], "ppc": ppcs[1]},
    }

    # In sequential run, group definitions by asset type and compute statistics for 1 group at a time
    groups = group_definitions(definitions)

    for (_scope, _asset), group in groups.items():
        try:
            table_values_a, ids_a = _asset_arrays(group, case_payloads["a"])
            table_values_b, ids_b = _asset_arrays(group, case_payloads["b"])
        except (KeyError, AttributeError, TypeError, ValueError) as exc:
            for definition in group:
                output.append({"scope": definition.scope, "asset_name": definition.asset_name,
                               "asset_id": definition.asset_id, "parameter_name": definition.parameter_name,
                               "parameter_id": definition.parameter_id, "display_name": definition.display_name,
                               "unit": definition.unit, "status": "NOT PERFORMED", "reason": str(exc),
                               "case_a": None, "case_b": None, "delta": None, "statistics": {}})
            continue

        ids = ids_a if len(ids_a) == len(ids_b) else None
        output.extend(compare_scope_asset(group, table_values_a, table_values_b, asset_ids=ids))

        # TODO: In parallel run, each metric is evaluated individually and metrics are distributed among workers equally
    return output
