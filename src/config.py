"""Load and validate engine configuration and metadata-driven metric definitions."""

from __future__ import annotations

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Any
import os
from dataclasses import dataclass, InitVar
from src.comparison.result_structures import MetricDefinition
from src.logger import logger

REQUIRED_COLUMNS = {
    "scope", "source", "asset name", "asset ID", "parameter name",
    "parameter ID", "enabled", "comparison", "tolerance",
    "severe_threshold", "unit", "display_name",
}
VALID_SCOPES = {"model", "state"}
VALID_COMPARISONS = {"exact", "absolute", "angle"}


def process_config(config_file: str):
    """
    Read the settings.yaml file
    While reading, validate if the settings provided are of valid datatypes
    """
    try:
        root_dir = Path(__file__).resolve().parents[1]
        config_file = root_dir / config_file
        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)

        config_obj = Config(config_dict=config_dict, root_dir=root_dir)
        config_obj.validate_config()

        # Create the output folder
        config_obj.output_path.mkdir(exist_ok=True)

        return config_obj
    except Exception as e:
        msg = f"Error while loading settings from {config_file}: {e}"
        logger.error(msg)
        raise RuntimeError(msg)


def _parse_bool(value: Any) -> bool:
    """Normalize supported boolean representations and reject unrecognized values."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def load_metric_definitions(path: str | Path) -> list[MetricDefinition]:
    """Load and validate the CSV. Disabled rows are intentionally discarded."""
    # Metric metadata is loaded once and normalized before it is used by comparison adapters.
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Metrics CSV is missing columns: {sorted(missing)}")
    definitions: list[MetricDefinition] = []
    for row_number, row in enumerate(frame.to_dict(orient="records"), start=2):
        try:
            enabled = _parse_bool(row["enabled"])
            if not enabled:
                continue
            scope = str(row["scope"]).strip().lower()
            source = str(row["source"]).strip().lower()
            comparison = str(row["comparison"]).strip().lower()
            tolerance = float(row["tolerance"])
            severe = None if pd.isna(row["severe_threshold"]) else float(row["severe_threshold"])
            if scope not in VALID_SCOPES:
                raise ValueError(f"scope must be one of {sorted(VALID_SCOPES)}")
            if comparison not in VALID_COMPARISONS:
                raise ValueError(f"comparison must be one of {sorted(VALID_COMPARISONS)}")
            if not np.isfinite(tolerance) or tolerance < 0:
                raise ValueError("tolerance must be finite and non-negative")
            if (
                    (scope == "model" and severe is None) or
                    (severe is not None and (not np.isfinite(severe) or severe < tolerance))
            ):
                # Force-correct rather than accepting contradictory thresholds.
                severe = tolerance
            definitions.append(MetricDefinition(
                scope=scope, source=source,
                asset_name=str(row["asset name"]).strip(),
                asset_id=str(row["asset ID"]).strip(),
                parameter_name=str(row["parameter name"]).strip(),
                parameter_id=str(row["parameter ID"]).strip(),
                enabled=True, comparison=comparison, tolerance=tolerance,
                severe_threshold=severe,
                unit=None if pd.isna(row["unit"]) else str(row["unit"]).strip(),
                display_name=str(row["display_name"]).strip(),
                match_strategy=(None if "match_strategy" not in row or pd.isna(row.get("match_strategy"))
                                else str(row.get("match_strategy")).strip()),
            ))
        except Exception as exc:
            raise ValueError(f"Invalid metrics CSV row {row_number}: {exc}") from exc
    return definitions


def check_for_error(config, *keys, expected_type=None, allow_empty=False):
    """
    Fetch the config setting and validate it against the expected datatype
    The setting can be blank only when allow_empty is True
    """
    param_name = ".".join(keys)

    try:
        value = config
        for key in keys:
            value = value[key]

        if value is None:
            if not allow_empty:
                msg = f"Configuration setting {param_name} cannot be empty"
                logger.error(msg)
                raise ValueError(msg)
            else:
                return None

        if expected_type is not None and not isinstance(value, expected_type):
            msg = f"Expected {expected_type.__name__}, got {type(value).__name__} for " \
                  f"configuration setting {param_name}."
            logger.error(msg)
            raise TypeError(msg)

        return value

    except Exception as e:
        msg = f"Configuration setting {param_name} is invalid: {e}"
        logger.error(msg)
        raise RuntimeError(msg)


@dataclass
class Config:
    config_dict: InitVar[dict]
    root_dir: InitVar[Path]

    def __post_init__(self, config_dict: dict, root_dir: Path):
        """
        Transfer the settings from dict into this object
        """
        self.case_a_path = root_dir / check_for_error(config_dict, "Case_IO", "CaseA_path", expected_type=str)
        self.case_a_format = check_for_error(config_dict, "Case_IO", "CaseA_format", expected_type=str).lower()
        self.case_b_path = root_dir / check_for_error(config_dict, "Case_IO", "CaseB_path", expected_type=str)
        self.case_b_format = check_for_error(config_dict, "Case_IO", "CaseB_format", expected_type=str).lower()
        self.output_path = root_dir / check_for_error(config_dict, "Case_IO", "Output_path", expected_type=str)

        # Preferences
        self.export_net = check_for_error(config_dict, "Preferences", "export_canonical_format", expected_type=bool)
        self.generate_side_by_side_comparator = check_for_error(config_dict, "Preferences",
                                                                "generate_side-by-side_comparator", expected_type=bool)
        self.enable_multiprocessing = check_for_error(config_dict, "Preferences", "enable_multiprocessing",
                                                      expected_type=bool)
        self.report_all_mismatches = check_for_error(config_dict, "Preferences", "report_all_mismatches", expected_type=bool)
        self.y_pu_tolerance = float(check_for_error(config_dict, "Report", "y_pu_tolerance", expected_type=str))
        if self.report_all_mismatches:
            # Report
            self.top_n_severe = 999999999
        else:
            # Report
            self.top_n_severe = check_for_error(config_dict, "Report", "top_n_severe", expected_type=int)

        # Termination clauses
        self.abort_if_network_mismatch = check_for_error(config_dict, "Termination_clauses",
                                                         "abort_if_network_mismatch", expected_type=bool)
        self.abort_at_first_mismatch = check_for_error(config_dict, "Termination_clauses", "abort_at_first_mismatch",
                                                       expected_type=bool)

        # Optional local-LLM insight report. No remote provider is required.
        ai_settings = config_dict.get("AI_Insights", {}) or {}
        self.generate_ai_insight_report = bool(ai_settings.get("enabled", False))
        self.ai_insight_base_url = str(ai_settings.get("base_url", "http://localhost:8000/v1"))
        self.ai_insight_model = str(ai_settings.get("model", "Qwen/Qwen3-14B"))
        self.ai_insight_timeout_seconds = float(ai_settings.get("timeout_seconds", 180))
        self.ai_insight_max_tokens = int(ai_settings.get("max_tokens", 2200))

        # METRIC DEFINITIONS
        metrics_path = Path(__file__).resolve().parents[1] / "config/metrics.csv"
        if metrics_path.exists():
            self.metric_definitions = load_metric_definitions(metrics_path)
        else:
            msg = "Metrics definitions file not found at {}".format(metrics_path)
            logger.error(msg)
            raise FileNotFoundError(msg)

    def validate_config(self):
        """Validate configuration settings and metric definitions before comparison begins."""
        valid_formats = ["matpower", "pandapower"]
        if self.case_a_format not in valid_formats:
            msg = "Case A has invalid format."
            logger.error(msg)
            raise IOError(msg)

        if self.case_b_format not in valid_formats:
            msg = "Case B has invalid format."
            logger.error(msg)
            raise IOError(msg)

        if not os.path.exists(self.case_a_path):
            msg = "Case A path is invalid."
            logger.error(msg)
            raise IOError(msg)

        if not os.path.exists(self.case_b_path):
            msg = "Case B path is invalid."
            logger.error(msg)
            raise IOError(msg)
