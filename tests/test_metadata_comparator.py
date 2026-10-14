import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.comparison.metadata_comparator import (
    MetricDefinition, SafeFormula, compare_arrays, load_metric_definitions,
)


def definition(comparison='absolute', tol=0.1, severe=1.0):
    return MetricDefinition('state', 'net', 'bus', 'res_bus', 'vm_pu', 'vm_pu',
                            True, comparison, tol, severe, 'pu', 'Voltage', None)


def test_absolute_vectorized_and_statistics():
    result = compare_arrays(definition(), np.array([1., 2., 3.]), np.array([1., 2.3, 5.]))
    assert result['status'] == 'SEVERE MISMATCH'
    assert result['statistics']['mismatch_count'] == 2
    assert result['statistics']['severe_mismatch_count'] == 1
    assert np.allclose(result['delta'], [0., .3, 2.])


def test_angle_wraparound():
    result = compare_arrays(definition('angle', tol=2.1, severe=5.), [179.], [-179.])
    assert result['status'] == 'MATCH'
    assert result['delta'][0] == pytest.approx(2.0)


def test_exact():
    result = compare_arrays(definition('exact', 0, None), [1, 2], [1, 3])
    assert result['status'] == 'MISMATCH'
    assert result['statistics']['mismatch_count'] == 1


def test_safe_formula_and_rejects_attribute_access():
    formula = SafeFormula()
    result = formula.evaluate('100*p_mw/sum(p_mw)', {'p_mw': np.array([1., 3.])})
    assert np.allclose(result, [25., 75.])
    with pytest.raises(Exception):
        formula.evaluate('__import__("os")', {})


def test_metadata_loader_filters_disabled_and_corrects_severe_threshold(tmp_path):
    p = tmp_path / 'metrics.csv'
    pd.DataFrame([{
        'scope':'state','source':'net','asset name':'bus','asset ID':'res_bus',
        'parameter name':'vm_pu','parameter ID':'vm_pu','enabled':True,
        'comparison':'absolute','tolerance':0.2,'severe_threshold':0.1,
        'unit':'pu','display_name':'Voltage','match_strategy':'bus_id'}, {
        'scope':'state','source':'net','asset name':'bus','asset ID':'res_bus',
        'parameter name':'va_degree','parameter ID':'va_degree','enabled':False,
        'comparison':'angle','tolerance':0.1,'severe_threshold':1,
        'unit':'degree','display_name':'Angle','match_strategy':'bus_id'}]).to_csv(p,index=False)
    defs = load_metric_definitions(p)
    assert len(defs) == 1
    assert defs[0].severe_threshold == 0.2


def test_metadata_loader_allows_removed_match_strategy_column(tmp_path):
    p = tmp_path / 'metrics.csv'
    pd.DataFrame([{
        'scope': 'model', 'source': 'ppc', 'asset name': 'bus', 'asset ID': 'bus',
        'parameter name': 'bus_id', 'parameter ID': 'BUS_I', 'enabled': True,
        'comparison': 'exact', 'tolerance': 0, 'severe_threshold': None,
        'unit': 'bus_id', 'display_name': 'Bus ID',
    }]).to_csv(p, index=False)
    definitions = load_metric_definitions(p)
    assert len(definitions) == 1
    assert definitions[0].match_strategy is None


def test_state_metadata_adapter_runs_net_and_derived_metrics(tmp_path):
    from src.comparison.metadata_adapters import run_metadata_comparisons

    metrics = pd.DataFrame([
        {'scope': 'state', 'source': 'net', 'asset name': 'bus', 'asset ID': 'res_bus',
         'parameter name': 'vm_pu', 'parameter ID': 'vm_pu', 'enabled': True,
         'comparison': 'absolute', 'tolerance': 0.01, 'severe_threshold': 0.1,
         'unit': 'pu', 'display_name': 'Bus Voltage Magnitude'},
        {'scope': 'state', 'source': 'derived', 'asset name': 'generation', 'asset ID': 'res_gen',
         'parameter name': 'p_share', 'parameter ID': '100*p_mw/sum(p_mw)', 'enabled': True,
         'comparison': 'absolute', 'tolerance': 0.01, 'severe_threshold': 50.0,
         'unit': '%', 'display_name': 'Generation Share'},
    ])
    path = tmp_path / 'metrics.csv'
    definitions = load_metric_definitions(path)
    metrics.to_csv(path, index=False)

    net_a = {
        'res_bus': pd.DataFrame({'vm_pu': [1.0, 0.99]}, index=[10, 20]),
        'res_gen': pd.DataFrame({'p_mw': [25.0, 75.0]}, index=[0, 1]),
    }
    net_b = {
        'res_bus': pd.DataFrame({'vm_pu': [1.0, 0.98]}, index=[10, 20]),
        'res_gen': pd.DataFrame({'p_mw': [50.0, 50.0]}, index=[0, 1]),
    }
    results = run_metadata_comparisons((net_a, net_b), (None, None), definitions, scope='state')
    by_name = {r['parameter_name']: r for r in results}
    assert by_name['vm_pu']['status'] == 'MISMATCH'
    assert by_name['vm_pu']['statistics']['mismatch_count'] == 1
    assert by_name['p_share']['status'] == 'MISMATCH'
    assert np.allclose(by_name['p_share']['delta'], [25.0, 25.0])
