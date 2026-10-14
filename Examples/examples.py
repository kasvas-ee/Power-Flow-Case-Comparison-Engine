"""Predefined scenarios that demonstrate matching and intentionally mismatched power-flow cases."""

from Examples.Main import Example
from functools import wraps


def print_summary(func):
    """Decorate an example runner to print its returned summary lines."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        """Run the wrapped example, print its summary, and return its result."""
        summary = func(*args, **kwargs)
        print("\n".join(summary))
        return summary

    return wrapper


class SillyExample:

    @staticmethod
    @print_summary
    def compare_diff_models():
        """Run an example in which the two input network models intentionally differ."""
        summary = Example.create("MismatchedModels")
        return summary


class Bus14Example:

    @staticmethod
    @print_summary
    def as_is():
        report = Example.create("IEEE14bus", reuse_inputs=False)
        return report

    @staticmethod
    @print_summary
    def output_mismatch():
        report = Example.create("IEEE14bus", reuse_inputs=False, scale_load_by=2.0)
        return report

    @staticmethod
    @print_summary
    def topology_mismatch():
        report = Example.create("IEEE14bus", reuse_inputs=False, create_topology_mismatch=True)
        return report

    @staticmethod
    @print_summary
    def count_mismatch():
        report = Example.create("IEEE14bus", reuse_inputs=False, drop_a_load=True)
        return report

    @staticmethod
    @print_summary
    def missing_metric():
        report = Example.create("IEEE14bus", reuse_inputs=False, drop_a_metric=True)
        return report


class Bus30Example:

    @staticmethod
    @print_summary
    def as_is():
        report = Example.create("IEEE30bus", reuse_inputs=False)
        return report


class Bus39Example:

    @staticmethod
    @print_summary
    def as_is():
        report = Example.create("IEEE39bus", reuse_inputs=False)
        return report
               
    @staticmethod
    @print_summary
    def loads_doubled():
        report = Example.create("IEEE39bus", reuse_inputs=False, scale_load_by=1.25)
        return report

    @staticmethod
    @print_summary
    def extreme_outliner():
        report = Example.create("IEEE39bus", reuse_inputs=False, scale_load_by=2, how_many_loads_in_percent=10)
        return report

    @staticmethod
    @print_summary
    def concentrated_mismatch():
        report = Example.create("IEEE39bus", reuse_inputs=False, scale_load_by=1.2, how_many_loads_in_percent=15)
        return report


class Bus57Example:

    @staticmethod
    @print_summary
    def as_is():
        report = Example.create("IEEE57bus", reuse_inputs=False)
        return report


class Bus118Example:

    @staticmethod
    @print_summary
    def as_is():
        report = Example.create("IEEE118bus", reuse_inputs=False)
        return report

    @staticmethod
    @print_summary
    def extreme_outliner():
        report = Example.create("IEEE118bus", reuse_inputs=False,
                                scale_load_by=1.2, how_many_loads_in_percent=5, scale_remaining_load_by=1.01)
        return report

    @staticmethod
    @print_summary
    def concentrated_mismatch():
        report = Example.create("IEEE118bus", reuse_inputs=False, scale_load_by=1.05, how_many_loads_in_percent=5)
        return report

    @staticmethod
    @print_summary
    def directional_pattern():
        report = Example.create("IEEE118bus", reuse_inputs=False,
                                scale_load_by=1.1, how_many_loads_in_percent=50, scale_remaining_load_by=0.9)
        return report


class Bus300Example:

    @staticmethod
    @print_summary
    def as_is():
        report = Example.create("IEEE300bus", reuse_inputs=False)  # , scale_impedance_by=1.1)  #, "C:/Users/KCH/PycharmProjects/Power-Flow-Case-Comparison-Engine/IOfiles/case6515rte.m")
        return report

    @staticmethod
    @print_summary
    def extreme_outliner():
        report = Example.create("IEEE300bus", reuse_inputs=False,
                                scale_load_by=1.1, how_many_loads_in_percent=10, scale_remaining_load_by=1.01)
        return report

    @staticmethod
    @print_summary
    def concentrated_mismatch():
        report = Example.create("IEEE300bus", reuse_inputs=False, scale_load_by=1.01, how_many_loads_in_percent=10)
        return report

    @staticmethod
    @print_summary
    def directional_pattern():
        report = Example.create("IEEE300bus", reuse_inputs=False,
                                scale_load_by=1.1, how_many_loads_in_percent=50, scale_remaining_load_by=0.9)
        return report


class Bus6515Example:

    @staticmethod
    def as_is():
        report = Example.create(
            "case6515rte", reuse_inputs=True,
            m_file_path="C:/Users/KCH/PycharmProjects/Power-Flow-Case-Comparison-Engine/IOfiles/case6515rte.m"
        )
        return report


class case1354pegase:

    @staticmethod
    @print_summary
    def as_is():
        report = Example.create(
            "case1354pegase", reuse_inputs=True,
            m_file_path="C:/Users/KCH/PycharmProjects/Power-Flow-Case-Comparison-Engine/IOfiles/case1354pegase.m"
        )
        return report


class case2869pegase:

    @staticmethod
    @print_summary
    def as_is():
        report = Example.create(
            "case2869pegase", reuse_inputs=True,
            m_file_path="C:/Users/KCH/PycharmProjects/Power-Flow-Case-Comparison-Engine/IOfiles/case2869pegase.m"
        )
        return report


class Bus9241Example:

    @staticmethod
    def as_is():
        report = Example.create("PEGASE9241bus", reuse_inputs=False,
                                m_file_path="C:/Users/KCH/PycharmProjects/Power-Flow-Case-Comparison-Engine/IOfiles/case9241pegase.m")
        return report


if __name__ == "__main__":

    # Pre-checks
    SillyExample.compare_diff_models()  # Compare 2 powerflow files generated from 2 completeld different models.
