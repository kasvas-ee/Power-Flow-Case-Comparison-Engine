"""Shared helpers for generating paired power-system cases and example-specific configuration files."""

import yaml
from pathlib import Path
import pandapower as pp
from pypower.api import runpf, ppoption, case14, case30, case39, case57, case118, case300
from pandapower.networks.power_system_test_cases import case1354pegase, case2869pegase, case6515rte, case9241pegase
from pypower.savecase import savecase, savemat
from src.logger import logger

ROOT = Path(__file__).parent.parent

CASES = {
    "IEEE14bus": {
        "pandapower": pp.networks.case14,
        "matpower": case14,
    },
    "IEEE30bus": {
        "pandapower": pp.networks.case30,
        "matpower": case30,
    },
    "IEEE39bus": {
        "pandapower": pp.networks.case39,
        "matpower": case39,
    },
    "IEEE57bus": {
        "pandapower": pp.networks.case57,
        "matpower": case57,
    },
    "IEEE118bus": {
        "pandapower": pp.networks.case118,
        "matpower": case118,
    },
    "IEEE300bus": {
        "pandapower": pp.networks.case300,
        "matpower": case300,
    },
    "case1354pegase": {
        "pandapower": pp.networks.case1354pegase,
        "matpower": case1354pegase,
    },
    "case2869pegase": {
        "pandapower": pp.networks.case2869pegase,
        "matpower": case2869pegase,
    },
    "case6515rte": {
        "pandapower": pp.networks.case6515rte,
        "matpower": case6515rte,
    },
    "PEGASE9241bus": {
        "pandapower": pp.networks.case9241pegase,
        "matpower": case9241pegase,
    },
    "MismatchedCases": {
            "pandapower": None,
            "matpower": None,
        },
}


def create_case(example_id: str, library):
    """Instantiate a named example case using the requested network library."""
    try:
        return CASES[example_id][library]()
    except KeyError:
        available = ", ".join(sorted(CASES))
        raise ValueError(
            f"Unknown example '{example_id}'. "
            f"Available examples: {available}"
        )


def generate_pandapower_case(filename, example_name, **options):
    # Panda Power JSON format
    """Generate and serialize a pandapower case with the requested scenario changes."""
    net = create_case(example_name, "pandapower")

    # Modify network (if required)
    if "scale_load_by" in options:
        factor = options["scale_load_by"]
        share = options.get("how_many_loads_in_percent", 100)
        remaining_factor = options.get("scale_remaining_load_by", 1)
        n_loads = round(len(net.load) * share / 100)
        selected_loads = net.load.index[:n_loads]  # np.random.choice(net.load.index, size=n_loads, replace=False)
        remaining_loads = net.load.index[n_loads:]
        net.load.loc[selected_loads, ["p_mw", "q_mvar"]] *= factor
        net.load.loc[remaining_loads, ["p_mw", "q_mvar"]] *= remaining_factor
        logger.info("Scaled {}% loads (both P & Q) by {}.".format(share, factor))

    if "create_topology_mismatch" in options:
        if options["create_topology_mismatch"]:
            lines = net.line
            candidates = lines.index[lines.in_service].to_numpy()

            pair = None

            for i, line_a in enumerate(candidates):
                a_from = lines.at[line_a, "from_bus"]
                a_to = lines.at[line_a, "to_bus"]

                for line_b in candidates[i + 1:]:
                    b_from = lines.at[line_b, "from_bus"]
                    b_to = lines.at[line_b, "to_bus"]

                    if len({a_from, a_to, b_from, b_to}) == 4:
                        pair = (line_a, line_b)
                        break

                if pair is not None:
                    break

            if pair is None:
                raise ValueError("No valid pair of lines found.")

            line_a, line_b = pair

            # Swap the to_bus endpoints
            lines.at[line_a, "to_bus"], lines.at[line_b, "to_bus"] = (
                lines.at[line_b, "to_bus"],
                lines.at[line_a, "to_bus"],
            )

            logger.info(f"Swapped to_bus values for lines {line_a} and {line_b}.")

    if "scale_impedance_by" in options:
        factor = options["scale_impedance_by"]
        net.line["r_ohm_per_km"] *= factor
        net.line["x_ohm_per_km"] *= factor

    if "drop_a_load" in options:
        net.load.drop(net.load.index[-1], inplace=True)
        logger.info(f"Dropped the last row from load data table.")


    # Solve
    try:
        pp.runpp(net)
    except pp.LoadflowNotConverged:
        msg = f"ERROR: Power flow did not converge for '{example_name}' in PandaPower."
        raise RuntimeError(msg)

    if "drop_a_metric" in options:
        net.res_load.drop(columns=["p_mw"], inplace=True)
        logger.info(f"Dropped the 'p_mw' metric from load results table.")

    pp.to_json(net, filename + ".json")  # json format
    pp.to_excel(net, filename + ".xlsx")  # xlsx format
    return


def generate_matpower_case(filename, example_name, m_file_path=None):
    """Generate and save the MATPOWER representation of the selected example case."""
    if m_file_path:
        import pandapower.converter as cv
        net = cv.from_mpc(m_file_path)
        mpc = cv.to_ppc(net, calculate_voltage_angles=True, init='flat')
    else:
        mpc = create_case(example_name, "matpower")
    results, success = runpf(mpc, ppoption(VERBOSE=0, OUT_ALL=0))
    if not success:
        msg = f"ERROR: Power flow did not converge for '{example_name}' in Matpower."
        raise RuntimeError(msg)

    savemat(filename + ".mat", results)  # .mat format
    savecase(filename, results)  #.py format
    return


def create_inputs(input_path, example_name, m_file_path=None, **options):
    """Create the paired Case A and Case B input files for an example."""
    Path(ROOT / input_path).mkdir(parents=True, exist_ok=True)
    generate_matpower_case(str(ROOT / input_path) + "/case_A", example_name, m_file_path)
    generate_pandapower_case(str(ROOT / input_path) + "/case_B", example_name, **options)
    return


def create_config(config_file, input_dir, output_dir):
    """Write an example-specific configuration based on the shared settings."""
    Path(ROOT / output_dir).mkdir(parents=True, exist_ok=True)
    with open( ROOT / "config/settings.yaml", "r") as f:
        ref_config_dict = yaml.safe_load(f)
        ref_config_dict["Case_IO"]["CaseA_path"] = input_dir + "/case_A.mat"
        ref_config_dict["Case_IO"]["CaseB_path"] = input_dir + "/case_B.json"
        ref_config_dict["Case_IO"]["Output_path"] = output_dir

    with open(ROOT / config_file, "w") as f:
        yaml.safe_dump(ref_config_dict, f, sort_keys=False, indent=2)
    return
