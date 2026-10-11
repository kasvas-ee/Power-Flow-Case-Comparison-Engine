import yaml
from pathlib import Path
import pandapower as pp
from pypower.api import runpf, case14, case30, case118, case300
from pandapower.networks.power_system_test_cases import case6515rte
from pypower.savecase import savecase, savemat

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
    "IEEE118bus": {
        "pandapower": pp.networks.case118,
        "matpower": case118,
    },
    "IEEE300bus": {
        "pandapower": pp.networks.case300,
        "matpower": case300,
    },
    "case6515rte": {
        "pandapower": pp.networks.case6515rte,
        "matpower": case6515rte,
    },
}


def create_case(example_id: str, library):
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
    net = create_case(example_name, "pandapower")

    # Modify network (if required)
    if "scale_load_by" in options:
        factor = options["scale_load_by"]
        net.load["p_mw"] *= factor
        net.load["q_mvar"] *= factor

    if "create_topology_mismatch" in options:
        if options["create_topology_mismatch"]:
            net.line.drop(net.line.index[-1], inplace=True)

    if "scale_impedance_by" in options:
        factor = options["scale_impedance_by"]
        net.line["r_ohm_per_km"] *= factor
        net.line["x_ohm_per_km"] *= factor

    # Solve
    try:
        pp.runpp(net)
    except pp.LoadflowNotConverged:
        msg = f"ERROR: Power flow did not converge for '{example_name}' in PandaPower."
        raise RuntimeError(msg)

    pp.to_json(net, filename + ".json")  # json format
    pp.to_excel(net, filename + ".xlsx")  # xlsx format
    return


def generate_matpower_case(filename, example_name, m_file_path=None):
    # MATPOWER .MAT format
    if m_file_path:
        import pandapower.converter as cv
        net = cv.from_mpc(m_file_path)
        mpc = cv.to_ppc(net, calculate_voltage_angles=True, init='flat')
    else:
        mpc = create_case(example_name, "matpower")
    results, success = runpf(mpc)
    if not success:
        msg = f"ERROR: Power flow did not converge for '{example_name}' in Matpower."
        raise RuntimeError(msg)

    savemat(filename + ".mat", results)  # .mat format
    savecase(filename, results)  #.py format
    return


def create_inputs(input_path, example_name, m_file_path=None, **options):
    Path(ROOT / input_path).mkdir(parents=True, exist_ok=True)
    generate_matpower_case(str(ROOT / input_path) + "/case_A", example_name, m_file_path)
    generate_pandapower_case(str(ROOT / input_path) + "/case_B", example_name, **options)
    return


def create_config(config_file, input_dir, output_dir):
    Path(ROOT / output_dir).mkdir(parents=True, exist_ok=True)
    with open( ROOT / "config/settings.yaml", "r") as f:
        ref_config_dict = yaml.safe_load(f)
        ref_config_dict["Case_IO"]["CaseA_path"] = input_dir + "/case_A.mat"
        ref_config_dict["Case_IO"]["CaseB_path"] = input_dir + "/case_B.json"
        ref_config_dict["Case_IO"]["Output_path"] = output_dir

    with open(ROOT / config_file, "w") as f:
        yaml.safe_dump(ref_config_dict, f, sort_keys=False, indent=2)
    return
