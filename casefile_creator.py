# #######################################################################################
# Create powerflow case files in 2 formats, that can be used as inputs to the main engine
# #######################################################################################
from pathlib import Path
import pandapower as pp
from pypower.api import case300, runpf

from pypower.savecase import savecase, savemat


def generate_pandapower_case(filename):
    # Panda Power JSON format
    net = pp.networks.case9241pegase()
    pp.runpp(net)
    pp.to_json(net, filename + ".json")  # json format
    pp.to_excel(net, filename + ".xlsx")  # xlsx format
    return


def generate_matpower_case(filename, m_file_path=None):
    # MATPOWER .MAT format
    if m_file_path:
        import pandapower.converter as cv
        net = cv.from_mpc(m_file_path)
        mpc = cv.to_ppc(net, calculate_voltage_angles=True, init='flat')
    else:
        mpc = case9241pegase()
    results, success = runpf(mpc)
    savemat(filename + ".mat", results)  # .mat format
    savecase(filename, results)  #.py format
    return


if __name__ == "__main__":
    input_path = "IOfiles/case9241pegase"
    Path(input_path).mkdir(exist_ok=True)
    generate_pandapower_case(input_path + "/case_A")
    generate_matpower_case(input_path + "/case_B", "IOfiles/case8387pegase.m")
