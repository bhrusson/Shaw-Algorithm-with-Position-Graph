import dill
from bqskit import Circuit
import matplotlib.pyplot as plt

circuit_lst = [
    "QAOA_16_compiled",
    "QuantumVolume_16",
    "QFT_16_compiled",
    "TFIM_n16_s100_compiled",
    "TFXY_n16_s100_compiled",
    "QAOA_20_compiled",
    "QuantumVolume_20",
    "QFT_20_compiled"
]

architecture_lst = [
    "H",
    "G2x3"
]

parameter_set = {
    "H": [["4", "5"], ["5", "6"]],
    "G2x3": [["3", "4"], ["4", "5"]],
}

import pickle


class DummyClass:
    def __init__(self, *args, **kwargs):
        pass


class IgnoringUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            # Return a dummy function or class to skip the problematic variable
            #print(f"Skipping problematic reference: {module}.{name}")
            return DummyClass  # Return a dummy class


"""
    SHAPER
"""

num_layout = 2
for circuit_idx in range(len(circuit_lst)):
    for architecture in architecture_lst:
        parameter = parameter_set[architecture][0] if circuit_idx < 5 else parameter_set[architecture][1]
        for param_idx in range(len(parameter)):
            param = parameter[param_idx]
            if param_idx == 1:
                continue
            with open(
                    f"bqskit/shuttling/qccd/new_result/SHAPER_{circuit_lst[circuit_idx]}_{architecture}_{param}_{num_layout}.pkl",
                    "rb") as input_file:
                data = IgnoringUnpickler(input_file).load()
                # print(f"Results of {circuit_lst[circuit_idx]}_{architecture}_{param}_{num_layout}")
                print(round(data[1], 2))
                #print(f"Runtime: {round(data[0] / 1e-6)} , Compile time: {round(data[1], 2)}")

"""
    QCCDSim
"""
# mapping = "Greedy"
# for circuit_idx in range(len(circuit_lst)):
#     for architecture in architecture_lst:
#         parameter = parameter_set[architecture][0] if circuit_idx < 5 else parameter_set[architecture][1]
#         for param_idx in range(len(parameter)):
#             param = parameter[param_idx]
#             if param_idx == 0 or circuit_idx == 4:
#                 continue
#             with open(
#                     f"bqskit/shuttling/qccd/new_result/QCCDSim_{circuit_lst[circuit_idx]}_{architecture}_{param}_{mapping}.pkl",
#                     "rb") as input_file:
#                 data = IgnoringUnpickler(input_file).load()
#                 print(f"Results of {circuit_lst[circuit_idx]}_{architecture}_{param}_{mapping}")
#                 print(f"Runtime: {round(data[1][0])} , Compile time: {round(data[0], 2)}")
