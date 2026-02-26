import dill
import pickle
from bqskit import Circuit
import matplotlib.pyplot as plt
import numpy as np
circuit_lst = [
    #"QAOA_wsq_128_compiled",
    # "QuantumVolume_wsq_32",
    "QFT_wsq_128_compiled",
    #"TFIM_wsq_n64_s100_compiled",
    #"TFXY_wsq_n32_s100_compiled"
]

architecture_lst = [
    "grid"
]

parameter_set = {
    "grid": ["4"],
}
"""
    SHAPER
"""
num_layout = 2
for circuit_idx in range(len(circuit_lst)):
    print(f"Results of {circuit_lst[circuit_idx]}")
    for architecture in architecture_lst:
        print(f"Architecture: {architecture}")
        for param in parameter_set[architecture]:
            print(f"Param: {param}")
            for idx in range(1):
                with open(
                        f"bqskit/shuttling/qccd/paper_result_grid/SHAW_{circuit_lst[circuit_idx]}_idx{idx}_{architecture}_{param}_{num_layout}.pkl",
                        "rb") as input_file:
                    data = pickle.load(input_file)
                    print(round(data[1], 2))
                    print(f"Runtime: {round(data[0] / 1e-6)} , Compile time: {round(data[1], 2)}")

"""
    SHAW
"""
# num_layout = 2
# for circuit_idx in range(len(circuit_lst)):
#     print(f"Results of {circuit_lst[circuit_idx]}")
#     for architecture in architecture_lst:
#         print(f"Architecture: {architecture}")
#         for param in parameter_set[architecture]:
#             print(f"Param: {param}")
#             for idx in range(0, 3):
#                 with open(
#                         f"bqskit/shuttling/qccd/paper_result_grid/SHAW_{circuit_lst[circuit_idx]}_idx{idx}_{architecture}_{param}_{num_layout}.pkl",
#                         "rb") as input_file:
#                     data = pickle.load(input_file)
#
#                     print(round(data[1], 2))
#                     print(f"Runtime: {round(data[0] / 1e-6)} , Compile time: {round(data[1], 2)}")
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

# min_SHAPER = [31396, 33246, 8733, 10001, 42063, 41684, 33126, 34956, 39401, 38299, 50113, 50010, 13307, 12971, 65029, 68521]
# min_QCCDSim = [34790, 39063, 11162, 13535, 45488, 36329, 36707, 42251, 27135, 45998, 55940, 51266, 17674, 15612, 56281, 70134 ]
#
# print(f"Outperform by {np.max((np.array(min_QCCDSim) - np.array(min_SHAPER))/np.min(min_QCCDSim))}")