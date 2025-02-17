import matplotlib.pyplot as plt
import numpy as np
import pickle

circuit_lst  = [
    "TFXY_n16_s100_compiled",
    "QFT_20_compiled"
]

architecture_lst = [
    "H",
    "G2x3"
]

parameter_set = {
    "TFXY_n16_s100_compiled_H": "5",
    "TFXY_n16_s100_compiled_G2x3": "4",
    "QFT_20_compiled_H": "6",
    "QFT_20_compiled_G2x3": "5",
}

layout_lst = ["4", "6", "8", "10"]

"""
    SHAPER
"""
mean_data = {}
std_data = {}
x_labels = []
for circuit in circuit_lst:
    for architecture in architecture_lst:
        parameter = parameter_set[f"{circuit}_{architecture}"]
        mean_data[f"{circuit}_{architecture}"] = []
        std_data[f"{circuit}_{architecture}"] = []
        x_labels.append(f"{circuit}_{architecture}")
        for layout in layout_lst:
            shuttling_time = []
            for index in range(1, 4):
                with open(
                        f"bqskit/shuttling/qccd/new_result/SHAPER_{circuit}_idx{index}_{architecture}_{parameter}_{layout}.pkl",
                        "rb") as input_file:
                    data = pickle.load(input_file)
                    shuttling_time.append(round(data[0] / 1e-6))
            print(f"SHAPER_{circuit}_{architecture}_num_layout:{layout}")
            print(f"AVG: {np.mean(shuttling_time)}")
            print(f"STD: {np.std(shuttling_time)}")
            mean_data[f"{circuit}_{architecture}"].append(np.mean(shuttling_time))
            std_data[f"{circuit}_{architecture}"].append(np.std(shuttling_time))

plt.figure(figsize=(10, 6), dpi=80)
for label in x_labels:
    plt.plot(layout_lst, mean_data[label], label=label)
    plt.fill_between(layout_lst, np.array(mean_data[label]) - np.array(std_data[label]), np.array(mean_data[label]) + np.array(std_data[label]), alpha=0.2)
plt.title("Average Performance by Layout")
plt.xlabel("Layout")
plt.ylabel("Average (AVG)")
plt.legend(title="Category")
plt.grid(True)
plt.tight_layout()
plt.show()