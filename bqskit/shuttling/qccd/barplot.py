import matplotlib.pyplot as plt
import numpy as np
circuits = [
    "QAOA_16_compiled",
    "QuantumVolume_16",
    "QFT_16_compiled",
    "TFIM_n16_s100_compiled",
    "TFXY_n16_s100_compiled",
    "QAOA_20_compiled",
    "QuantumVolume_20",
    "QFT_20_compiled"
]



# Updated data with variance (error bars)
shaper_means = [31730.2, 10685.6, 46407.4, 36623.6, 41655.6, 50540.4, 15087.8, 64098.4]
shaper_errors = [1556.58, 858.26, 3282.64, 183.19, 1661.9, 2245, 819.98, 2647.94]

qccdsim_means = [34790, 11162, 45488, 36707, 27135, 55940, 17674, 56281]

x = np.arange(len(circuits))  # The label locations
width = 0.35  # Width of the bars

# Plotting with error bars
fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, shaper_means, width, yerr=shaper_errors, label="SHAPER (µs)", capsize=5, alpha=0.7)
bars2 = ax.bar(x + width/2, qccdsim_means, width, label="QCCDSim (µs)", capsize=5, alpha=0.7)

# Adding labels, title, and customizing axes
ax.set_xlabel("Circuits", fontsize=12)
ax.set_ylabel("Time (µs)", fontsize=12)
ax.set_title("Comparison of SHAPER and QCCDSim Shuttling Times on H-architecture", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(circuits, rotation=45, ha="right")
ax.legend()
ax.grid(True)
# Adding value annotations on bars
for bar, mean, error in zip(bars1, shaper_means, shaper_errors):
    height = bar.get_height()
    ax.annotate(f'{int(mean)}', xy=(bar.get_x() + bar.get_width()/2, height + error + 5),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

# Final adjustments
fig.tight_layout()
plt.show()