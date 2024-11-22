import matplotlib.pyplot as plt
import numpy as np

# Configuration Data
configurations_h = [
    ("QAOA16", 4), ("QAOA16", 5), ("QV16", 4), ("QV16", 5),
    ("QFT16", 4), ("QFT16", 5), ("TFIM_n16", 4), ("TFIM_n16", 5),
    ("TFXY_n16", 4), ("TFXY_n16", 5), ("QAOA20", 5), ("QAOA20", 6),
    ("QV20", 5), ("QV20", 6), ("QFT20", 5), ("QFT20", 6)
]

configurations_g2x3 = [
    ("QAOA16", 3), ("QAOA16", 4), ("QV16", 3), ("QV16", 4),
    ("QFT16", 3), ("QFT16", 4), ("TFIM_n16", 3), ("TFIM_n16", 4),
    ("TFXY_n16", 3), ("TFXY_n16", 4), ("QAOA20", 4), ("QAOA20", 5),
    ("QV20", 4), ("QV20", 5), ("QFT20", 4), ("QFT20", 5)
]

# Metric Data for SHAPER (in microseconds) and Variance (standard deviation) for H and G2x3 Architectures
shaper_data_h = [
    31730.2, 32930.2, 10685.6, 10093.2, 46407.4, 45327.4, 36623.6, 34929.8,
    41655.6, 39664.8, 50540.4, 51607.2, 15087.8, 14820.4, 64098.4, 68868
]
shaper_variance_h = [
    1556.58, 1738.39, 858.26, 1278.1, 3282.64, 1669.24, 183.19, 1963.2,
    1661.9, 1061.5, 2245, 1863.65, 819.98, 897.94, 2647.94, 3851.95
]

shaper_data_g2x3 = [
    35885.6, 34939.4, 10133, 10190.8, 49130.6, 45736, 38309.4, 36476.6,
    47026.2, 44769, 49964, 52225.2, 14247.4, 14502.8, 67417.2, 71231.4
]
shaper_variance_g2x3 = [
    985.65, 971.89, 732.49, 441.05, 3131.96, 2038.83, 1524.73, 1062.7,
    1084.89, 2596.35, 502.18, 2659.79, 876.97, 937.83, 2909.05, 1923.54
]

# Metric Data for QCCDSim (in microseconds), -1 indicates failure for H and G2x3 Architectures
qccdsim_data_h = [
    -1, 34790, -1, 11162, -1, 45488, -1, 36707,
    -1, 27135, -1, 55940, -1, 17674, -1, 56281
]

qccdsim_data_g2x3 = [
    -1, 39063, -1, 13535, -1, 36329, -1, 42251,
    -1, 45998, -1, 51266, -1, 15612, -1, 70134
]

# Plotting parameters
width = 0.35  # Width of the bars

# Function to plot data for an architecture
def plot_architecture(x, ions_labels, shaper_data, shaper_variance, qccdsim_data, title, filename):
    fig, ax = plt.subplots(figsize=(18, 10))

    # SHAPER Time Bars with Error Bars
    bars_shaper = ax.bar(x - width / 2, shaper_data, width, yerr=shaper_variance, capsize=5,
                         label='SHAPER Time (μs)', color='skyblue', alpha=0.8, error_kw={'elinewidth': 1.5})

    # Annotate SHAPER bars with their exact values right above the error bars
    for i, bar in enumerate(bars_shaper):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + shaper_variance[i] + 200,
                f"{int(bar.get_height())}", ha='center', va='bottom', fontsize=12, color='black')

    # QCCDSim Bars
    for i in range(len(x)):
        if qccdsim_data[i] != -1:
            # Plot regular bar if QCCDSim was successful
            ax.bar(x[i] + width / 2, qccdsim_data[i], width, label='QCCDSim Time (μs)' if i == 0 else "", color='orange')
        else:
            # Plot hatched bar if QCCDSim failed, same height as SHAPER bar
            ax.bar(x[i] + width / 2, shaper_data[i], width, label='QCCDSim Fail' if i == 0 else "",
                   color='gray', hatch='//', alpha=0.6)
            # Add a red cross to indicate failure right above the hatched bar
            ax.scatter(x[i] + width / 2, shaper_data[i] + 500, color='red', marker='x', s=100,
                       label='QCCDSim Fail' if i == 0 else "")

    # Set title and labels for the plot
    ax.set_title(title, fontsize=22, fontweight='bold')
    ax.set_ylabel('Execution Time (μs)', fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(ions_labels, rotation=45, ha='center', fontsize=16)

    ax.legend(fontsize=14)

    # Adjust layout for better visualization
    plt.tight_layout()
    # Save figure as a PDF file
    fig.savefig(filename, format='pdf')
    plt.close(fig)

# Plot for H Architecture
x_h = np.arange(len(configurations_h))  # X locations for H architecture configurations
ions_labels_h = [f"{config[1]} ions/trap" for config in configurations_h]
plot_architecture(x_h, ions_labels_h, shaper_data_h, shaper_variance_h, qccdsim_data_h, "H Architecture", "H_Architecture.pdf")

# Plot for G2x3 Architecture
x_g2x3 = np.arange(len(configurations_g2x3))  # X locations for G2x3 architecture configurations
ions_labels_g2x3 = [f"{config[1]} ions/trap" for config in configurations_g2x3]
plot_architecture(x_g2x3, ions_labels_g2x3, shaper_data_g2x3, shaper_variance_g2x3, qccdsim_data_g2x3, "G2x3 Architecture", "G2x3_Architecture.pdf")
