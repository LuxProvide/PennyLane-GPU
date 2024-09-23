import matplotlib.pyplot as plt
import numpy as np

# Data (excluding qubit 20)
data = [
    # (layers, wires, time)
    (1, 21, 0.0183475938004752),
    (1, 22, 0.030220753358056147),
    (1, 23, 0.05632735012720028),
    (1, 24, 0.10702985773483913),
    (1, 25, 0.22521671800253293),
    (1, 26, 0.49934502869534),
    (1, 27, 1.0159491056886811),
    (1, 28, 2.0272764820450297),
    (1, 29, 3.9475580672733486),
    (1, 30, 9.322227556724101),
    (1, 31, 19.836258312687278),
    (1, 32, 46.61148637571993),
    (1, 33, 97.9333991643507),
    (2, 21, 0.02790863540334006),
    (2, 22, 0.0423265533366551),
    (2, 23, 0.06849264989917477),
    (2, 24, 0.12328315239089231),
    (2, 25, 0.25003725732676685),
    (2, 26, 0.5563214237336069),
    (2, 27, 1.108413304357479),
    (2, 28, 2.253672916054105),
    (2, 29, 4.373635108039404),
    (2, 30, 10.01874750937956),
    (2, 31, 21.278700340073556),
    (2, 32, 49.51534373802133),
    (2, 33, 104.63370479270816)
]

# Separate data for each layer
layer1_data = [(d[1], d[2]) for d in data if d[0] == 1]
layer2_data = [(d[1], d[2]) for d in data if d[0] == 2]

# Extract wires and times
wires1, times1 = zip(*layer1_data)
wires2, times2 = zip(*layer2_data)

# Set up the plot
fig, ax = plt.subplots(figsize=(16, 10))

# Set the width of each bar and positions
width = 0.4
x = np.arange(len(wires1))

# Create the bars
ax.bar(x - width/2, times1, width, label='1 Layer', alpha=0.8)
ax.bar(x + width/2, times2, width, label='2 Layers', alpha=0.8)

# Customize the plot
ax.set_ylabel('Time (seconds)', fontsize=16)
ax.set_xlabel('Number of Wires', fontsize=16)
ax.set_title('Execution Time vs. Number of Wires (16 GPUs)', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(wires1, rotation=45, ha='right')
ax.tick_params(axis='x', labelsize=16)  # Increase stick label (x-axis) font size
ax.legend(fontsize=12)

# Use logarithmic scale for y-axis due to large range of values
ax.set_yscale('log')

# Adjust layout and display the plot
plt.tight_layout()
plt.savefig('gpu_timing_layered_barplot_clean.png', dpi=300)
plt.close()

print("Clean layered bar plot saved as gpu_timing_layered_barplot_clean.png")
