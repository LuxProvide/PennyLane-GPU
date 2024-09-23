import matplotlib.pyplot as plt
import numpy as np

# Data
data = [
    # (num_gpus, layers, time)
    (16, 1, 205.30952749797143),
    (16, 2, 217.25349353502193),
    (16, 3, 231.05748629996864),
    (32, 1, 100.87877867626958),
    (32, 2, 109.72431400992598),
    (32, 3, 118.13628906236652),
    (64, 1, 49.170140151089676),
    (64, 2, 54.901664013043046),
    (64, 3, 60.49359157968623),
    (128, 1, 22.41683646532086),
    (128, 2, 25.42828756896779),
    (128, 3, 28.74829689001975)
]

# Separate data for each layer
layer_1_data = [(d[0], d[2]) for d in data if d[1] == 1]
layer_2_data = [(d[0], d[2]) for d in data if d[1] == 2]
layer_3_data = [(d[0], d[2]) for d in data if d[1] == 3]

# Extract GPUs and times
gpus, times_1 = zip(*layer_1_data)
_, times_2 = zip(*layer_2_data)
_, times_3 = zip(*layer_3_data)

# Set up the plot
fig, ax = plt.subplots(figsize=(12, 8))

# Set the width of each bar and positions
width = 0.25
x = np.arange(len(gpus))

# Define colors for colorblind-friendly palette
colors = ['#0072B2', '#D55E00', '#009E73']  # Blue, Red, Green

# Create the bars with specific colors
ax.bar(x - width, times_1, width, label='1 Layer', color=colors[0], alpha=0.8)
ax.bar(x, times_2, width, label='2 Layers', color=colors[1], alpha=0.8)
ax.bar(x + width, times_3, width, label='3 Layers', color=colors[2], alpha=0.8)

# Add lines for perfect scaling and actual scaling
# Assuming perfect scaling is inversely proportional to the number of GPUs (for 1 Layer baseline)
perfect_scaling = [times_1[0] * (gpus[0] / g) for g in gpus]
ax.plot(x, perfect_scaling, label='Perfect Scaling', linestyle='--', color='black', marker='o')

# Actual scaling for 1 Layer
ax.plot(x, times_1, label='Actual Scaling (1 Layer)', linestyle='-', color='blue', marker='x')

# Customize the plot
ax.set_ylabel('Time (seconds)', fontsize=16)
ax.set_xlabel('Number of GPUs', fontsize=14)
ax.set_title('Execution Time vs. Number of GPUs and Layers (34 Wires)', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(gpus)
ax.legend(fontsize=12)

# Increase tick label font size
ax.tick_params(axis='both', which='major', labelsize=12)

# Add number of nodes as a secondary x-axis
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(x)
ax2.set_xticklabels([f"{gpu // 4}" for gpu in gpus])
ax2.set_xlabel("Number of Nodes", fontsize=14)
ax2.tick_params(axis='x', which='major', labelsize=12)

# Adjust layout and display the plot
plt.tight_layout()
plt.savefig('gpu_scaling_layered_barplot_with_scaling.png', dpi=300)
plt.close()

print("GPU scaling layered bar plot with scaling saved as gpu_scaling_layered_barplot_with_scaling.png")
