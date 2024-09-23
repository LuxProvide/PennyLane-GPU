import matplotlib.pyplot as plt
import re
import numpy as np

# Read the data from the file
with open('gpu_timing_data.txt', 'r') as f:
    lines = f.readlines()

# Extract GPU counts and timing data
gpu_counts = []
times = []
for line in lines:
    match = re.search(r'num_gpus:\s+(\d+).*time:\s+(\d+\.\d+)', line)
    if match:
        gpu_counts.append(int(match.group(1)))
        times.append(float(match.group(2)))

# Create the plot
plt.figure(figsize=(14, 8))

# Use range(len(gpu_counts)) for x-axis to ensure equal spacing
x = range(len(gpu_counts))
bar_width = 0.8  # Increased bar width for thicker bars
bars = plt.bar(x, times, width=bar_width, align='center')

# Customize the plot
plt.xlabel('Number of GPUs', fontsize=14)
plt.ylabel('Time (seconds)', fontsize=14)
plt.title('GPU Count vs. Execution Time', fontsize=16)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Set x-axis ticks and labels
plt.xticks(x, gpu_counts, fontsize=12)

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}',
             ha='center', va='bottom', fontsize=10)

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig('gpu_timing_barplot_thick.png', dpi=300)
plt.close()

print("Thick bar plot saved as gpu_timing_barplot_thick.png")
