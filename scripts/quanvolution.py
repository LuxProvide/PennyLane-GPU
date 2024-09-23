import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import time
import os

##############################################################################
# Setting of the main hyper-parameters of the model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

n_epochs = 30   # Number of optimization epochs
n_layers = 1    # Number of random layers
n_train = 50    # Size of the train dataset
n_test = 30     # Size of the test dataset

SAVE_PATH = "../save/quanvolution/"  # Data saving folder
PREPROCESS = True           # If False, skip quantum processing and load data from SAVE_PATH
np.random.seed(0)           # Seed for NumPy random number generator
tf.random.set_seed(0)       # Seed for TensorFlow random number generator

##############################################################################
# Loading of the MNIST dataset
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We import the MNIST dataset from *Keras*. To speedup the evaluation of this demo
# we use only a small number of training and test images. Obviously, better
# results are achievable when using the full dataset.

mnist_dataset = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist_dataset.load_data()

# Reduce dataset size
train_images = train_images[:n_train]
train_labels = train_labels[:n_train]
test_images = test_images[:n_test]
test_labels = test_labels[:n_test]

# Normalize pixel values within 0 and 1
train_images = train_images / 255
test_images = test_images / 255

# Add extra dimension for convolution channels
train_images = np.array(train_images[..., tf.newaxis], requires_grad=False)
test_images = np.array(test_images[..., tf.newaxis], requires_grad=False)

import numpy as np
from timeit import default_timer as timer

def quanv(image, circuit):
    """Convolves the input image with many applications of the same quantum circuit."""
    out = np.zeros((14, 14, 4))
    for j in range(0, 28, 2):
        for k in range(0, 28, 2):
            q_results = circuit(image[j:j+2, k:k+2, 0].flatten())
            out[j // 2, k // 2] = q_results
    return out

def q_process(images, circuit, device_name):
    print(f"Quantum pre-processing of images on {device_name} device:")
    q_images = []
    for idx, img in enumerate(images):
        print(f"{idx + 1}/{len(images)}        ", end="\r")
        q_images.append(quanv(img, circuit))
    return np.array(q_images)

def create_circuit(dev, num_wires, num_layers):
#    rand_params = np.random.uniform(0, 2 * np.pi, (num_layers, num_wires))
    rand_params = np.random.uniform(high=2 * np.pi, size=(num_layers, num_wires))

    @qml.qnode(dev)
    def circuit(phi):
        for j, p in enumerate(phi):
            qml.RY(np.pi * p, wires=j % num_wires)
        qml.RandomLayers(rand_params, wires=list(range(num_wires)))
        return [qml.expval(qml.PauliZ(j % num_wires)) for j in range(4)]
    return circuit

def run_experiment(wires, layers, num_runs):
    dev_gpu = qml.device('lightning.gpu', wires=wires)
    dev_cpu = qml.device('lightning.qubit', wires=wires)

    circuit_cpu = create_circuit(dev_cpu, wires, layers)
    circuit_gpu = create_circuit(dev_gpu, wires, layers)

    def time_process(process_func):
        timings = []
        for _ in range(num_runs):
            start = timer()
            process_func()
            timings.append(timer() - start)
        return np.mean(timings)

    cpu_time = time_process(lambda: q_process(train_images, circuit_cpu, "lightning.qubit"))
    gpu_time = time_process(lambda: q_process(train_images, circuit_gpu, "lightning.gpu"))

    print(f"Completed: Wires={wires}, Layers={layers}, CPU time={cpu_time:.4f}s, GPU time={gpu_time:.4f}s")
    return cpu_time, gpu_time

# Parameters to test
wires_range = [1, 2, 4, 5, 10, 15, 18, 20]
layers_range = [3]
num_runs = 1

# Dictionary to store timing results
timing_results = {
    'cpu': np.zeros((len(wires_range), len(layers_range))),
    'gpu': np.zeros((len(wires_range), len(layers_range)))
}

# Run simulations for each combination of parameters
for i, wires in enumerate(wires_range):
    for j, layers in enumerate(layers_range):
        cpu_time, gpu_time = run_experiment(wires, layers, num_runs)
        timing_results['cpu'][i, j] = cpu_time
        timing_results['gpu'][i, j] = gpu_time


import matplotlib.pyplot as plt

for j, layers in enumerate(layers_range):
    fig, ax = plt.subplots(figsize=(12, 8))

    width = 0.35  # Width of the bars
    x = np.arange(len(wires_range))  # X-axis positions

    # Create bar plots for CPU and GPU side by side
    ax.bar(x - width/2, timing_results['cpu'][:, j], width, label='CPU')
    ax.bar(x + width/2, timing_results['gpu'][:, j], width, label='GPU', hatch='//')

    # Formatting the plot
    ax.set_xlabel('Number of Wires (Qubits)')
    ax.set_ylabel('Average Time (seconds)')
    ax.set_title(f'CPU vs (one)GPU Timing Results (Layers={layers})')
    ax.set_xticks(x)
    ax.set_xticklabels(wires_range)
    ax.legend()

    # Optionally, uncomment the following line for logarithmic scale
    # ax.set_yscale('log')

    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot to a file
    filename = f'cpu_vs_gpu_timing_layers_{layers}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free up memory

    print(f"Plot saved as {filename}")
