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
n_train = 16    # Size of the train dataset
n_test = 16     # Size of the test dataset

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

from mpi4py import MPI
from pennylane import numpy as np
from timeit import default_timer as timer

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def quanv(image, circuit):
    """Convolves the input image with many applications of the same quantum circuit."""
    out = np.zeros((14, 14, 4))
    for j in range(0, 28, 2):
        for k in range(0, 28, 2):
            q_results = circuit(image[j:j+2, k:k+2, 0].flatten())
            out[j // 2, k // 2] = q_results
    return out

def q_process(images, circuit, device_name):
    """Quantum pre-processing of images."""
    q_images = []
    for idx, img in enumerate(images):
        q_images.append(quanv(img, circuit))
    return np.array(q_images)

def create_circuit(dev, num_wires, num_layers):
    rand_params = np.random.uniform(0, 2 * np.pi, (num_layers, num_wires))

    @qml.qnode(dev)
    def circuit(phi):
        for j, p in enumerate(phi):
            qml.RY(np.pi * p, wires=j % num_wires)
        qml.RandomLayers(rand_params, wires=list(range(num_wires)))
        return [qml.expval(qml.PauliZ(j % num_wires)) for j in range(4)]

    return circuit

def run_experiment(wires, layers, num_runs, train_images, rank):
    dev_gpu = qml.device('lightning.gpu', wires=wires, mpi=True)

    circuit_gpu = create_circuit(dev_gpu, wires, layers)

    def time_process(process_func):
        timings = []
        for _ in range(num_runs):
            start = timer()
            process_func()
            timings.append(timer() - start)
        return np.mean(timings)

    # Distribute images across MPI processes
    images_per_process = len(train_images) // size
    local_images = train_images[rank * images_per_process:(rank + 1) * images_per_process]

    gpu_time = time_process(lambda: q_process(local_images, circuit_gpu, f"lightning.gpu rank {rank}"))

    print(f"Rank {rank} completed: Wires={wires}, Layers={layers}, GPU time={gpu_time:.4f}s")
    return gpu_time

# Parameters to test
wires_range = [20, 25, 30, 32, 33]
layers_range = [3]
num_runs = 1

# Assuming train_images is available for all processes
# Load or broadcast train_images across all MPI ranks here (not shown for brevity)

timing_results = {
    'gpu': np.zeros((len(wires_range), len(layers_range)))
}

# Run simulations for each combination of parameters in parallel
for i, wires in enumerate(wires_range):
    for j, layers in enumerate(layers_range):
        gpu_time = run_experiment(wires, layers, num_runs, train_images, rank)
        # Gather results from all ranks
        all_gpu_times = comm.gather(gpu_time, root=0)

        if rank == 0:
            timing_results['gpu'][i, j] = np.mean(all_gpu_times)
            print(f"Average GPU time for Wires={wires}, Layers={layers}: {timing_results['gpu'][i, j]:.4f}s")


# Create a plot for the GPU timing across different wires
def plot_timing_results(wires_range, timing_results, output_filename):
    fig, ax = plt.subplots()

    # Plot GPU times for each layer (assuming one layer in this example)
    ax.plot(wires_range, timing_results['gpu'][:, 0], marker='o', label=f'Layers={layers_range[0]}')

    # Set plot labels and title
    ax.set_xlabel('Number of Wires (Qubits)')
    ax.set_ylabel('GPU Time (s)')
    ax.set_title('GPU Timing for Different Numbers of Qubits (Wires)')
    ax.grid(True)
    ax.legend()

    # Save plot to file
    plt.savefig(output_filename)
    plt.show()

# Example usage of the function
output_file = 'multi_gpu_timing_plot.png'
plot_timing_results(wires_range, timing_results, output_file)

output_file  # Returning file path to access the image
