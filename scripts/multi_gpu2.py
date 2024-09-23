from mpi4py import MPI
import pennylane as qml
import numpy as np
from timeit import default_timer as timer

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Set number of runs for timing averaging
num_runs = 3

# Choose number of qubits (wires)
n_wires = 34

# Set range of layers to test
layer_range = range(1, 4)  # 1 to 3 inclusive

# Instantiate CPU (lightning.qubit) or GPU (lightning.gpu) device.
# mpi=True to switch on distributed simulation
dev = qml.device('lightning.gpu', wires=n_wires, mpi=True)

# Set target wires for probability calculation
prob_wires = range(n_wires)

# Create QNode of device and circuit
@qml.qnode(dev)
def circuit(weights):
    qml.StronglyEntanglingLayers(weights, wires=list(range(n_wires)))
    return qml.probs(wires=prob_wires)

# Function to run circuit for a given number of layers
def run_circuit(n_layers):
    # Set trainable parameters for calculating circuit at the rank=0 process
    if rank == 0:
        params = np.random.random(qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_wires))
    else:
        params = None

    # Broadcast the trainable parameters across MPI processes from rank=0 process
    params = comm.bcast(params, root=0)

    # Run, calculate the quantum circuit probabilities and average the timing results
    timing = []
    for _ in range(num_runs):
        start = timer()
        local_probs = circuit(params)
        end = timer()
        timing.append(end - start)

    return np.mean(timing)

# Run the circuit for each number of layers in the range
results = []
for n_layers in layer_range:
    time = run_circuit(n_layers)

    # MPI barrier to ensure all calculations are done
    comm.Barrier()

    if rank == 0:
        results.append((n_layers, time))
        print(f"num_gpus: {size}, wires: {n_wires}, layers: {n_layers}, time: {time}")

# Print summary of results
if rank == 0:
    print("\nSummary of results:")
    for n_layers, time in results:
        print(f"Layers: {n_layers}, Average time: {time}")
