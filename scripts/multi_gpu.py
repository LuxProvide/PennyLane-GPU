from mpi4py import MPI
import pennylane as qml
import numpy as np
from timeit import default_timer as timer

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Set number of runs for timing averaging
num_runs = 3

# Choose number of qubits (wires) and circuit layers
n_wires = 34
n_layers = 2

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

# Set trainable parameters for calculating circuit Jacobian at the rank=0 process
if rank == 0:
    params = np.random.random(qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_wires))
else:
    params = None

# Broadcast the trainable parameters across MPI processes from rank=0 process
params = comm.bcast(params, root=0)

# Run, calculate the quantum circuit Jacobian and average the timing results
timing = []
for t in range(num_runs):
    start = timer()
    local_probs = circuit(params)
    end = timer()
    timing.append(end - start)

# MPI barrier to ensure all calculations are done
comm.Barrier()

if rank == 0:
    print("num_gpus: ", size, " wires: ", n_wires, " layers ", n_layers, " time: ", qml.numpy.mean(timing))
