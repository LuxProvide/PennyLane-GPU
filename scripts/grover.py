import pennylane as qml
from pennylane import numpy as np

# Define the values of the properties
property_values = [5, 9, 3, 7]  # Example values for properties P1, P2, P3, P4
target_sum = sum(property_values) / 2  # We want the values to be equally divided

# Define the number of qubits and wires (2^4 = 16 combinations)
n_qubits = 4
dev = qml.device('default.qubit', wires=n_qubits)

# Define the oracle to check for a fair distribution of property values
def oracle():
    for i in range(16):
        binary_str = format(i, '04b')  # Binary representation of the number
        set_a = [property_values[j] for j in range(4) if binary_str[j] == '1']
        set_b = [property_values[j] for j in range(4) if binary_str[j] == '0']

        if sum(set_a) == sum(set_b):
            qml.PauliX(wires=0)  # Mark the solution

# Define the Grover diffusion operator
def diffusion_operator():
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
        qml.PauliX(wires=i)
    qml.MultiControlledX(wires=list(range(n_qubits)))
    for i in range(n_qubits):
        qml.PauliX(wires=i)
        qml.Hadamard(wires=i)

# Define the full Grover algorithm circuit
@qml.qnode(dev)
def grover_circuit():
    # Step 1: Apply Hadamard to all qubits to create equal superposition
    for i in range(n_qubits):
        qml.Hadamard(wires=i)

    # Step 2: Apply Grover's algorithm for sqrt(N) iterations
    num_iterations = int(np.pi / 4 * np.sqrt(16))  # sqrt(16) = 4

    for _ in range(num_iterations):
        oracle()  # Apply the oracle to mark the solutions
        diffusion_operator()  # Apply the diffusion operator

    return qml.probs(wires=list(range(n_qubits)))

# Run the Grover circuit
probabilities = grover_circuit()
print(f"Probabilities of each combination: {probabilities}")

