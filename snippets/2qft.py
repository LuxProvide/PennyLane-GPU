@qml.qnode(dev)
def two_qubit_qft():
    # Apply a Hadamard gate to the first qubit
    qml.Hadamard(wires=0)
    
    # Apply a controlled-Phase gate (CRZ) between qubits with phase π/2
    qml.CRZ(np.pi/2, wires=[0, 1])
    
    # Apply a Hadamard gate to the second qubit
    qml.Hadamard(wires=1)
    
    # SWAP the two qubits
    qml.SWAP(wires=[0, 1])
    
    return qml.state()

# Draw the circuit
fig, ax = qml.draw_mpl(swap_gate_circuit)()
plt.show()