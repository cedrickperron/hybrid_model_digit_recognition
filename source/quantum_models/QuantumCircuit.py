import pandas as pd
import tensorflow as tf
import numpy as np
import random

import pennylane as qml
import pennylane.numpy as npp
import matplotlib.pyplot as plt


# Add a term to ignore a specific warning
import warnings  # Import the warnings module
#warnings.filterwarnings("ignore", message="Contains tensors of types {'autograd', 'tensorflow'}")
warnings.filterwarnings("ignore")


## DEFINE THE QUANTUM CIRCUIT
# ------------------------------------------------------- #
class BaseQuantumCircuit:
    def __init__(self, n_layers, n_wires):
        self.n_layers = n_layers
        self.n_wires = n_wires
        self.dev = qml.device("default.qubit", wires=range(n_wires))

    def state_preparation(self, x):
        """Encode the classical data into a quantum state."""
        qml.AngleEmbedding(x, wires=range(self.n_wires))

    def create_qnode(self):
        @qml.qnode(self.dev, diff_method="backprop", interface="tf")
        def qnode(inputs, params):
            return self.quantum_circuit(inputs, params)
        return qnode

    def draw_circuit(self, x, params, plot_name):
        qnode = self.create_qnode()
        fig, ax = qml.draw_mpl(qnode)(x, params)
        plt.savefig(plot_name)
        plt.show()

    def normalize_inputs(self, x):
        type_ = 0
        if isinstance(x, tf.Tensor):
            x = x.numpy()
            type_ = 1
        x = npp.array(x) / npp.linalg.norm(x)
        if type_ == 1:
            x = tf.convert_to_tensor(x)
        return x



class MPSQuantumCircuit(BaseQuantumCircuit):
    def __init__(self, n_layers, n_wires, n_block_wires=2):
        super().__init__(n_layers, n_wires)
        self.n_block_wires = n_block_wires

    def initialize_parameters(self):
        """Initialize random parameters for MPS."""
        n_blocks = self.n_wires // self.n_block_wires
        n_params_block = self.get_n_params_block()
        shape = (self.n_layers, self.n_wires-1, n_params_block)
        return npp.random.random(size = shape) #npp.array([[random.uniform(0, 2) * np.pi for _ in range(n_params_block)] for _ in range(n_blocks)])

    def get_weight_shapes(self):
        """Return the shapes of the weights in the quantum circuit."""
        n_blocks = self.n_wires // self.n_block_wires
        n_params_block = self.get_n_params_block()
        return {"weights": (n_blocks, n_params_block)}
    
    def block(self, params, wires):
        qml.RY(params[0], wires=wires[0])
        qml.RY(params[1], wires=wires[1])
        qml.CNOT(wires=[wires[0], wires[1]])
        qml.RY(params[2], wires=wires[0])
        qml.RY(params[3], wires=wires[1])

    def get_n_params_block(self):
        """Return the number of parameters in each MPS block."""
        return 2 * self.n_block_wires  # Two RY rotations per qubit

    def quantum_circuit(self, x, params, scalar_output=False, normalize=True):
        """Build the MPS quantum circuit."""
        if normalize:
            x = self.normalize_inputs(x)
        self.state_preparation(x)
        for i in range(self.n_layers):
            for j in range(self.n_wires - 1):
                # Debugging output
                #print(f"Layer {i}, Block {j}, Params shape: {params.shape}")
                #print(f"Layer {i}, Block {j}, Params: {params[i, j]}")
                self.block(params[i, j], wires=[j, j + 1])
        if scalar_output:
            return qml.expval(qml.sum([qml.PauliZ(i) for i in range(self.n_wires)]))
        else:
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_wires)]

'''
    def quantum_circuit(self, x, params, scalar_output = False, normalize = True):
        """Build the MPS quantum circuit."""
        if normalize:
            x = self.normalize_inputs(x)
        self.state_preparation(x)
        for _ in range(self.n_layers):
            qml.MPS(range(self.n_wires), self.n_block_wires, self.block, params=params)
        if scalar_output:
            return qml.expval(qml.sum(*[qml.PauliZ(i) for i in range(self.n_wires)]))
        else:
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_wires)]
'''




class StronglyEntanglingQuantumCircuit(BaseQuantumCircuit):
    def initialize_parameters(self):
        """Initialize random parameters for StronglyEntanglingLayers."""
        shape = qml.StronglyEntanglingLayers.shape(n_layers=self.n_layers, n_wires=self.n_wires)
        return npp.random.random(size=shape)
    def get_weights_shape(self):
        return {'weights':qml.StronglyEntanglingLayers.shape(n_layers=self.n_layers, n_wires=self.n_wires)}

    def quantum_circuit(self, x, params, scalar_output=False, normalize = True):
        """Build the StronglyEntangling quantum circuit."""
        if normalize:
            x = self.normalize_inputs(x)
        self.state_preparation(x)
        for _ in range(self.n_layers):
            qml.StronglyEntanglingLayers(weights=params, wires=range(self.n_wires))
        if scalar_output:
            return qml.expval(qml.sum(*[qml.PauliZ(i) for i in range(self.n_wires)]))
        else:
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_wires)]




if __name__ == '__main__':
    # Example usage
    n_layers = 5
    n_wires = 4

    X_train = np.random.random((5, n_wires))  # Example training data
    X_train = tf.convert_to_tensor(X_train)

    # Initialize the MPS quantum circuit
    mps_circuit = MPSQuantumCircuit(n_layers, n_wires)
    params_mps = mps_circuit.initialize_parameters()
    mps_circuit.draw_circuit( X_train[1], params_mps, "MPS_circuit.png")

    # Initialize the StronglyEntanglingLayers quantum circuit
    strongly_entangling_circuit = StronglyEntanglingQuantumCircuit(n_layers, n_wires)
    params_entangling = strongly_entangling_circuit.initialize_parameters()
    strongly_entangling_circuit.draw_circuit(X_train[1], params_entangling, "SEL_circuit.png")



