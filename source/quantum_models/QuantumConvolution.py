import numpy as np
import tensorflow as tf
import pennylane as qml
from pennylane import numpy as npp
from QuantumCircuit import BaseQuantumCircuit

class QuantumConvolutionLayer(BaseQuantumCircuit):
    def __init__(self, n_layers, n_wires, kernel_size):
        super().__init__(n_layers, n_wires)
        self.kernel_size = kernel_size

    def initialize_parameters(self):
        shape = (self.n_layers, self.kernel_size**2)
        return npp.random.uniform(high = 2*np.pi, size=shape)

    def quantum_circuit(self, params):
        """Build the quantum circuit for a single patch."""
        self.state_preparation(self.kernel_size**2)
        for l in range(self.n_layers):
            RandomLayers(params[l], wires=range(self.kernel_size**2))
        return [qml.expval(qml.PauliZ(j)) for j in range(self.kernel_size**2)]

    def quantum_convolution(self, x, params, scalar_output=False, normalize=True):
        """Perform quantum convolution over the input tensor."""
        if normalize:
            x = self.normalize_inputs(x)
        outputs = []
        for i in range(0, x.shape[1] - self.kernel_size + 1):
            for j in range(0, x.shape[2] - self.kernel_size + 1):
                patch = x[:, i:i+self.kernel_size, j:j+self.kernel_size]
                patch_flat = npp.reshape(patch, (x.shape[0], -1))
                outputs.extend(self.build_quantum_circuit(patch_flat, params))
        return outputs



# Example usage
n_layers = 3
n_wires = 16  # Number of wires in the circuit
kernel_size = 4

X_train = np.random.random((5, 28, 28))  # Example training data
X_train = tf.convert_to_tensor(X_train)

# Initialize the quantum convolution layer
q_conv_layer = QuantumConvolutionLayer(n_layers, n_wires, kernel_size)
params_q_conv = q_conv_layer.initialize_parameters()
outputs = q_conv_layer.quantum_convolution(X_train[0], params_q_conv)

# Output shape: (n_patches * kernel_size**2) for scalar_output=False
# Output shape: (n_patches) for scalar_output=True
print(f"Output shape: {len(outputs)}")


# Example usage
n_epochs = 20
n_layers = 1
n_train = 500
n_test = 50
save_path = "./"
kernel_size = 4

qcl = QuantumConvolutionLayer(n_epochs, n_layers, n_train, n_test, save_path, kernel_size)
qcl.train_and_save()
