import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import Sequential, layers, Input, Model
from tensorflow.keras.regularizers import L1, L2, L1L2
import numpy as np
import random

import pennylane as qml
import pennylane.numpy as npp
import matplotlib.pyplot as plt

from sys import path
path.append("../classical_models/")
from NeuralNetwork import NeuralNetworkClassifier
path.append("../quantum_models/")
from QuantumCircuit import StronglyEntanglingQuantumCircuit, BaseQuantumCircuit, MPSQuantumCircuit

from time import time





# FILTER OUT SOME WARNING
import warnings
#warnings.filterwarnings("ignore", message="You are casting an input of type complex128 to an incompatible dtype float32.")
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')





def default_layer(X_train, units, regularizer=None, dropout_rate=0.3):
    """
    Creates a default layer
    """
    input_shape = X_train.shape[1:] + (1,)
    return [
        layers.Conv2D(16, kernel_size=(5, 5), padding='same', activation='relu', kernel_regularizer=regularizer, input_shape=input_shape),
        layers.Dropout(rate=dropout_rate),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, kernel_size=(5, 5), padding="same", activation='relu', kernel_regularizer = regularizer),
        layers.Dropout(rate=dropout_rate),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
    ]




class QuantumNeuralNetwork(NeuralNetworkClassifier):
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(X_train, X_test, y_train, y_test)
        self.x = Input(shape= self.X_train.shape[1:] + (1,))
        


    def apply_qlayers(self, x, qm_circuit, concatenate = True, axis = 1, normalize = False):
        """
        Takes as inputs:
        x - Input to pass to the quantum layer
        qm_circuit - quantum circuit from one of the class ( MPSQuantumCircuit, StronglyEntanglingQuantumCircuit)

        """
        n_wires = qm_circuit.n_wires
        n_qm_circuits = x.shape[1] // n_wires
    
        weight_shape = qm_circuit.initialize_parameters().shape

        """ We need to define the qnode dynamically, and cannot use the one inside the class BaseQuantumCircuit """
        dev = qml.device("default.qubit", wires=range(n_wires))
        @qml.qnode(dev, diff_method="backprop", interface="tf")
        def qnode(inputs, weights):
            return qm_circuit.quantum_circuit(inputs, weights, normalize = normalize)
            
        qlayers = []
        for i in range(n_qm_circuits):
            qlayer = qml.qnn.KerasLayer(qnode, weight_shapes =  {'weights': weight_shape}, output_dim=n_wires, trainable=True) # weight_specs = {"weights": {"initializer": "random_uniform"}}
            qlayers.append(qlayer)

        ## Split the input tensor (x) and pass it to the qlayers
        x = layers.Lambda(lambda x: tf.split(x, n_qm_circuits, axis = axis))(x) 
        output = [qlayer(out) for q_layer, out in zip(qlayers, x)]

        if concatenate == True:
            output = layers.Concatenate(axis=axis)(output)

        return output

    def apply_dense_layer(self, x, m, activation = None, regularizer = None):
        """
        Apply a Dense Layer (for renormalization)

        Inputs:
        x - inputs of the layer
        m - number of units of the Dense layer (size of the output)
        activation - Activation function of the dense layer (
        
        """
        x = layers.Dense(m, activation = activation, kernel_regularizer = regularizer)(x)
        return x

    def apply_classical_layers(self, x, layer_list):
        """
        Applies a list of layers to the model
        """
        for layer in layer_list:
            x = layer(x)
        return x

    def create_model(self, inputs, outputs):
        """
        Creates the model
        """
        self.model = Model(inputs = inputs, outputs = outputs)
        return self.model

    def create_default_model(self, m, MPS, units = 150, regularizer = None, dropout_rate = 0.3, *args, **kwargs):
        """
        Creates the default model (choosing either MPS or StronglyEntanglingLayer)

        """
        x = self.x
        inputs = x
        
        layer_list = default_layer(x, units = units, dropout_rate = dropout_rate, regularizer = regularizer)
        x = self.apply_classical_layers(x, layer_list)  ## Apply the first layers
        x = self.apply_dense_layer(x, m, activation = "relu", regularizer = regularizer)  ## Apply the normalization layer to compact it into m

        # Create a MPS or a StronglyEntanglingQuantumCircuit
        qm_circuit = MPSQuantumCircuit( *args, **kwargs ) if MPS else StronglyEntanglingQuantumCircuit( *args, **kwargs)

        x = self.apply_qlayers(x, qm_circuit, concatenate = True)

        x = self.apply_dense_layer(x, 10, activation = 'softmax')

        self.model = self.create_model(inputs, x)
        return self.model

    def save_accuracy(self, MPS, filename="../../result/accuracy_data.txt", max_cut=None):
            if MPS:
                class_name = self.__class__.__name__ + "_MPS"
            else:
                class_name = self.__class__.__name__ + "_SEL"

            # Get the testing accuracy
            if max_cut is None:
                # Predict on the training data
                predictions = self.predict(self.X_train)
                testing_accuracy = self.score(self.y_train, predictions)

                # Get the validation accuracy
                validation_accuracy = self.accuracy
            else:
                # Predict on the training data with max_cut
                predictions = self.predict(self.X_train[:max_cut])
                testing_accuracy = self.score(self.y_train[:max_cut], predictions)

                # Get the validation accuracy with max_cut
                if self.accuracy is None:
                    val_predictions = self.predict(self.X_test[:max_cut])
                    validation_accuracy = self.score(self.y_test[:max_cut], val_predictions)
                else:
                     validation_accuracy = self.accuracy

            # Check if class_name is already in the file
            found = False
            with open(filename, "r") as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if class_name in line:
                        found = True
                        # Update the accuracy for the existing class_name
                        lines[i] = f"{class_name}     {testing_accuracy:.4f}     {validation_accuracy:.4f}     {str(max_cut)}\n"
                        break

            # If class_name is not found, append the new data
            if not found:
                lines.append(f"{class_name}     {testing_accuracy:.4f}     {validation_accuracy:.4f}     {str(max_cut)}\n")
                
            # Write the updated data to the file
            with open(filename, "w") as f:
                f.writelines(lines)

    def save_time(self, start_time, MPS, filename="../../result/time_data.txt"):
        if MPS:
            class_name = self.__class__.__name__ + " MPS"
        else:
            class_name = self.__class__.__name__ + " SEL"

        current_time = time() - start_time

        # Check if class_name is already in the file
        found = False
        with open(filename, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if class_name in line:
                    found = True
                    # Update the time for the existing class_name
                    lines[i] = f"{class_name}     {current_time}\n"
                    break

        # If class_name is not found, append the new data
        if not found:
            lines.append(f"{class_name}     {current_time}\n")

        # Write the updated data to the file
        with open(filename, "w") as f:
            f.writelines(lines)
        
            

if __name__ == '__main__':
    from sys import path
    path.append("../../utils/")
    import pre_processing
    path.append("../")
    
    from pre_processing import Dataset
    data = Dataset()
    data.default()
    #
    n_train = 3000; n_test = 800

    n_layers = 5
    n_wires = 5
    m = 40
    MPS = False
    filename = "quantum_nn_model_SEL.keras"
    s_t = time()
    # Create an instance of QuantumNeuralNetwork
    quantum_nn = QuantumNeuralNetwork(data.train_images[:n_train], data.test_images[:n_test], data.train_labels[:n_train], data.test_labels[:n_test])
    quantum_nn.model = quantum_nn.create_default_model(m, dropout_rate = 0.30, regularizer = None, MPS =  MPS, n_layers=n_layers, n_wires= n_wires)
    quantum_nn.fit(learning_rate = 0.002, epochs = 70)


    quantum_nn.save_model(filename = filename)
    quantum_nn.save_time(s_t,  MPS = MPS)
    quantum_nn.save_accuracy(MPS =  MPS, max_cut = 750)



    

    


