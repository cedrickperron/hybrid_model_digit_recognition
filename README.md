# My Project

Compare an hybrid (classical-quantum) machine learning model to a classical machine learning model for digit recognition.

Also, we compare a quantum convolution to a classical convolution of the image.

For our code, we take the framework of arxiv:2304.09224 (Quantum machine learning for image classification by Arsenii Senokosov, Alexander Sedykh, Asel Sagingalieva, and Alexey Melnikov)

## Installation

pip install tensorflow==2.12.0

pip install pennylane==0.34.0

## Usage

The entire code can be run in `Hybrid-Model.ipynb`

Directory:
`checkpoints` stores the information about the models history (qm_fitting.pkl, class_fitting.pkl, qm_conv_qm_fitting.pkl, qm_conv_class_fitting.pkl)
and contains information about the models' weights (qm_model_weights.pkl, class_model_weights.pkl, qm_conv_qm_model_weights.pkl, qm_conv_class_model_weights.pkl)
`quantumconvol` stores the images that were convolved using a quantum algorithms.
