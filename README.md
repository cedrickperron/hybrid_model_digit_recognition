# My Project

Use a hybrid classical quantum neural networks for digit recognition system. Thus far, our results for the quantum neural networks are incapable of predicting accurately validation data.

We also build a quantvolution network (in jupyter file). The images are stored in

For our code, we take the framework of arxiv:2304.09224 (Quantum machine learning for image classification by Arsenii Senokosov, Alexander Sedykh, Asel Sagingalieva, and Alexey Melnikov)

## Installation


## Usage

The entire code can be run in `Hybrid-Model.ipynb`

Directory:
`checkpoints` stores the information about the models history (qm_fitting.pkl, class_fitting.pkl, qm_conv_qm_fitting.pkl, qm_conv_class_fitting.pkl)
and contains information about the models' weights (qm_model_weights.pkl, class_model_weights.pkl, qm_conv_qm_model_weights.pkl, qm_conv_class_model_weights.pkl)
`quantumconvol` stores the images that were convolved using a quantum algorithms.
