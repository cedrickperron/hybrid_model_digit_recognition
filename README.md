# My Project

Use a hybrid classical quantum neural networks for digit recognition system. Thus far, our results for the quantum neural networks are incapable of predicting accurately validation data.

We also build a quantvolution network (in jupyter file). The images are stored in quantumconvol/.

For our code, we take the framework of arxiv:2304.09224 (Quantum machine learning for image classification by Arsenii Senokosov, Alexander Sedykh, Asel Sagingalieva, and Alexey Melnikov). Their architecture and results are inconsistent with the natural world.

## Setup

```bash
bash setup_env.sh
source ./.virtualenvs/jupteach/bin/activate
```


## Usage


## Procedure
Can run each model individually by running the model file:
```bash
python ./source/classical_models/KNNClassifier.py
```

```bash
python ./source/classical_models/LogisticRegressor.py
```

```bash
python ./source/classical_models/RandomForest.py
```


```bash
python ./source/classical_models/NeuralNetwork.py
```

```bash
python ./source/quantum_models/QuantumNN.py
```
This will run the default model (Strongly Entangling Layers circuit). To run the (Matrix Product State circuit), you need to set MPS = True in __name__ == '__main__'

**The models above will be saved in the directory: `save/` and their performance stored in the directory: `result/` **

FOR THE CONVOLUTION

It can be run in `Hybrid-Model.ipynb`, but takes a while

Directory:
`quantumconvol` stores the images that were convolved using a quantum algorithms.


## TODO
- Figure out the optimal choice of parameters for our model to predict unseen data well. (Need to parallelize the code for this step).
- Create a new quanvolution classes.
- Get the accuracy/predictions of the different models.
