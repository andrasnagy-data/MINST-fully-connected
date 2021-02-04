**Fully connected Neural Net trained on the MINST dataset.**

Model architecture: Input layer size = the flattened image size = 784 neurons, both of the 2 hidden layers consists of 16 neurons and the last fully connected layer has 10 neurons, which equals to the number of classes to predict. After the Cross-Entropy Loss function, the last layer consists of probabilities assigned to the respective class label (represented by a neuron), and the model predicts the class that has the highest probability after the "forward pass".

*Note*: The model starts over-fitting after the 5th epoch.

During this project, I learned about data sets (dataset class for deep learning), data loaders (iterator wrapped around the dataset class), model architecture and training loop (with evaluation).

The model's accuracy after the 5th epoch was ~94 %.
