from typing import overload

import numpy
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class NeuralNetwork:
    # Available object params/variables:
    # input_nodes, output_nodes, hidden_layers, hidden_nodes, learning_rate
    # weights   - list of numpy matrices
    # biases    - list of numpy vectors

    def __init__(self, input_nodes, output_nodes, hidden_layers, hidden_nodes, learning_rate, init_weight=None,
                 init_bias=None):
        """
        :param input_nodes: Number of input nodes / features.
        :param output_nodes: Number of output nodes / classes.
        :param hidden_layers: Number of total hidden layers.
        :param hidden_nodes: List of integers where the ith element is the number of neurons in the ith hidden layer.
        :param learning_rate: The learning rate used.
        :param init_weight: (Optional) initialized weight on all nodes in the network. If not specified weights are random.
        :param init_bias: (Optional) initialized bias on all nodes.
        """

        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.hidden_layers = hidden_layers
        self.hidden_nodes = hidden_nodes
        self.learning_rate = learning_rate

        # -------------------------------
        # Weight initialization
        # Represented by list containing np.arrays
        # list[i] has a 2d matrix with weights connected to and going into nodes on layer i
        # The matrix has all weights coming from a certain neuron on rows
        # M[i][j] - weight coming from neuron i on prev. layer to neuron j on current layer
        self.weights = list()
        for i in range(hidden_layers):
            if i == 0:
                size = (input_nodes, hidden_nodes[i])
                f_i = input_nodes
            else:
                size = (hidden_nodes[i - 1], hidden_nodes[i])
                f_i = hidden_nodes[i - 1]

            if init_weight is None:
                W = np.random.uniform(low=(-2.4 / f_i), high=(2.4 / f_i),
                                      size=size)  # random weights on uniform distribution
            else:
                W = np.full(size, init_weight)
            self.weights.append(W)

        # Weights final hidden layer -> output nodes:
        size = (hidden_nodes[hidden_layers - 1], output_nodes)
        f_i = hidden_nodes[hidden_layers - 1]

        if init_weight is None:
            W = np.random.uniform(low=(-2.4 / f_i), high=(2.4 / f_i),
                                  size=size)  # random weights on uniform distribution
        else:
            W = np.full(size, init_weight)
        self.weights.append(W)

        # -------------------------------
        # Bias initialization
        # Represented by list containing np.arrays
        # list[i] has a 1d array with biases for neurons on layer i
        self.biases = list()

        for i in range(hidden_layers):
            if i == 0:
                f_i = input_nodes
            else:
                f_i = hidden_nodes[i - 1]

            if init_bias is None:
                biases_layer = np.random.uniform(low=(-2.4 / f_i), high=(2.4 / f_i),
                                                 size=hidden_nodes[i])  # random biases on uniform distribution
            else:
                biases_layer = np.full(hidden_nodes[i], init_bias)
            self.biases.append(biases_layer)

        # Bias last hidden layer -> output nodes
        f_i = hidden_nodes[hidden_layers - 1]
        if init_bias is None:
            biases_output_layer = np.random.uniform(low=(-2.4 / f_i), high=(2.4 / f_i),
                                                    size=output_nodes)  # random biases on uniform distribution
        else:
            biases_output_layer = np.full(output_nodes, init_bias)
        self.biases.append(biases_output_layer)
        # --------------------------------


    # Predict/output labels for some provided features
    def predict(self, features: list):
        """
        Predicts output labels for some object's features.
        :param features: List of object features (input layer).
        :return: List with activations.
        """
        if len(features) != self.input_nodes:
            raise Exception("Input feature size not compatible with this network!")

        Y = np.array(features)
        for i in range(self.hidden_layers + 1):  # go through each weight matrix
            W = self.weights[i]
            Y = W.transpose() @ Y + self.biases[i]  # Multiplication layer by layer
            Y = sigmoid(Y)

            # print("Predictions layer " + str(i) + ": " + str(Y))
        # print("Predictions: " + str(Y) + str(np.argmax(Y)))
        return Y.tolist()   # convert numpy vector to list!

    def predict_multiple(self, feature_arr: numpy.array):
        """
        Predicts output labels for multiple objects.
        :param feature_arr: 2dim numpy array with multiple objects' features.
        :return: Label predictions list for each object.
        """
        predictions = []
        for i in range(len(feature_arr)):
            predictions.append(np.argmax(self.predict(feature_arr[i])) + 1)
        return predictions


    def get_activations(self, features):
        """
        Compute activations on each layer.
        :param features: List containing the object's features.
        :return: List of layer activations. Element i is a numpy-array of activations on layer i.
        """
        if len(features) != self.input_nodes:
            raise Exception("Input feature size not compatible with this network!")

        activations = list()

        Y = np.array(features)
        for i in range(self.hidden_layers + 1):  # go through each weight matrix
            W = self.weights[i]
            Y = W.transpose() @ Y + self.biases[i]  # Multiplication layer by layer
            Y = sigmoid(Y)
            activations.append(Y)

        return activations

    def train(self, X_train, Y_train, epochs, X_validation=None, Y_validation=None):  # X_train and Y_train are lists
        """
        :param Y_validation:
        :param X_validation:
        :param X_train: Training set with features of objects.
        :param Y_train: Training set labels.
        :param epochs: How many epochs to consider.
        """

        epochs_accuracy_validation = list()

        for p in range(epochs):  # p - iteration
            weights_prev_iter = self.weights.copy()  # maintain copies of prev. iteration p

            # Go through the training set:
            for q in range(len(Y_train)):

                # Get Input and Output:
                X = np.array(X_train[q])
                Y = self.convert_label_to_numpy_arr(round(Y_train[q]))

                # List of activations on each layer:
                activations = self.get_activations(X_train[q])

                # List of error gradients for layer i
                error_gradient = list()
                # Weight correction for weight between neurons j and k
                weight_correction = 0

                # Calculate error gradient for neuron i on the output layer
                for i in range(self.output_nodes):
                    actual_output = activations[self.hidden_layers][i]
                    error_gradient.append(actual_output * (1 - actual_output) * (Y[i] - actual_output))

                    # Update weights between output layer and prev hidden layer
                    for j in range(self.hidden_nodes[self.hidden_layers - 1]):
                        weight_correction = self.learning_rate * activations[self.hidden_layers - 1][j] * \
                                            error_gradient[i]
                        weight = weights_prev_iter[self.hidden_layers][j][i] + weight_correction
                        self.weights[self.hidden_layers][j][i] = weight

                # Go through network layers in reverse:
                for l in range(len(self.weights) - 2, -1, -1):
                    W = weights_prev_iter[l]  # Considering prev iter. weights between layer l & l+1 !!!!!!
                    aux = list()  # list of current layer error gradients that will replace prev layer error gradients

                    # Neuron i on layer l+1
                    for i in range(W.shape[1]):
                        actual_output = activations[l][i]
                        sum = 0

                        # Calculate the sum of products between error gradients of layer l + 2 and weights of neuron i and layer l + 2
                        for j in range(weights_prev_iter[l + 1].shape[1]):
                            sum = sum + weights_prev_iter[l + 1][i][j] * error_gradient[j]
                        aux.append(actual_output * (1 - actual_output) * sum)

                        # Neuron j on layer l (and neuron i on layer l+1)
                        if l != 0:
                            for j in range(W.shape[0]):
                                weight_correction = self.learning_rate * activations[l - 1][j] * aux[i]
                                weight = weights_prev_iter[l][j][i] + weight_correction
                                self.weights[l][j][i] = weight
                        else:
                            for j in range(self.input_nodes):
                                weight_correction = self.learning_rate * X[j] * aux[i]
                                weight = weights_prev_iter[l][j][i] + weight_correction
                                self.weights[l][j][i] = weight

                    # Update list of error gradients
                    error_gradient = aux
            if X_validation is not None:
                accuracy = self.evaluate(X_validation, Y_validation)
                epochs_accuracy_validation.append(accuracy)

        return epochs_accuracy_validation   # empty list if we do not have validation

    def convert_label_to_numpy_arr(self, label: int):
        """
        Used internally. Converts one label to a numpy array with only one activation corresponding to the label number.
        """
        Y = np.zeros(self.output_nodes)
        Y[label - 1] = 1  # neuron on position label-1 is the only one activated
        return Y


    def evaluate(self, Xtest, Ytest):
        """
        Evaluates the neural network on the given test data.
        :param Xtest: test data
        :param Ytest: test labels
        :return: accuracy of the network on the test data
        """
        correct = 0
        wrong = 0
        for i in range(len(Xtest)):
            Y = self.predict(Xtest[i])
            # we add 1 because we have classes starting with 1 and there is no class 0
            if np.argmax(Y) + 1 == Ytest[i]:
                correct += 1
            else:
                wrong += 1
        return correct / (correct + wrong)




    # toString function:
    def __str__(self):
        s = "Neural network: \n" + "Features: " + str(self.input_nodes) + \
            "\nHidden Layers: " + str(self.hidden_layers) + "\nHidden_nodes on each layer: " + str(self.hidden_nodes) + \
            "\nOutput Nodes: " + str(self.output_nodes) + "\nLearning rate: " + str(self.learning_rate) + '\n'
        for i in range(len(self.weights)):
            s += "Weight Matrix between layers " + str(i) + "<->" + str(i + 1) + '\n'
            s += str(self.weights[i]) + "\n"
            s += "Biases between layers " + str(i) + "<->" + str(i + 1) + '\n'
            s += str(self.biases[i]) + "\n"

        return s
