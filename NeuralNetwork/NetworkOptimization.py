import numpy as np
import matplotlib.pyplot as plt

from NeuralNetwork import NeuralNetwork

def cross_validation(X_train, Y_train, X_validation, Y_validation):
    train_network_different_hidden_neurons(X_train, Y_train, X_validation, Y_validation)
    train_network_different_hidden_layers(X_train, Y_train, X_validation, Y_validation)
    train_network_different_initial_weights(X_train, Y_train, X_validation, Y_validation)


def train_network(hidden_layers, learning_rate, epochs, hidden_nodes, X_train, Y_train,
                  X_validation=None, Y_validation=None, init_weight=None, init_bias=None):
    """
    Trains and returns a neural network.
    """
    labels = np.unique(Y_train)
    no_features = X_train.shape[1]

    network = NeuralNetwork(input_nodes=no_features, output_nodes=labels.shape[0], hidden_layers=hidden_layers,
                            hidden_nodes=hidden_nodes, learning_rate=learning_rate, init_weight=init_weight,
                            init_bias=init_bias)
    # print(str(network))

    # Train the network:
    accuracy_accross_epochs = network.train(X_train, Y_train, epochs, X_validation, Y_validation)

    return network, accuracy_accross_epochs

def accuracy_of_hyper_params(times_trained, X_validation, Y_validation,
                             hidden_layers, learning_rate, epochs, hidden_nodes, X_train, Y_train, init_weight=None, init_bias=None):
    """
    Trains a network n times and returns the average accuracy.
    """
    total_accuracy = 0
    for i in range(times_trained):
        network, accuracy_accross_epochs = train_network(hidden_layers, learning_rate, epochs, hidden_nodes, X_train, Y_train, init_weight, init_bias)

        accuracy = network.evaluate(X_validation, Y_validation)
        total_accuracy += accuracy

    return total_accuracy / times_trained


def train_network_different_initial_weights(X_train, Y_train, X_validation, Y_validation):

    performances = []

    for i in range(10):
        # network = train_network(hidden_layers=1, learning_rate=0.1, epochs=10, hidden_nodes=[7], X_train=X_train, Y_train=Y_train)
        network, accuracy_across_epochs = train_network(hidden_layers=1, learning_rate=0.1, epochs=10, hidden_nodes=[7],
                                                        X_train=X_train, Y_train=Y_train)
        accuracy = network.evaluate(X_validation, Y_validation)
        performances.append(accuracy)

        print("Accuracy random network " + str(i) + ": " + str(accuracy))

    # ----------------------
    # Plot performances:
    plt.bar(np.arange(1, 11), performances, align='center')

    for i in range(10):
        plt.text(i+0.7, performances[i], str(round(performances[i], 2)))

    plt.title("Accuracy of random initialized weights")
    plt.xlabel("Initializations")
    plt.ylabel("Accuracy/Performance")
    plt.show()
    # ----------------------


def train_network_different_hidden_layers(X_train, Y_train, X_validation, Y_validation):
    min_hidden_layer_size = 1
    max_hidden_layer_size = 3
    step_hidden_layer = 1
    hidden_layer = list(range(min_hidden_layer_size, max_hidden_layer_size, step_hidden_layer))
    hidden_layer.extend([max_hidden_layer_size])

    performances = []    # for each layer a list of performances across epochs
    epochs_list = list(range(1, 21, 4))


    for layers_amount in hidden_layer:
        performances_across_epochs = []

        for epoch in epochs_list:
            accuracy = accuracy_of_hyper_params(times_trained=10, X_validation=X_validation, Y_validation=Y_validation,
                                                hidden_layers=layers_amount, learning_rate=0.1, epochs=epoch,
                                                hidden_nodes=np.full(layers_amount, 5),
                                                X_train=X_train, Y_train=Y_train)
            performances_across_epochs.append(accuracy)
        performances.append(np.array(performances_across_epochs))
        print("Accuracy with " + str(layers_amount) + " hidden layers: " + str(performances_across_epochs))


    # ----------------------
    # Plot performances:
    # plt.plot(hidden_layer, performances)

    for i in range(len(hidden_layer)):
        plt.plot(epochs_list, performances[i], label=hidden_layer[i])
    plt.legend(title="Hidden layers", loc='upper left')
    plt.title("Number of hidden layers vs. Accuracy")
    plt.xlabel("Number of epochs (5 hidden neurons per layer!)")
    plt.ylabel("Accuracy/Performance")
    plt.show()
    # ----------------------

    max_index = np.argmax(performances)
    best_layers_amount = hidden_layer[max_index]
    return best_layers_amount

def train_network_different_hidden_neurons(X_train, Y_train, X_validation, Y_validation):
    min_hidden_node_size = 7
    max_hidden_node_size = 30
    step_hidden_node = 3
    nodes = list(range(min_hidden_node_size, max_hidden_node_size, step_hidden_node)) # 7 to 30, step of 3
    nodes.extend([max_hidden_node_size])

    performances = []

    for nodes_amount in nodes:
        accuracy = accuracy_of_hyper_params(times_trained=10, X_validation=X_validation, Y_validation=Y_validation,
                                            hidden_layers=1, learning_rate=0.1, epochs=10, hidden_nodes=np.full(1, nodes_amount),
                                            X_train=X_train, Y_train=Y_train)
        performances.append(accuracy)
        print("Accuracy " + str(nodes_amount) + " hidden nodes: " + str(accuracy))

    # ----------------------
    # Plot performances:
    plt.plot(nodes, performances)
    plt.title("Number of hidden neurons vs. Accuracy")
    plt.xlabel("Number of hidden neurons (1 layer considered)")
    plt.ylabel("Accuracy/Performance")
    plt.show()
    # ----------------------

    # Return best neuron amount:
    max_index = np.argmax(performances)
    best_neurons_amount = nodes[max_index]
    return best_neurons_amount


def network_architecture_performance(hidden_layers, learning_rate, epochs, hidden_nodes,
                                     X_train, Y_train, X_validation, Y_validation):

    network, accuracy_accross_epochs = train_network(hidden_layers, learning_rate, epochs, hidden_nodes, X_train,
                                                     Y_train, X_validation, Y_validation)

    plt.plot(np.arange(1, epochs + 1), accuracy_accross_epochs)
    plt.title("Epochs vs. Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()

