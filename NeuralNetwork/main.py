from sklearn.model_selection import train_test_split

import NetworkOptimization
import TrainingAndEvaluation
import numpy as np


def read_input(featuresFile, targetsFile):
    features = np.genfromtxt(featuresFile, delimiter=",")
    targets = np.genfromtxt(targetsFile, delimiter=",")
    return features, targets


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    features, targets = read_input("data/features.txt", "data/targets.txt")
    labels = np.unique(targets)
    no_features = features.shape[1]

    # Split the input in train, validation and test sets:
    X_train, X_test, Y_train, Y_test = train_test_split(features, targets, test_size=0.3, random_state=30)
    X_validation, X_test, Y_validation, Y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=30)

    # Cross validation on optimizing the following hyperparameters:
    # hidden neurons amount, amount of hidden layers & initial weights (weights are randomly reset each time)
    # Running this will take a while!!!
    # NetworkOptimization.cross_validation(X_train, Y_train, X_validation, Y_validation)

    # -----------------------------
    # Our final, best neural network:
    hidden_layers = 1
    learning_rate = 0.1
    epochs = 20
    hidden_nodes = [26]   # nodes on each layer represented by a list
    # -----------------------------

    # Plot of the performance of the training set and the validation set during training, across epochs:
    # NetworkOptimization.network_architecture_performance(hidden_layers, learning_rate, epochs, hidden_nodes,
    #                                                     X_train, Y_train, X_validation, Y_validation)

    # Train and output success rate, confusion matrix:
    network, accuracy = TrainingAndEvaluation\
        .train_and_evaluate_neural_network(hidden_layers, learning_rate, epochs, hidden_nodes, labels, no_features,
                                           X_train, Y_train, X_test, Y_test, init_weight=None, init_bias=None,
                                           print_network=True, plot_confusion_matrix=True)

    # -----------------------------
    # Feed and output the unknown set:

    features_unknown = np.genfromtxt("data/unknown.txt", delimiter=",")
    predictions = network.predict_multiple(features_unknown)
    filename = "Group_72_classes.txt"
    file = open(filename, "w")
    for i in range(len(predictions) - 1):
        file.write(str(predictions[i]) + ",")
    file.write(str(predictions[len(predictions) - 1]))
    file.close()

    print("Done printing to file!")

