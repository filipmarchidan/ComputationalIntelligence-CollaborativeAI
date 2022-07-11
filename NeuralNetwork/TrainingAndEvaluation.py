import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from NeuralNetwork import NeuralNetwork


def train_and_evaluate_neural_network(hidden_layers, learning_rate, epochs, hidden_nodes, labels, no_features,
                                      X_train, Y_train, X_test, Y_test, init_weight=None, init_bias=None,
                                      print_network=False, plot_confusion_matrix=False):

    network = NeuralNetwork(input_nodes=no_features, output_nodes=labels.shape[0], hidden_layers= hidden_layers,
                            hidden_nodes=hidden_nodes, learning_rate=learning_rate, init_weight=init_weight, init_bias=init_bias)

    # Train the network:
    network.train(X_train, Y_train, epochs=epochs)

    if print_network:
        print("Training complete!")
        print(str(network))

    # Evaluate after training
    accuracy = network.evaluate(X_test, Y_test)
    print("Success rate:" + "\n" + str(accuracy))


    if plot_confusion_matrix:
        np.set_printoptions(precision=2)
        plot_confusion_matrix_(network.predict_multiple(X_test), Y_test, classes=labels,
                              title='Confusion matrix, without normalization', accuracy=accuracy)
        plt.show()

    return network, accuracy

def plot_confusion_matrix_(y_true, y_pred, classes,
                          normalize=False,
                          title=None, accuracy=-1,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes.astype('int32')
    if normalize:
        cm = cm.astype('int') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")


    if (accuracy != -1):
        fig.text(.5, .0001, "Total accuracy: " + str(accuracy) + '\n', ha='center')

    fig.tight_layout()

    return ax