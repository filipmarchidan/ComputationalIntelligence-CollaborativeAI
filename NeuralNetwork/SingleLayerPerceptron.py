import numpy as np
import matplotlib.pyplot as plt

weight1 = 0.3
weight2 = -0.1

learning_rate = 0.1
threshold = 0.2

x1 = np.array([0, 0, 1, 1])
x2 = np.array([0, 1, 0, 1])

yAND = np.array([0, 0, 0, 1])
yOR = np.array([0, 1, 1, 1])
yXOR = np.array([0, 1, 1, 0])
evalAND = list()
evalOR = list()
evalXOR = list()


def activationFunction(x):
    if x >= 0:
        return 1
    else:
        return 0


def predict(x1, x2):
    global weight1, weight2, threshold
    return activationFunction(round(weight1 * x1 + weight2 * x2 - threshold, 1))


def updateWeights(x1, x2, yd):
    global weight1, weight2
    aux = learning_rate * (yd - predict(x1, x2))
    # print("delta --> w1", aux * x1, " w2:", aux * x2)
    weight1 = round(weight1 + aux * x1, 1)
    weight2 = round(weight2 + aux * x2, 1)


def train(yd, epochs, eval):
    global x1, x2, weight1, weight2
    for j in range(epochs):
        pred = list()
        actual = list()
        print("Epoch: ", j + 1)
        for i in range(len(x1)):
            print(x1[i], x2[i], " | ", yd[i], predict(x1[i], x2[i]), " | ", (yd[i] - predict(x1[i], x2[i])), " | ",
                  weight1, weight2)
            pred.append(predict(x1[i], x2[i]))
            actual.append(yd[i])
            updateWeights(x1[i], x2[i], yd[i])
        eval.append(evaluate(pred, actual))
        # print("x1: ", x1[i], "x2: ", x2[i], "y: ", yd[i], "predict: ", predict(x1[i], x2[i]), "weight1: ", weight1, "weight2: ", weight2)


def test():
    global x1, x2, yAND, yOR, yXOR, weight1, weight2

    print("AND")
    weight1 = 0.3
    weight2 = -0.1
    print("OR")
    train(yOR, 6, evalOR)
    plotGraphOR(5, evalOR)
    weight1 = 0.3
    weight2 = -0.1
    print("XOR")
    train(yXOR, 6, evalXOR)
    plotGraphXOR(5, evalXOR)

    train(yAND, 6, evalAND)
    plotGraphAnd(5, evalAND)
    # print("Weight1: ", weight1)
    # print("Weight2: ", weight2)


def plotGraphAnd(epochs, evalList):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(evalList, label = "AND error")
    plt.legend(fancybox=True, shadow=True, borderpad=1)
    plt.title("Accuracy of AND per epoch")
    plt.xlim([0, epochs])
    plt.ylim(0, 1.1)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()

def plotGraphOR(epochs, evalList):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(evalList, label = "OR error")
    plt.legend(fancybox=True, shadow=True, borderpad=1)
    plt.title("Accuracy of OR per epoch")
    plt.xlim([0, epochs])
    plt.ylim(0, 1.1)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()

def plotGraphXOR(epochs, evalList):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(evalList, label = "XOR error")
    plt.legend(fancybox=True, shadow=True, borderpad=1)
    plt.title("Accuracy of XOR per epoch")
    plt.xlim([0, epochs])
    plt.ylim(0, 1.1)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()

def evaluate(Xtest, Ytest):
    """
        Evaluates the neural network on the given test data.
        :param Xtest: test data
        :param Ytest: test labels
        :return: accuracy of the network on the test data
        """
    correct = 0
    wrong = 0
    for i in range(len(Xtest)):
        # we add 1 because we have classes starting with 1 and there is no class 0
        if Xtest[i] == Ytest[i]:
            correct += 1
        else:
            wrong += 1
    return correct / (correct + wrong)
