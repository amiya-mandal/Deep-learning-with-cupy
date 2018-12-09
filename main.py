import random
from load_data import load_mnist

from activation_function import *


def L_layer_model(X, Y, parameters, learning_rate=0.001, print_cost=False):
    cp.random.seed(1)
    costs = []

    for i in range(0, len(X)):
        AL, caches = liner_model_forward(X[i], parameters)
        cost = compute_cost(AL, Y)
        grads = linear_model_backward(AL, Y[i], caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    
    return parameters


def main():

    train, test = load_mnist()

    layers_dims = [784, 256, 64, 10]
    parameters = initialize_parameters_deep(layers_dims)

    for i in range(3):
        t = list(train)
        random.shuffle(t)
        X, y = zip(*t)
        parameters = L_layer_model(X, y, parameters)

main()




