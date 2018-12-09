import cupy as cp
import matplotlib.pyplot as plt
from collections import OrderedDict 

def base_sigmoid(Z):
    return 1/(1+cp.exp(-Z))

def sigmoid(Z):
    A = base_sigmoid(Z)
    assert (A.shape == Z.shape)
    cache = Z
    return A, cache

def sigmoid_backward(dA, cache):
    Z = cache

    s = base_sigmoid(Z)
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)

    return dZ

def relu(Z):
    A = cp.maximum(0, Z)
    assert (A.shape == Z.shape)
    cache = Z
    return A, cache

def relu_backward(dA, cache):
    Z = cache
    dZ = cp.array(dA, copy=True)
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ

def initialize_parameters_deep(layers):

    parameter = OrderedDict()
    cp.random.seed(1)

    for i in range(1, len(layers)):
        parameter['W' + str(i)] = cp.random.randn(layers[i], layers[i-1]) / cp.sqrt(layers[i-1])
        parameter['b' + str(i)] = cp.zeros((layers[i], 1))
        assert(parameter['W' + str(i)].shape == (layers[i], layers[i-1]))
        assert(parameter['b' + str(i)].shape == (layers[i], 1))
    
    return parameter

def linear_forward(A, W, b):

    Z = cp.dot(W, A) + b
    
    cache = (A, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache

def liner_model_forward(X,params):

    caches = []
    A = X
    layers = len(params) // 2

    for i in range(1, layers):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, params["W"+str(i)], params["b"+str(i)], activation='relu')
        caches.append(cache)

    AL, cache = linear_activation_forward(A, params['W' + str(layers)], params['b' + str(layers)], activation = "sigmoid")
    caches.append(cache)

    return AL, caches

def compute_cost(AL, Y):
    m = Y.shape[1]

    cost = cp.multiply((1.0/m), (-cp.dot(Y, cp.log(AL).T) - cp.dot(1-Y, cp.log(AL).T)))
    cost = cp.squeeze(cost)

    assert(cost.shape == ())

    return cost

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1./m * cp.dot(dZ,A_prev.T)
    db = 1./m * cp.sum(dZ, axis = 1, keepdims = True)
    dA_prev = cp.dot(W.T,dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

def linear_model_backward(AL, Y, caches):
    grads = OrderedDict()
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    dAL = - (cp.divide(Y, AL) - cp.divide(1 - Y, 1 - AL))
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")
    
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters

def predict(X, y, parameters):
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = cp.zeros((1,m))
    
    # Forward propagation
    probas, caches = liner_model_forward(X, parameters)

    
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    print("Accuracy: "  + str(cp.sum((p == y)/m)))
        
    return p

def print_mislabeled_images(classes, X, y, p):
    a = p + y
    mislabeled_indices = cp.asnumpy(cp.asarray(cp.where(a == 1)))
    plt.rcParams['figure.figsize'] = (40.0, 40.0) # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]
        
        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction: " + classes[int(p[0,index])].decode("utf-8") + " \n Class: " + classes[y[0,index]].decode("utf-8"))

