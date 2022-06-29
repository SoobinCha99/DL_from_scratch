import numpy as np 

def identity_function(x): 
    return x 

def sigmoid_function(x): 
    output = 1 / 1+ np.exp(-x)
    return output 

def relu_function(x):
    return np.maximum(0,x) 

def init_network(): 
    network = {} 
    network["W1"] = np.random.rand(2,3)
    network["b1"] = np.random.rand(3,)
    network["W2"] = np.random.rand(3,3)
    network["b2"] = np.random.rand(3,)
    network["W3"] = np.random.rand(3,2)
    network["b3"] = np.random.rand(2,)

    return network
    
def feed_forward(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"] 

    a1 = np.dot(x,W1) + b1 
    z1 = sigmoid_function(a1)
    a2 = np.dot(z1,W2) + b2 
    z2 = sigmoid_function(a2) 
    a3 = np.dot(z2,W3) + b3 
    y = identity_function(a3)

    return y