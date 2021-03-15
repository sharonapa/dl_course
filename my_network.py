import numpy as np


def initialize_parameters(layer_dims):
    '''
    :param layer_dims:  an array of the dimensions of each layer in the network
    (layer 0 is the size of the flattened input, layer L is the output softmax)
    :return: a dictionary containing the initialized W and b parameters of each layer (W1…WL, b1…bL)
    '''
    w_b_parmas_dic = {}
    for i in range(1, len(layer_dims)):
        layer_dim_prev = layer_dims[i - 1]
        layer_dim = layer_dims[i]
        w_b_parmas_dic['W' + str(i)] = np.random.randn(layer_dim_prev, layer_dim)
        w_b_parmas_dic['b' + str(i)] = np.zeros((layer_dim, 1))

    return w_b_parmas_dic


def linear_forward(A, W, b):
    '''
        Implement the linear part of a layer's forward propagation
        :param A: the activations of the previous layer
        :param W: the weight matrix of the current layer (of shape [size of current layer, size of previous layer])
        :param b: the bias vector of the current layer (of shape [size of current layer, 1])
        :return: Z – the linear component of the activation function (i.e., the value before applying the non-linear function)
        linear_cache – a dictionary containing A, W, b (stored for making the backpropagation easier to compute)
    '''

    # A is in fact the original X features after activations, so layer 1 get the original X vector (as there is no
    # activation yet)
    Z = np.dot(W.T, A) + b
    linear_cache = {'A': A, 'W': W, 'b': b}
    return Z, linear_cache


def softmax(Z):
    '''

    :param Z:the linear component of the activation function
    :return:
    '''

    A = np.exp(Z) / (np.sum(np.exp(Z)))
    activation_cache = Z
    return A, activation_cache


def relu(Z):
    '''

    :param Z:the linear component of the activation function
    :return: A – the activations of the layer
    activation_cache – returns Z, which will be useful for the backpropagation

    '''

    A = np.maximum(Z, 0)
    activation_cache = Z

    return A, activation_cache


if __name__ == "__main__":
    z_relu = np.maximum([1,2,-3],0)

    zz = np.zeros((6, 1))

    # row vector
    dummy_X = np.random.uniform(0, 255, 2).reshape(-1, 1)  # mnist image
    # print('1')

    net_dims = [2, 4, 10]
    w_b_params_dic = initialize_parameters(net_dims)
    Z, lin_cache = linear_forward(dummy_X, w_b_params_dic['W1'], w_b_params_dic['b1'])
    print(Z)

    A, activation_cache = softmax(Z)

    print(A, activation_cache)
