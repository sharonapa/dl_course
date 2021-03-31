import random
import tensorflow as tf
from matplotlib import cm

from my_network import L_layer_model
import matplotlib.pyplot as plt


def get_mnist():
    # Model / data parameters
    num_classes = 10
    input_shape = (28, 28, 1)

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape((-1, 784))
    x_test = x_test.reshape((-1, 784))
    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train_hot = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test_hot = tf.keras.utils.to_categorical(y_train, num_classes)
    return (x_train, y_train_hot), (x_test, y_test_hot)


def train_mnist():
    random.seed(123)

    #(x_train, y_train), (x_test, y_test) = get_mnist()
    # X = x_train
    # Y = y_train
    # X = X.T
    # Y = Y.T
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], 28 * 28)  # 784
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)

    x_train = x_train.astype('float64')
    x_test = x_test.astype('float64')

    x_train = x_train / 255
    x_test = x_test / 255

    y_train = y_train.reshape(x_train.shape[0], 1)
    y_test = y_test.reshape(x_test.shape[0], 1)

    num_classes = 10
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    plt.title((y_train[0]))
    plt.imshow(x_train[0].reshape(28, 28), cmap=cm.binary)


    layers_dims = [784, 20, 7, 5, 10]
    w_b_parmas_dic, costs = L_layer_model(x_train, y_train, layers_dims, learning_rate=0.009, num_iterations=50000, batch_size=100,
                                          use_batchnorm=False)
    print("costs: ", costs)


if __name__ == "__main__":
    train_mnist()
