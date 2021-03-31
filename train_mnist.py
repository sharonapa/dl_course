import tensorflow as tf

from my_network import L_layer_model


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
    (x_train, y_train), (x_test, y_test) = get_mnist()
    X = x_train
    Y = y_train
    layers_dims = [X.shape[1], 20, 7, 5, 10]

    X = X.T
    Y = Y.T
    w_b_parmas_dic, costs = L_layer_model(X, Y, layers_dims, learning_rate=0.009, num_iterations=100, batch_size=4,
                                          use_batchnorm=False)
    print("costs: ", costs)


if __name__ == "__main__":
    train_mnist()
