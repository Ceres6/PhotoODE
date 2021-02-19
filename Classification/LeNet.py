import keras
from keras.layers import Input, Conv2D, Activation, AvgPool2D, Dense, Flatten
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pathlib


# to run file: python ./Classification/LeNet.py


class LeNet:
    """
    Class to instantiate LeNet Neural Networks
    """

    def __init__(self, labels, activation='relu', optimizer='adam', lr=0.0001,
                 loss='categorical_crossentropy', input_shape=32):
        self.input_shape = input_shape
        self.optimizer = Adam(lr=lr)
        if optimizer != 'adam':
            if optimizer == 'sgd':
                self.optimizer = SGD(lr=lr)
            else:
                print(f'optimizer: {optimizer} not registered, using default: adam')

        self.model = keras.Sequential(
            [
                Input(shape=(input_shape, input_shape, 1)),
                Conv2D(filters=6, kernel_size=9, strides=1, padding="same"),
                Activation(activation),
                AvgPool2D(pool_size=(2, 2), strides=2, padding='same'),
                Conv2D(filters=16, kernel_size=9, strides=1, padding="same"),
                Activation(activation),
                AvgPool2D(pool_size=(2, 2), strides=2, padding='same'),
                Conv2D(filters=16, kernel_size=5, strides=1, padding="same"),
                Activation(activation),
                AvgPool2D(pool_size=(2, 2), strides=2, padding='same'),
                Flatten(),
                Dense(256),
                Dense(128),
                Dense(len(labels), activation='softmax')
            ]
        )
        # print(f'model summary: {self.model.summary()}')
        self.model.compile(optimizer=self.optimizer, loss=loss, metrics=['accuracy'])
        # atributes for training
        self.train_set = []
        self.test_set = []

    def train(self, data_file, batch_size=32, epochs=10):
        # load data
        database = np.load(data_file)
        dataset, labels = database['database'], database['label']
        labels = to_categorical(labels)
        # split dataset in train and test
        x_train, x_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.1, stratify=labels)
        x_train = x_train.reshape(x_train.shape[0], self.input_shape, self.input_shape, 1)
        x_test = x_test.reshape(x_test.shape[0], self.input_shape, self.input_shape, 1)
        self.train_set = [x_train, y_train]
        self.test_set = [x_test, y_test]
        return self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))


if __name__ == '__main__':
    print('just testing')
