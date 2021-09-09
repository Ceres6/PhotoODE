from typing import Iterable

import keras
from keras.layers import InputLayer, Conv2D, Activation, AvgPool2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam, SGD

from classification.neural_network import NeuralNetwork


class LeNet(NeuralNetwork):
    """Class to instantiate LeNet Neural Networks"""

    def __init__(self, labels: Iterable[str], *, activation: str = 'relu', optimizer: str = 'adam', lr: float = 0.0001,
                 loss: str = 'categorical_crossentropy', input_shape: float = 32):
        super().__init__()
        self._input_shape = (input_shape, input_shape, 1)
        self._labels = labels
        self.optimizer = Adam(learning_rate=lr)
        if optimizer != 'adam':
            if optimizer == 'sgd':
                self.optimizer = SGD(learning_rate=lr)
            else:
                print(f'optimizer: {optimizer} not registered, using default: adam')

        self._model = keras.Sequential(
            [
                InputLayer(input_shape=self._input_shape),
                Conv2D(filters=6, kernel_size=9, strides=1, padding='same'),
                Activation(activation),
                AvgPool2D(pool_size=(2, 2), strides=2, padding='same'),
                Conv2D(filters=16, kernel_size=9, strides=1, padding='same'),
                Activation(activation),
                AvgPool2D(pool_size=(2, 2), strides=2, padding='same'),
                Conv2D(filters=16, kernel_size=5, strides=1, padding='same'),
                Activation(activation),
                AvgPool2D(pool_size=(2, 2), strides=2, padding='same'),
                Flatten(),
                Dense(256),
                Dense(128),
                Dense(len(labels), activation='softmax')
            ]
        )
        self._model.compile(optimizer=self.optimizer, loss=loss, metrics=['accuracy'])

    @property
    def model(self):
        return self._model

    @property
    def labels(self):
        return self._labels

    @property
    def input_shape(self):
        return self._input_shape
