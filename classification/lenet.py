import json
from datetime import datetime
import pathlib
import os

import keras
from keras.layers import Input, Conv2D, Activation, AvgPool2D, Dense, Flatten
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
from keras.callbacks import History
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix

import matplotlib.pyplot as plt
import numpy as np


# to run file: python ./classification/lenet.py


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
        self.history = dict()

    def train(self, data_file, *, batch_size=32, epochs=10, logs_dir=None, weights_dir=None):
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
        history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))
        self.history = history.history
        if logs_dir or weights_dir:
            now = datetime.now()
            date_string = now.strftime("%Y_%m_%d_%H_%M_%S")
            data_string = f"d={date_string}_e={epochs}_b={batch_size}"
        if logs_dir:
            with open(f"{logs_dir}/loss_history_lenet_{data_string}.log", 'w') as file:
                file.write(json.dumps(self.history, indent=4, sort_keys=True))
            accuracy, cohen_kappa = self.accuracy_metrics()
            with open(f"{logs_dir}/accuracy_metrics_lenet_{data_string}.log", 'w') as file:
                file.write(accuracy)
                file.write(f'Cohen Kappa score : {cohen_kappa}')
        if weights_dir:
            self.save_weights(f"{weights_dir}/loss_history_lenet_{data_string}.h5", save_format='h5')

    def save_weights(self, file_path, *, save_format='h5'):
        self.model.save_weights(file_path, save_format=save_format)

    def load_weights(self, file):
        self.model.load_weights(file)

    def predict(self, img):
        return self.model.predict(img)

    def accuracy_metrics(self):
        # Calculate probability of each class for each image in test set
        y_pred = self.model.predict(self.test_set[0], batch_size=32, verbose=1)
        # Take highest probability class
        y_pred_max = np.argmax(y_pred, axis=1)

        y_true = []
        for i in range(len(self.test_set[1])):
            y_true.append(np.argmax(self.test_set[1][i]))

        accuracy = classification_report(y_true, y_pred_max)
        cohen_kappa = cohen_kappa_score(y_true, y_pred_max)
        return accuracy, cohen_kappa

    def train_test_plt(self):
        fig = plt.figure(figsize=(15, 5))
        fig.add_subplot(1, 2, 1)

        plt.plot(self.history['val_loss'])
        plt.plot(self.history['loss'])
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Test', 'Train'], loc='upper left')

        fig.add_subplot(1, 2, 2)
        plt.plot(self.history['val_accuracy'])
        plt.plot(self.history['accuracy'])
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Test', 'Train'], loc='upper left')

        fig.subplots_adjust(wspace=0.2, hspace=1)


if __name__ == '__main__':
    print('just testing')
