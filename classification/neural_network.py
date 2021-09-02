import json
from datetime import datetime
from typing import Iterable, Union, Optional
from pathlib import Path
from operator import itemgetter
from abc import ABC, abstractmethod

from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from preprocessing.image_edition import image_to_square, resize_threshold


class NeuralNetwork(ABC):
    def __init__(self):
        self.history = {}
        self.train_set = []
        self.test_set = []
        pass

    @property
    @abstractmethod
    def input_shape(self) -> Iterable[int]:
        pass

    @property
    @abstractmethod
    def model(self) -> Model:
        pass

    @property
    @abstractmethod
    def labels(self) -> Iterable[str]:
        pass

    def get_train_test_sets(self, *, data_file: Union[str, Path]):
        database = np.load(data_file)
        dataset, labels = database['database'], database['label']
        labels = to_categorical(labels)
        # split dataset in train and test
        x_train, x_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.1, stratify=labels)
        x_train = x_train.reshape(x_train.shape[0], *self.input_shape)
        x_test = x_test.reshape(x_test.shape[0], *self.input_shape)
        self.train_set = (x_train, y_train)
        self.test_set = (x_test, y_test)

    def save_weights(self, file_path: Union[str, Path], *, save_format: str = 'h5'):
        self.model.save_weights(file_path, save_format=save_format)

    def load_weights(self, file: Union[str, Path]):
        self.model.load_weights(file)

    def train(self, *, data_file: Union[str, Path], batch_size: int = 32, epochs: int = 10,
              logs_dir: Optional[Union[str, Path]] = None, weights_dir=None):
        # load data
        self.get_train_test_sets(data_file=data_file)
        history = self.model.fit(*self.train_set, epochs=epochs, batch_size=batch_size, validation_data=self.test_set)
        self.history = history.history
        if logs_dir or weights_dir:
            now = datetime.now()
            date_string = now.strftime('%Y_%m_%d_%H_%M_%S')
            data_string = f'd={date_string}_e={epochs}_b={batch_size}'
        if logs_dir:
            with open(f'{logs_dir}/loss_history_{self.__class__.__name__.lower()}_{data_string}.log', 'w') as file:
                file.write(json.dumps(self.history, indent=4, sort_keys=True))
            accuracy, cohen_kappa = self.accuracy_metrics()
            with open(f'{logs_dir}/accuracy_metrics_{self.__class__.__name__.lower()}_{data_string}.log', 'w') as file:
                file.write(accuracy)
                file.write(f'Cohen Kappa score : {cohen_kappa}')
        if weights_dir:
            self.save_weights(f'{weights_dir}/{self.__class__.__name__.lower()}_weights_{data_string}.h5',
                              save_format='h5')

    def predict(self, img: np.ndarray):
        return self.model.predict(img)

    def predict_array(self, array: Iterable[np.ndarray]):
        predictions_results = [None] * len(array)
        for img_idx, img in enumerate(array):

            squared_img = image_to_square(img)
            resized_img = resize_threshold(squared_img, self.input_shape[:2])
            input_img = resized_img[np.newaxis, :, :, np.newaxis]
            prediction_array = self.predict(input_img)
            prediction_list = [[None, prob] for prob in prediction_array.tolist()[0]]
            for index, row in enumerate(prediction_list):
                row[0] = self.labels[index]
            # sorted by precision
            prediction = sorted(prediction_list, key=itemgetter(1))[-1]
            predictions_results[img_idx] = prediction[0]
        return predictions_results

    def accuracy_metrics(self):
        y_pred = self.model.predict(self.test_set[0], batch_size=32, verbose=1)
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


