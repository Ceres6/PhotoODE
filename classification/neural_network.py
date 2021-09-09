from typing import Iterable, Union
from pathlib import Path
from operator import itemgetter
from abc import ABC, abstractmethod


from tensorflow.keras import Model
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

    def load_weights(self, file: Union[str, Path]):
        self.model.load_weights(file)

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
