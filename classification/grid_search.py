import pathlib
import json
from typing import Union, Iterable

import numpy as np
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from lenet import LeNet

file_path = pathlib.Path(__file__).absolute()
data_dir = file_path.parents[1] / 'dataset'
data_file = sorted(file for file in data_dir.iterdir() if 'classification_set' in file.name)[-1]
label_file = sorted(file for file in data_dir.iterdir() if 'label_dict' in file.name)[-1]
with open(label_file, 'r') as f:
    label_dict = json.loads(f.read())

labels_values = label_dict.values()


def create_model(lr):
    lenet = LeNet(labels_values, lr=lr)
    return lenet.model


# TODO: Allow other NeuralNetwork subclasses
def grid_search(*, epochs, lr, **kwargs):
    model = KerasClassifier(build_fn=create_model)
    print(kwargs)
    param_grid = dict(epochs=epochs, lr=lr, **kwargs)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    # TODO: get sets without class instantiation (classmethod?)
    lenet = LeNet(labels_values)
    lenet.get_train_test_sets(data_file=data_file)
    x, y = lenet.train_set
    # x_test, y_test = lenet.test_set
    # x = (*x_train, *x_test)
    # y = (*y_train, *y_test)
    return grid.fit(x, y)


if __name__ == '__main__':
    lrs = [0.001, 0.0001, 0.00001]
    batch_sizes = [64, 128]
    grid_result = grid_search(epochs=[10], lr=lrs, batch_size=batch_sizes)
    print(grid_result.best_params_)
    print(grid_result.best_score_)
    scores = [x[1] for x in grid_result.grid_scores_]
    scores = np.array(scores).reshape(len(lrs), len(batch_sizes))

    for ind, i in enumerate(lrs):
        plt.plot(batch_sizes, scores[ind], label='lr: ' + str(i))
        plt.legend()
        plt.xlabel('Batch Size')
        plt.ylabel('Mean score')
        plt.show()
