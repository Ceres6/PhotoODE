from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

import pathlib

from LeNet import LeNet

# to run file: python ./Classification/train.py


def train_test_plt(fitted_model):
    fig = plt.figure(figsize=(15, 5))
    fig.add_subplot(1, 2, 1)

    plt.plot(fitted_model.history['val_loss'])
    plt.plot(fitted_model.history['loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Test', 'Train'], loc='upper left')

    fig.add_subplot(1, 2, 2)
    plt.plot(fitted_model.history['val_accuracy'])
    plt.plot(fitted_model.history['accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Test', 'Train'], loc='upper left')

    fig.subplots_adjust(wspace=0.2, hspace=1)


def accuracy_metrics(instance):
    # Calculate probability of each class for each image in test set
    y_pred = instance.model.predict(lenet.test_set[0], batch_size=32, verbose=1)
    # Take highest probability class
    y_pred_max = np.argmax(y_pred, axis=1)

    y_true = []
    for i in range(len(lenet.test_set[1])):
        y_true.append(np.argmax(lenet.test_set[1][i]))

    print(classification_report(y_true, y_pred_max))  # Comparamos los labels reales frente a los predichos
    # Precisión normalizada.
    print('Kappa de Cohen: ', cohen_kappa_score(y_true, y_pred_max))


def create_label_dict(classification_dir):
    lab_dict = {}
    for index, subdir in enumerate(classification_dir.iterdir()):
        # stores name of the label
        label = subdir.parts[-1]
        # encoding  and storage of label
        lab_dict[index] = label
    return lab_dict


if __name__ == '__main__':
    # Path to files
    file_path = pathlib.Path(__file__)
    label_dir = file_path.parents[1] / 'dataset' / 'classification'
    data_file = file_path.parents[1] / 'dataset' / 'classification_set.npz'

    # labels dict and array creation
    label_dict = create_label_dict(label_dir)
    labels = label_dict.values()

    # Model instantiation
    lenet = LeNet(labels)

    # Model training
    fitted_lenet = lenet.train(data_file=data_file)

    # Training metrics
    #train_test_plt(fitted_lenet)
    #accuracy_metrics(lenet)
