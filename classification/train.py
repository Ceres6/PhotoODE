import pathlib
import json

from lenet import LeNet


if __name__ == '__main__':
    file_path = pathlib.Path(__file__).absolute()
    data_dir = file_path.parents[1] / 'dataset'
    data_file = sorted(file for file in data_dir.iterdir() if 'classification_set' in file.name)[-1]
    label_file = sorted(file for file in data_dir.iterdir() if 'label_dict' in file.name)[-1]
    with open(label_file, 'r') as f:
        label_dict = json.loads(f.read())

    labels = label_dict.values()

    # Model instantiation
    lenet = LeNet(labels)

    # Model training
    fitted_lenet = lenet.train(data_file=data_file)

    # Training metrics
    # train_test_plt(fitted_lenet)
    # accuracy_metrics(lenet)
