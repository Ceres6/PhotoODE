import pathlib
import json
from operator import itemgetter
import logging

import numpy as np

# base_dir = pathlib.Path(__file__).parents[0]
# sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from segmentation.xy_segmentation import xy_segmentation
from preprocessing.image_edition import image_to_square, resize_threshold
from classification.lenet import LeNet
from parsing.parser import XYParser
from solver.solver import Solver

# To run: python predict.py
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)  # select logging level
    pad = [3] * 4  # [top, bottom, left, right]
    
    # Model instantiation and loading
    base_dir = pathlib.Path(__file__).parents[0]
    dataset_dir = base_dir / 'dataset'

    # TODO: allow file selection
    with open([str(file) for file in dataset_dir.iterdir() if 'label_dict' in str(file)][0], "r") as f:
        labels = list(json.loads(f.read()).values())

    logging.debug(f"length of labels {len(labels)}")
    lenet = LeNet(labels)
    weights_dir = base_dir / 'classification' / 'weights'
    weights_file = sorted([path for path in weights_dir.iterdir()])[-1]
    lenet.load_weights(weights_file)
    input_shape = lenet.model.layers[0].input_shape[1:3]
    
    # Input directory
    img_dir = base_dir / 'dataset' / 'segmentation'
    save_dir = base_dir / 'segmentation' / 'segmented'
    
    # Find all images to segment
    for img_path in img_dir.iterdir():
        if img_path.is_dir():
            continue
        segmentation_results, segmentation_structure = xy_segmentation(img_path)
        # Deepcopy of segmentation results to store predictions and parse LaTeX
        # equation_structure = deepcopy(segmentation_results)
        # predictions_prob = [None] * len(segmentation_results.last_level)
        # TODO: Move array prediction to classifier
        predictions_results = []
        for group_index, image_group in enumerate(segmentation_results.segmentation_groups):
            for img_idx, img in enumerate(image_group.segmented_images):

                squared_img = image_to_square(img)
                resized_img = resize_threshold(squared_img, input_shape)

                # segmentation_results[-1][1][-1] = img

                # cv.imwrite(f"{save_dir}/seg.png", img)
                input_img = resized_img[np.newaxis, :, :, np.newaxis]
                prediction_array = lenet.predict(input_img)
                prediction_list = [[None, prob] for prob in prediction_array.tolist()[0]]
                for index, row in enumerate(prediction_list):
                    row[0] = labels[index]
                # sorted by precision
                prediction = sorted(prediction_list, key=itemgetter(1))[-1]
                predictions_results.append(prediction[0])
        logging.info(f'results are: {predictions_results}')
        latex_expression = XYParser(predictions_results, segmentation_structure).last_level.parsed_groups[0]
        latex_solution = Solver(latex_expression, 'y').latex_solution
        logging.info(f'solution latex is: {latex_solution}')
