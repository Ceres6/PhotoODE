import pathlib
import json
from operator import itemgetter
from copy import deepcopy
import logging

import numpy as np
from typing import List

# base_dir = pathlib.Path(__file__).parents[0]
# sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from segmentation.xy_segmentation import xy_segmentation
from preprocessing.image_edition import image_to_square, resize_threshold
from classification.lenet import LeNet
# from classification.label_dict import label_dict
# from parser.parser import parse

# To run: python predict.py
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)  # select logging level
    debug = False
    pad = [3] * 4  # [top, bottom, left, right]
    
    # Model instantiation and loading
    base_dir = pathlib.Path(__file__).parents[0]
    dataset_dir = base_dir / 'dataset'
    # print(list(dataset_dir.iterdir()))

    with open([str(file) for file in dataset_dir.iterdir() if 'label_dict' in str(file)][0], "r") as f:
        labels = list(json.loads(f.read()).values())
        print(labels)
    # labels = list(label_dict.values())
    logging.debug(f"length of labels {len(labels)}")
    lenet = LeNet(labels)
    weights_dir = base_dir / 'classification' / 'weights'
    weights_file = sorted([path for path in weights_dir.iterdir()])[-1]
    lenet.load_weights(weights_file)
    # weights = lenet.model.get_weights()
    input_shape = lenet.model.layers[0].input_shape[1:3]
    
    # Input directory
    img_dir = base_dir / 'dataset' / 'segmentation'
    save_dir = base_dir / 'segmentation' / 'segmented'
    
    # Find all images to segment
    for img_path in img_dir.iterdir():
        if img_path.is_dir():
            continue
        # print(f"image path: {img_path}")
        segmentation_results = xy_segmentation(img_path, pad=pad, debug=debug)
        # Deepcopy of segmentation results to store predictions and parse LaTeX
        equation_structure = deepcopy(segmentation_results)
        predictions_prob = [None] * len(segmentation_results[-1][1])
        # TODO: Change double loop with itertools
        for group_index, image_group in enumerate(segmentation_results[-1][1]):
            for img_idx, img in enumerate(image_group):

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
                equation_structure[-1][1][group_index][img_idx] = prediction[0]
        logging.debug(f"prediction array: {equation_structure[-1]}")
        # latex_expression = parse(equation_structure)
        # print(latex_expression)
        # print(prediction)
        """
        for idx, img in enumerate(segmentation_results[-1][1]):
            segmentation_results[-1][1][idx] = bounding_square(img)"""
