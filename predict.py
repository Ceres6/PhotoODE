import pathlib
import sys
import cv2 as cv
import numpy as np

base_dir = pathlib.Path(__file__).parents[0]
sys.path.append(str(pathlib.Path(__file__).parents[0]))

from utils.utils import show_image
from segmentation.xy_segmentation import xy_segmentation
from preprocessing.image_edition import bounding_square, resize_threshold
from classification.LeNet import LeNet

# To run: python predict.py
if __name__ == '__main__':
    
    debug = True
    pad = [3] * 4  # [top, bottom, left, right]
    
    # Model instantiation and loading
    labels = [
        '(', ')', '+', ',', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', 
        '9', 'A', 'alpha', 'b', 'beta', 'C', 'cos', 'd', 'Delta', 'e', 'f', 
        'forward_slash', 'G', 'gamma', 'H', 'i', 'j', 'k', 'l', 'lambda', 
        'log', 'M', 'mu', 'N', 'o', 'p', 'phi', 'pi', 'q', 'R', 'S', 'sigma', 
        'sin', 'sqrt', 'T', 'tan', 'theta', 'u', 'v', 'w', 'X', 'y', 'z', '[',
        ']'
        ]
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
        # TODO: check if it's image file
        if img_path.is_dir():
            continue
        print(f"image path: {img_path}")
        segmentation_results = xy_segmentation(img_path, pad=pad, debug=debug)
        img = segmentation_results[-1][1][-1]
        img = bounding_square(img, debug=False)
        img = resize_threshold(img, input_shape)
        segmentation_results[-1][1][-1] = img
        
        cv.imwrite(f"{save_dir}/seg.png", img)
        img = img[np.newaxis, :, :, np.newaxis]
        prediction = lenet.predict(img)
            
        print(prediction)
        """
        for idx, img in enumerate(segmentation_results[-1][1]):
            segmentation_results[-1][1][idx] = bounding_square(img)"""
