from typing import Iterable, Tuple
import logging
import cv2
import numpy as np
from scipy import ndimage

from utils.utils import show_image

from settings import LOG_LEVEL


def bounding_square(objs: Iterable[np.ndarray]) -> Tuple[int]:
    """Returns a single bounding rectangle that contains all input objects"""
    x = min([obj[1].start for obj in objs])
    x_stop = max([obj[1].stop for obj in objs])
    w = x_stop - x
    y = min([obj[0].start for obj in objs])
    y_stop = max([obj[0].stop for obj in objs])
    h = y_stop - y
    return x, y, w, h


def image_to_square(img: np.ndarray, *, pad: Iterable[int] = [1]*4, white_value=255, threshold=80, debug=False) -> np.ndarray:
    """Function to broadcast an image into a white-background squared_image"""
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected img of type np.ndarray instead found {type(img)}")

    _, img_inv = cv2.threshold(img, threshold, white_value, cv2.THRESH_BINARY_INV)
    
    structure = np.ones((3, 3), dtype='uint8')
    img_labeled, img_ncomponents = ndimage.label(img_inv, structure)

    objs = ndimage.find_objects(img_labeled)
    x, y, w, h = bounding_square(objs)
    
    if w < h:
        crop_img = cv2.copyMakeBorder(img[y:y + h, x:x + w], pad[0], pad[1], 0, 0,
                                      cv2.BORDER_CONSTANT, value=white_value)
    elif w > h:
        crop_img = cv2.copyMakeBorder(img[y:y + h, x:x + w], 0, 0, pad[2], pad[3],
                                      cv2.BORDER_CONSTANT, value=white_value)
    else:
        crop_img = cv2.copyMakeBorder(img[y:y + h, x:x + w], pad[0], pad[1], pad[2],
                                      pad[3], cv2.BORDER_CONSTANT, value=white_value)
    height, width = crop_img.shape
    side = max(height, width)
    
    squared_image = np.ones((side, side), np.uint8) * white_value
    try:
        squared_image[int((side - height) / 2):height + int((side - height) / 2),
                      int((side - width) / 2):width + int((side - width) / 2)] = crop_img
    except ValueError as e:
        logging.error(f"side: {side}, height: {height}, width: {width}")
        logging.error(f"{int((side - height) / 2)}:{height + int((side - height) / 2)}")
        logging.error(f"{int((side - width) / 2)}:{width + int((side - width) / 2)}")
        raise e
    if debug:
        show_image(crop_img, "original")
        show_image(squared_image, "squared_image")
    
    return squared_image


def resize_threshold(image: np.ndarray, size: int, *, normalize=False, debug=False) -> np.ndarray:
    """
    Returns the image resized to the input size and thresholded to a value with one connected component if possible
    """
    if normalize:
        image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        white_value = 1
        threshold_values = np.arange(0.3, white_value, 0.1)
    else:
        white_value = 255
        threshold_values = range(80, white_value, 25)

    resized_img = cv2.resize(image, size, interpolation=cv2.INTER_LANCZOS4)

    structure = np.ones((3, 3), dtype='uint8')
    for threshold in threshold_values:
        _, img_inv = cv2.threshold(resized_img, threshold, white_value, cv2.THRESH_BINARY_INV)
        _, ncomponents = ndimage.label(img_inv, structure)   
        if ncomponents == 1:
            break
    if LOG_LEVEL == 'DEBUG':
        show_image(resized_img, "resized")
    _, resized_img = cv2.threshold(resized_img, threshold, white_value, cv2.THRESH_BINARY)
    return resized_img
