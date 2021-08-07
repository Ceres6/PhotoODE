import cv2 as cv
import numpy as np
from scipy import ndimage

from utils.utils import show_image


def bounding_square(objs):
    x = min([obj[1].start for obj in objs])
    x_stop = max([obj[1].stop for obj in objs])
    w = x_stop - x
    y = min([obj[0].start for obj in objs])
    y_stop = max([obj[0].stop for obj in objs])
    h = y_stop - y
    return x, y, w, h


def image_to_square(img, *, pad=[1]*4, debug=False):
    """Fucntion to broadcast an image into a white-background square"""
    
    _, img_inv = cv.threshold(img, 80, 255, cv.THRESH_BINARY_INV)
    
    structure = np.ones((3, 3), dtype='uint8')
    img_labeled, img_ncomponents = ndimage.label(img_inv, structure) 
    
    # try:
    #     assert(img_ncomponents == 1)
    # except AssertionError:
    #     print(f"There are too many components: {img_ncomponents}")
    #     return
        
    objs = ndimage.find_objects(img_labeled)
    x, y, w, h = bounding_square(objs)
    
    if w < h:
        crop_img = cv.copyMakeBorder(img[y:y+h, x:x+w], pad[0], pad[1], 0, 0,
                                     cv.BORDER_CONSTANT, value=255)
    elif w > h:
        crop_img = cv.copyMakeBorder(img[y:y+h, x:x+w], 0, 0, pad[2], pad[3],
                                     cv.BORDER_CONSTANT, value=255)
    else:
        crop_img = cv.copyMakeBorder(img[y:y+h, x:x+w], pad[0], pad[1], pad[2], 
                                     pad[3], cv.BORDER_CONSTANT, value=255)
    height, width = crop_img.shape
    side = max(height, width)
    
    square = np.ones((side, side), np.uint8) * 255   
    try:
        square[int((side - height) / 2):height + int((side - height) / 2), 
               int((side - width) / 2):width + int((side - width) / 2)] = crop_img
    except ValueError as e:
        print(f"side: {side}, height: {height}, width: {width}")
        print(f"{int((side - height) / 2)}:{height + int((side - height) / 2)}")
        print(f"{int((side - width) / 2)}:{width + int((side - width) / 2)}")
        raise e
    # cv2.imwrite(out_img,square)
    if debug:
        show_image(crop_img, "original")
        show_image(square, "square")
    
    return square


def resize_threshold(image, size, *, debug=False):
    
    img = cv.resize(image, size, interpolation=cv.INTER_LANCZOS4)
 
    threshold_values = (80, 100, 150, 180, 200, 215, 230)
    structure = np.ones((3, 3), dtype='uint8')
    for threshold in threshold_values:
        _, img_inv = cv.threshold(img, threshold, 255, cv.THRESH_BINARY_INV)   
        _, ncomponents = ndimage.label(img_inv, structure)   
        if ncomponents == 1:
            break
    if debug:
        show_image(img, "resized")
    _, img = cv.threshold(img, threshold, 255, cv.THRESH_BINARY)
    return img
