from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np


def check_list_types(checked_list: List, checked_type: Any) -> bool:
    """Checks if all the elements of a list are of the specified type"""
    if not isinstance(checked_list, list):
        return False
    return all(isinstance(checked_list_element, checked_type) for checked_list_element in checked_list)


def show_image(im: np.ndarray, title: str, *, cmap: str = 'gray') -> None:
    """Plots the input image with pyplot"""
    # axes for debugging purposes
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.imshow(im, cmap=cmap)
    plt.show()


def normalize_img(image: np.ndarray) -> np.ndarray:
    """Transform an image to 0-255 range"""
    print(f"min = {np.amin(image)}, {np.amax(image)}")
    zero_image = image - np.amin(image)
    print(f"max {np.amax(zero_image)}")
    return (zero_image * 255.0 / np.amax(zero_image)).astype('uint8')
