from typing import List, Any

import matplotlib.pyplot as plt
import numpy as np


def show_image(im: np.ndarray, title: str, *, cmap: str = 'gray') -> None:
    # axes for debugging purposes
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.imshow(im, cmap=cmap)


def check_list_types(checked_list: List, checked_type: Any) -> bool:
    """Checks if all the elements of a list are of the specified type"""
    if not isinstance(checked_list, list):
        return False
    return all(isinstance(checked_list, checked_type))
