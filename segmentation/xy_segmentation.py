# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 18:43:19 2021

@author: Carlos Espa Torres
"""

import pathlib
from enum import IntEnum, unique
from typing import List, Tuple, Union
# import itertools
import logging
from copy import deepcopy

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.measurements import label

from utils.utils import show_image, check_list_types


@unique
class SegmentationOperation(IntEnum):
    BEGINNING = 0
    X_SEGMENTATION = 1
    Y_SEGMENTATION = 2
    ROOT_REMOVAL = 3
    NONE = 4


class SegmentationGroup:
    def __init__(self, segmentation_operation: SegmentationOperation,
                 segmented_images: List[np.ndarray], segmentation_levels: List[int] = None):
        if not isinstance(segmentation_operation, SegmentationOperation):
            raise TypeError('segmentation_operation arg must be of type SegmentationOperation')
        if not check_list_types(segmented_images, np.ndarray):
            raise TypeError('segmented_images arg must be of type List[np.ndarray]')
        if segmentation_operation is SegmentationOperation.X_SEGMENTATION:
            if not check_list_types(segmentation_levels, int):
                raise TypeError('segmentation_levels arg must be of type List[int]')
            if len(segmented_images) != len(segmentation_levels):
                raise ValueError('segmentation_levels and segmented_images must have same size for X_SEGMENTATION')
            if segmentation_levels[0] != 0:
                raise ValueError('first segmentation level must be zero')
        elif segmentation_levels is not None:
            raise ValueError('segmentation_levels must be None for non X_SEGMENTATION operations')
        self.__segmentation_operation: SegmentationOperation = segmentation_operation
        self.__segmented_images: List[np.ndarray] = segmented_images
        self.__segmentation_levels: List[int] = segmentation_levels

    def serialize(self):
        return {'segmentation_operation': self.segmentation_operation, 'segmented_images': self.segmented_images,
                'segmentation_levels': self.segmentation_levels}

    @property
    def segmentation_operation(self) -> SegmentationOperation:
        return self.__segmentation_operation

    @property
    def segmented_images(self) -> List[np.ndarray]:
        return self.__segmented_images

    @segmented_images.setter
    def segmented_images(self, a):
        self.__segmented_images = [a for img in self.__segmented_images]

    @property
    def segmentation_levels(self) -> List[int]:
        return self.__segmentation_levels


class SegmentationLevel:
    def __init__(self, segmentation_groups: Union[List[SegmentationGroup], SegmentationGroup] = None):
        if check_list_types(segmentation_groups, SegmentationGroup):
            self.__segmentation_groups: List[SegmentationGroup] = segmentation_groups
        elif isinstance(segmentation_groups, SegmentationGroup):
            self.__segmentation_groups = [segmentation_groups]
        elif segmentation_groups is None:
            self.__segmentation_groups = list()
        else:
            raise TypeError('segmentation_groups arg must be of type SegmentationGroup')

    def serialize(self):
        return {"segmentation_groups": [group.serialize() for group in self.segmentation_groups]}

    @property
    def segmentation_groups(self) -> List[SegmentationGroup]:
        return self.__segmentation_groups

    @segmentation_groups.setter
    def segmentation_groups(self, new_value: List[SegmentationGroup]):
        if check_list_types(new_value, SegmentationGroup):
            self.__segmentation_groups: List[SegmentationGroup] = new_value
        elif isinstance(new_value, SegmentationGroup):
            self.__segmentation_groups = [new_value]
        else:
            raise TypeError('new_value arg must be of type List[SegmentationGroup]')

    def add_group(self, new_group: SegmentationGroup):
        if not isinstance(new_group, SegmentationGroup):
            raise TypeError('new_group arg must be of type SegmentationGroup')
        self.__segmentation_groups.append(new_group)


class XYSegmentationResults:
    def __init__(self, image: np.ndarray):
        if not isinstance(image, np.ndarray):
            raise TypeError('image arg must be of type np.ndarray')
        self.image = image
        self.__continue_division = True
        self.__segmentation_levels: List[SegmentationLevel] = []

    def perform_segmentation(self):
        image_array = [self.image]
        segmentation_group = SegmentationGroup(SegmentationOperation.BEGINNING, image_array)
        self.add_level(SegmentationLevel([segmentation_group]))

        while self.__continue_division and len(self.segmentation_levels) < 4:
            # flag to determine whether the division is complete or not
            self.__continue_division = False
            self.__division_step()

    def serialize(self):
        return {"segmentation_levels": [level.serialize() for level in self.segmentation_levels]}


    @property
    def segmentation_levels(self) -> List[SegmentationLevel]:
        return self.__segmentation_levels

    # @segmentation_levels.setter
    # def segmentation_levels(self, levels: List[SegmentationLevel]):
    #     if not check_list_types(levels, SegmentationLevel):
    #         raise TypeError(f"Expected levels arg of type List[SegmentationLevel] but got {type(levels)}")
    #     self.__segmentation_levels = levels

    @property
    def last_level(self) -> SegmentationLevel:
        return self.__segmentation_levels[-1]

    @property
    def previous_level(self) -> SegmentationLevel:
        """Returns the previous level to the last one created"""
        return self.__segmentation_levels[-2]

    @property
    def segmented_images(self):
        return [image for group in self.last_level.segmentation_groups for image in group.segmented_images]

    def add_level(self, new_level: SegmentationLevel) -> None:
        if not isinstance(new_level, SegmentationLevel):
            raise TypeError('new_level arg must be of type SegmentationLevel')
        self.__segmentation_levels.append(new_level)

    def __division_step(self):
        """method to produce the next segmentation level"""
        self.add_level(SegmentationLevel())
        if self.last_level is self.previous_level:
            raise ValueError('same reference in distinct levels')
        logging.debug(f'there are {len(self.previous_level.segmentation_groups)} segmentation groups')
        for group_index, image_group in enumerate(self.previous_level.segmentation_groups):
            logging.debug(f'there are {len(image_group.segmented_images)} images in group {group_index}')
            for image_index, image in enumerate(image_group.segmented_images):
                logging.debug(''.join((f'starting division of order {len(self.segmentation_levels) - 1}',
                                       f' of picture {image_index} of group {group_index}')))
                try:
                    # Expect different returns depending on OpenCV version
                    if int(cv.__version__.split('.')[0]) < 4:
                        _, contours, _ = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                    else:
                        contours, _ = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

                except TypeError as error:
                    logging.error('Conflicting image is:')
                    logging.error(image)
                    raise TypeError(error)
                # If there is more than one contour, division can still happen
                if len(contours) > 2:
                    logging.debug(f'detected {len(contours) - 1} contours')
                    im2 = image.copy()
                    for i, _ in enumerate(contours):
                        cv.drawContours(im2, contours, i, (100, 100, 100), 1)
                    self.__continue_division = True
                    if logging.root.level == logging.DEBUG:
                        show_image(im2,
                                   f'division of order {len(self.segmentation_levels) - 1} of picture {image_index}')
                    self.last_level.add_group(self.__segment_image(image))
                else:
                    logging.debug('just one contour detected')
                    self.last_level.add_group(SegmentationGroup(SegmentationOperation.NONE, [image]))

    def __segment_image(self, img: np.ndarray) -> SegmentationGroup:
        _, img_inv = cv.threshold(img, 80, 255, cv.THRESH_BINARY_INV)
        # Get all connected components in the projection
        connection_structure = np.ones((3, 3), dtype='uint8')
        _, img_ncomponents = label(img_inv, connection_structure)
        if img_ncomponents < 2:
            logging.debug("Just one component detected")
            return SegmentationGroup(SegmentationOperation.NONE, [img])
        # Get projection over x axis
        x_projection = np.matrix(np.amax(img_inv, axis=0))

        x_labeled, x_ncomponents = label(x_projection, connection_structure)
        if x_ncomponents > 1:
            return self.__x_division(img, x_labeled, x_ncomponents)
        else:
            # Get projection over y axis
            y_projection = np.matrix(np.amax(img_inv, axis=1)).transpose()

            # Get all connected components in the projection
            y_labeled, y_ncomponents = label(y_projection, connection_structure)

            if y_ncomponents > 1:

                return self.__y_division(img, y_labeled, y_ncomponents)

            else:
                return self.__mask_removal(img, img_inv)

    @staticmethod
    def __x_division(img: np.ndarray, x_labeled: np.ndarray, x_ncomponents: int) -> SegmentationGroup:
        """Function that divides an image into vertical components"""
        # Array of segmented images
        segmented_images = [None] * x_ncomponents
        # Array of level of sub/superscript
        segmented_levels = [None] * x_ncomponents
        segmented_levels[0] = 0

        logging.debug('performing vertical division')
        if logging.root.level == logging.DEBUG:
            # Axes for cropped images
            rows = int(np.sqrt(x_ncomponents))
            cols = x_ncomponents / rows
            if np.isclose(int(cols), cols):
                cols = int(cols)
            else:
                cols = int(cols) + 1
            fig, axs = plt.subplots(rows, cols)

        for component_index in range(x_ncomponents):
            x, y = np.argmax(x_labeled == component_index + 1), 0
            w, h = np.count_nonzero(x_labeled == component_index + 1), img.shape[0]
            cropped_image = cv.copyMakeBorder(img[y:y + h, x:x + w], 0, 0, pad[2], pad[3],
                                              cv.BORDER_CONSTANT, value=255)

            _, img_inv = cv.threshold(cropped_image, 80, 255, cv.THRESH_BINARY_INV)
            moments = cv.moments(img_inv)
            y_centroid = int(moments['m01'] / moments['m00'])
            logging.debug(f'centroid of element {component_index} is: {y_centroid}')

            if component_index > 0:
                if (previous_y_centroid - y_centroid) * 0.9 > diff_min:
                    segmented_levels[component_index] = segmented_levels[component_index - 1] + 1
                    logging.debug("level up")
                    logging.debug(f'current level: {segmented_levels[component_index]}')
                elif (y_centroid - previous_y_centroid) * 0.9 > diff_max:
                    segmented_levels[component_index] = segmented_levels[component_index - 1] - 1
                    logging.debug("level down")
                    logging.debug(f'current level: {segmented_levels[component_index]}')
                    logging.debug(f"y_max: {y_max}, prev_y = {previous_y_centroid}, diff_max: {diff_max}, diff: {y_centroid - previous_y_centroid}")
                else:
                    segmented_levels[component_index] = segmented_levels[component_index - 1]
                    logging.debug('same level')
                    logging.debug(f'current level: {segmented_levels[component_index]}')

            # References for next component to compare levels
            if component_index < x_ncomponents - 1:
                previous_y_centroid = y_centroid
                # Find highest black pixel
                y_min = np.min(np.where(np.amax(img_inv, axis=1)))
                y_max = np.max(np.where(np.amax(img_inv, axis=1)))
                diff_min = previous_y_centroid - y_min
                diff_max = y_max - previous_y_centroid

            segmented_images[component_index] = cropped_image

            if logging.root.level == logging.DEBUG:
                if rows > 1:
                    axs[int(component_index / cols)][component_index % cols].imshow(segmented_images[component_index],
                                                                                    cmap='gray')
                else:
                    axs[component_index].imshow(segmented_images[component_index], cmap='gray')
        return SegmentationGroup(SegmentationOperation.X_SEGMENTATION, segmented_images,
                                 segmented_levels)  # ('x', segmented_levels), segmented_images

    @staticmethod
    def __y_division(img: np.ndarray, y_labeled: np.ndarray, y_ncomponents: int) -> SegmentationGroup:
        """Function that divides an image into horizontal components"""
        logging.debug(f'y_ncomponents: {y_ncomponents}')
        logging.debug('performing horizontal division')

        if logging.root.level == logging.DEBUG:
            # Axes for cropped images
            rows = int(np.sqrt(y_ncomponents))
            cols = y_ncomponents / rows
            if np.isclose(int(cols), cols):
                cols = int(cols)
            else:
                cols = int(cols) + 1
            fig, axs = plt.subplots(rows, cols)

        segmented_images = [None] * y_ncomponents
        for idx in range(y_ncomponents):
            x, y = 0, np.argmax(y_labeled == idx + 1)
            w, h = img.shape[1], np.count_nonzero(y_labeled == idx + 1)
            cropped_image = cv.copyMakeBorder(img[y:y + h, x:x + w], pad[0], pad[1], 0, 0,
                                              cv.BORDER_CONSTANT, value=255)

            segmented_images[idx] = cropped_image

            if logging.root.level == logging.DEBUG:
                if rows > 1:
                    axs[int(idx / cols)][idx % cols].imshow(segmented_images[idx], cmap='gray')
                else:
                    axs[idx].imshow(segmented_images[idx], cmap='gray')
        return SegmentationGroup(SegmentationOperation.Y_SEGMENTATION, segmented_images)  # ('y',), segmented_images

    @staticmethod
    def __mask_removal(img: np.ndarray, img_inv: np.ndarray) -> SegmentationGroup:
        """Function that splits radical from radicand"""
        logging.debug('performing radical-radicand split')

        # Expect different returns depending on OpenCV version
        if int(cv.__version__.split('.')[0]) < 4:
            _, contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        else:
            contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        bound_rects = [None] * len(contours)

        for i, c in enumerate(contours):
            bound_rects[i] = cv.boundingRect(c)

        # Avoid misconnection interpretation
        if len(bound_rects) > 1:
            # Find biggest box
            rect_areas = [w * h for (x, y, w, h) in bound_rects]
            logging.debug(f"Bounding boxes {bound_rects}")
            big_rect_idx = np.argsort(rect_areas)[-1]

            # Split radical and radicand
            mask = np.ones(img.shape)
            mask = cv.drawContours(mask, contours, big_rect_idx, 0, cv.FILLED)
            radical = img.copy()
            radical[mask.astype(np.bool)] = 255
            radicand = cv.drawContours(img, contours, big_rect_idx, 255, cv.FILLED)
            segmented_images = [radical, radicand]
            operation = SegmentationOperation.ROOT_REMOVAL  # ('r',)
            if logging.root.level == logging.DEBUG:
                # Axes for cropped images
                fig, axs = plt.subplots(2)
                axs[0].imshow(radical, cmap='gray')
                axs[1].imshow(radicand, cmap='gray')
        else:
            for i, _ in enumerate(contours):
                # Connect split symbol
                im2 = img.copy()
                cv.drawContours(im2, contours, i, (100, 100, 100), 1)
            segmented_images = [im2]
            # set operation to none
            operation = SegmentationOperation.NONE  # ('n',)
        return SegmentationGroup(operation, segmented_images)  # operation, segmented_images


def xy_segmentation(image_path: Union[pathlib.Path, str]) -> Tuple[List[np.ndarray], XYSegmentationResults]:
    """
    Returns the segmentation results and the segmentation structure of the image provided by path input
    """
    img = cv.imread(str(image_path), 0)
    # Apply padding
    img = cv.copyMakeBorder(img, *segmentation_padding, cv.BORDER_CONSTANT, value=255)

    # Prepare image in grayscale
    _, img = cv.threshold(img, 180, 255, cv.THRESH_BINARY)
    xy_segmenter = XYSegmentationResults(img)
    xy_segmenter.perform_segmentation()
    segmentation_results = [img for group in xy_segmenter.last_level.segmentation_groups for img in group.segmented_images]
    segmentation_structure = deepcopy(xy_segmenter)

    for level in segmentation_structure.segmentation_levels:
        for group in level.segmentation_groups:
            group.segmented_images = 0

    return segmentation_results, segmentation_structure


def dict_to_xy_segmentation_results(dict_) -> XYSegmentationResults:
    """Parses an input dictionary into an XYSegmentationResults Object"""
    segmentation_results = XYSegmentationResults(np.ndarray(0))
    for level in dict_['segmentation_levels']:
        groups = []
        for group in level['segmentation_groups']:
            segmentation_operation = SegmentationOperation(group['segmentation_operation'])
            segmented_images = [np.ndarray(obj) if isinstance(obj, int) else np.array(obj)
                                for obj in group['segmented_images']]
            segmentation_levels = group['segmentation_levels']
            groups.append(SegmentationGroup(segmentation_operation, segmented_images, segmentation_levels))
        segmentation_results.add_level(SegmentationLevel(groups))
    return segmentation_results


segmentation_padding = [3] * 4
pad = segmentation_padding
if __name__ == '__main__':

    debug = True
    pad = [3] * 4  # [top, bottom, left, right]
    file_path = pathlib.Path(__file__)
    img_dir = file_path.parents[1] / 'dataset' / 'segmentation'

    """
    saving_dir = file_path.parents[0] / 'segmented'
    # Erase previous contents of the directory
    
    for file_path in saving_dir.iterdir():
        os.unlink(file_path) 
    """
    # Find all images to segment
    for img_path in img_dir.iterdir():
        xy_segmentation_results = xy_segmentation(img_path)
