# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 18:43:19 2021

@author: Carlos
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import os
from scipy.ndimage.measurements import label


def _x_division(img, x_projection, x_labeled, x_ncomponents, *, pad=[3] * 4,
               debug=False):
    """Function that divides an image into vertical components"""
    # Array of segmented images
    crop_imgs = [None] * x_ncomponents
    # Array of level of sub/superscript
    levels = [None] * x_ncomponents
    levels[0] = 0
    if debug:
        print('performing vertical division')
        # Axes for cropped images
        rows = int(np.sqrt(x_ncomponents))
        cols = x_ncomponents / rows
        if np.isclose(int(cols), cols):
            cols = int(cols)
        else:
            cols = int(cols) + 1
        fig, axs = plt.subplots(rows, cols)
        
    
    for idx in range(x_ncomponents):            
        x, y = np.argmax(x_labeled == idx + 1), 0
        w, h = np.count_nonzero(x_labeled == idx + 1), img.shape[0]
        crop_img = cv.copyMakeBorder(img[y:y+h, x:x+w], 0, 0, pad[2], pad[3],
                                     cv.BORDER_CONSTANT, value=255)
        
        _, img_inv = cv.threshold(crop_img, 80, 255, cv.THRESH_BINARY_INV)
        M = cv.moments(img_inv)
        cy = int(M['m01']/M['m00']) 
        if debug:
            print(f'centroid of element {idx} is: {cy}')
        
        

        if idx > 0:
            if cy0 - cy > diff_min:
                levels[idx] = levels[idx-1] + 1
                if debug:
                    print("level up")
                    print(f'current level: {levels[idx]}')
            elif cy - cy0 > diff_max:
                levels[idx] = levels[idx-1] - 1
                if debug:
                    print("level down")
                    print(f'current level: {levels[idx]}')
            else:
                levels[idx] = levels[idx-1]
                if debug:
                    print('same level')
                    print(f'current level: {levels[idx]}')
        
        # References for next component to compare levels
        if idx < x_ncomponents - 1:
            cy0 = cy
            # Find highest black pixel
            y_min = np.min(np.where(np.amax(img_inv, axis=1)))
            y_max = np.max(np.where(np.amax(img_inv, axis=1)))
            diff_min = cy0 - y_min
            diff_max = y_max - cy0
            
        crop_imgs[idx] = crop_img
        
        if debug:
            if rows > 1:
                axs[int(idx / cols)][idx % cols].imshow(crop_imgs[idx], 
                                                        cmap='gray')
            else:
                axs[idx].imshow(crop_imgs[idx], cmap='gray')

    return (('x', levels), crop_imgs)


def _y_division(img, y_projection,y_labeled, y_ncomponents, *, pad=[3] * 4,
               debug=False):
    """Function that divides an image into horizontal components"""
    print(f'y_ncomponents: {y_ncomponents}')
    crop_imgs = [None] * y_ncomponents
    if debug:
        print('performing horizontal division')
        # Axes for cropped images
        rows = int(np.sqrt(y_ncomponents))
        cols = y_ncomponents / rows
        if np.isclose(int(cols), cols):
            cols = int(cols)
        else:
            cols = int(cols) + 1
        fig, axs = plt.subplots(rows, cols)
    
    for idx in range(y_ncomponents):
        x, y = 0, np.argmax(y_labeled == idx + 1)
        w, h = img.shape[1], np.count_nonzero(y_labeled == idx + 1)
        crop_img = cv.copyMakeBorder(img[y:y+h, x:x+w], pad[0], pad[1], 0, 0,
                                     cv.BORDER_CONSTANT, value=255)

        crop_imgs[idx] = crop_img
        
        if debug:
            if rows > 1:
                axs[int(idx / cols)][idx % cols].imshow(crop_imgs[idx], cmap='gray')
            else:
                axs[idx].imshow(crop_imgs[idx], cmap='gray')
                
    return (('y',), crop_imgs)

def _root_removal(img, img_inv, *, debug=False):
    """Function that splits radical from radicand"""
    if debug:
        print('performing radical-radican split')
    
    # Expect different returns depending on OpenCV version
    if int(cv.__version__.split('.')[0]) < 4:
        _, contours, _ = cv.findContours(img_inv, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)   
    else:
        contours, _ = cv.findContours(img_inv, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    bound_rects = [None]*len(contours)
    
    for i, c in enumerate(contours):
        bound_rects[i] = cv.boundingRect(c)
    
    # Avoid misconnection interpretation
    if len(bound_rects) > 1:
        # Find biggest box
        rect_areas = [w*h for (x, y, w, h) in bound_rects ]
        print(f"Bounding boxes {bound_rects}")
        big_rect_idx = np.argsort(rect_areas)[-1]
        
        # Split radical and radicand
        mask = np.ones(img.shape)
        mask = cv.drawContours(mask, contours, big_rect_idx, 0, cv.FILLED)
        radical = img.copy()
        radical[mask.astype(np.bool)] = 255 
        radicand = cv.drawContours(img, contours, big_rect_idx, 255, cv.FILLED)
        crop_imgs = [radical, radicand]
        if debug:
            # Axes for cropped images
            fig, axs = plt.subplots(2)
            axs[0].imshow(radical, cmap='gray')
            axs[1].imshow(radicand, cmap='gray')
    else:
        for i, _ in enumerate(contours):
            # Connect split symbol
            im2 = img.copy()
            cv.drawContours(im2, contours, i, (100, 100, 100), 1)
        crop_imgs = [im2]
    
    return (('r',), crop_imgs)
    
def _division_step(img, *, pad=[3] * 4, debug=False):
    """Function that controls the flow of image division"""
    _, img_inv = cv.threshold(img, 80, 255, cv.THRESH_BINARY_INV)
    # array to append cropped images
    print(f"division step debug is {debug}")
    
    # Get projection over x axis
    x_projection = np.matrix(np.amax(img_inv, axis=0))
    
    # Get all connected components in the projection
    structure = np.ones((3,3), dtype='uint8')
    x_labeled, x_ncomponents = label(x_projection, structure)   
    
            
    if x_ncomponents > 1:
        return _x_division(img, x_projection, x_labeled, x_ncomponents, pad=pad,
                          debug=debug)
    else:
        # Get projection over y axis
        y_projection = np.matrix(np.amax(img_inv, axis=1)).transpose()
        
        # Get all connected components in the projection
        y_labeled, y_ncomponents = label(y_projection, structure)
        
        if y_ncomponents > 1:
            
            return _y_division(img, y_projection, y_labeled, y_ncomponents, 
                              pad=pad, debug=debug)
            
        else:
            return _root_removal(img, img_inv, debug=debug)
            
    
def _show_image(im, title):
    # axes for debugging purposes
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.imshow(im, cmap='gray')
    

def xy_segmentation(img_path, *, pad=[3] * 4, debug=False):
    continue_division = True
    img = cv.imread(str(img_path), 0)
    # Apply padding 
    img = cv.copyMakeBorder(img, *pad, cv.BORDER_CONSTANT, value=255)
    
    # Prepare image and negative in grayscale
    _, img = cv.threshold(img, 230, 255, cv.THRESH_BINARY)

    imgs = [
            [
                [
                    ('b',)
                ],
                [img]
            ]
        ]
        
    while continue_division and len(imgs) < 4:
        # flag to determine whether the division is complete or not
        continue_division = False
        # Add new component to image list
        imgs.append([[],[]])
        for idx, im in enumerate(imgs[-2][1]):
            print(f'starting division of order {len(imgs) - 1} of picture {idx}')
            try:
                # Expect different returns depending on OpenCV version
                if int(cv.__version__.split('.')[0]) < 4:
                    _, contours, _ = cv.findContours(im, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)   
                else:
                    contours, _ = cv.findContours(im, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            except TypeError as error:
                print('Conflicting image is:')
                print(im)
                raise TypeError(error)
            # If there is more than one contour, division can still happen
            if len(contours) > 2:
                print(f'detected {len(contours) - 1} contours')
                im2 = im.copy()
                for i, _ in enumerate(contours):
                    cv.drawContours(im2, contours, i, (100, 100, 100), 1)
                continue_division = True 
                _show_image(im2, f'division of order {len(imgs) - 1} of picture {idx}')
                
                division_results = _division_step(im, pad=pad, debug=debug)
                imgs[-1][0].append(division_results[0])                        
                imgs[-1][1].extend(division_results[1]) 
            else:
                print('just one contour detected')
                imgs[-1][0].append(('n',))
                imgs[-1][1].append(im)
    return imgs

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
        segmentation_results = xy_segmentation(img_path, pad=pad, 
                                               debug=debug)