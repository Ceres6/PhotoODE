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


def x_division(img, img_inv, x_projection, x_labeled, x_ncomponents, debug=False):
    """Function that divides an image into vertical components"""
    crop_imgs = []
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
        crop_imgs.append(crop_img)
        if debug:
            if rows > 1:
                axs[int(idx / cols)][idx % cols].imshow(crop_imgs[-1], cmap='gray')
            else:
                axs[idx].imshow(crop_imgs[-1], cmap='gray')
    
    return crop_imgs


def y_division(img, img_inv, y_projection,y_labeled, y_ncomponents, debug=False):
    """Function that divides an image into horizontal components"""
    crop_imgs = []
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
        # print(f'appending image of size {crop_img.shape}')
        crop_imgs.append(crop_img)
        if debug:
            if rows > 1:
                axs[int(idx / cols)][idx % cols].imshow(crop_imgs[-1], cmap='gray')
            else:
                axs[idx].imshow(crop_imgs[-1], cmap='gray')
                
    return crop_imgs

def root_removal(img, img_inv, debug=False):
    """Function that splits radical from radicand"""
    crop_imgs = []
    if debug:
        print('performing radical-radican split')
        
    _, img_inv = cv.threshold(img, 80, 255, cv.THRESH_BINARY_INV)
    
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
    
    return crop_imgs
    
def division_step(img, debug=False):
    """Function that controls the flow of image division"""
    crop_imgs = []
    _, img_inv = cv.threshold(img, 80, 255, cv.THRESH_BINARY_INV)
    # array to append cropped images
    
    
    # Get projection over x axis
    x_projection = np.matrix(np.amax(img_inv, axis=0))
    
    # Get all connected components in the projection
    x_labeled, x_ncomponents = label(x_projection, structure)   
    
            
    if x_ncomponents > 1:
        crop_imgs.extend(x_division(img, img_inv, x_projection, x_labeled, 
                                       x_ncomponents, debug=debug))
    else:
        # Get projection over y axis
        y_projection = np.matrix(np.amax(img_inv, axis=1)).transpose()
        
        # Get all connected components in the projection
        y_labeled, y_ncomponents = label(y_projection, structure)
        
        if y_ncomponents > 1:
            
            crop_imgs.extend(y_division(img, img_inv, y_projection, y_labeled, 
                                        y_ncomponents, debug=debug))
            
        else:
            crop_imgs.extend(root_removal(img, img_inv, debug=debug))
            
            
    return crop_imgs       
    #for idx, image in enumerate() 
    
def show_image(im, title):
    # axes for debugging purposes
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.imshow(im, cmap='gray')
    
# Define global padding
pad = [3] * 4  # [top, bottom, left, right]
    
if __name__ == '__main__':
    file_path = pathlib.Path(__file__)
    img_dir = file_path.parents[1] / 'dataset' / 'segmentation'
    saving_dir = file_path.parents[0] / 'segmented'
    # Erase previous contents of the directory
    structure = np.ones((3,3), dtype='uint8')
    
    for file_path in saving_dir.iterdir():
        os.unlink(file_path) 
    # Find all images to segment
    for img_path in img_dir.iterdir():
        continue_division = True
        img = cv.imread(str(img_path), 0)
        # Apply padding 
        img = cv.copyMakeBorder(img, *pad, cv.BORDER_CONSTANT, value=255)
        
        # Prepare image and negative in grayscale
        _, img = cv.threshold(img, 230, 255, cv.THRESH_BINARY)

        imgs = [
                [
                    [1],
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
                    show_image(im2, f'division of order {len(imgs) - 1} of picture {idx}')
                    
                    new_img = division_step(im, debug=True)
                    imgs[-1][0].append(len(new_img))
                    imgs[-1][1].extend(new_img) 
                else:
                    print('just one contour detected')
                    imgs[-1][0].append(1)
                    imgs[-1][1].append(im)
        