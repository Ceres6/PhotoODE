# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 19:38:35 2021

@author: Carlos
"""
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import os


def contour_split(img, img_name, save_dir):
    """
    Find contours of an image and saves a new image for each contour found
    """
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)    
    ret, thresh = cv.threshold(imgray, 80, 255, cv.THRESH_BINARY)
    ret2, thresh_img = cv.threshold(img, 80, 255, cv.THRESH_BINARY)
    _, contours, hier = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    #cv.drawContours(thresh_img, contours, -1, (0,255,0), 3)
    # mask = np.ones(img.shape[:2], dtype="uint8") * 255
    segmented_symbols = []
    for index, contour in enumerate(contours):
        mask = np.ones(thresh_img.shape)
        mask = cv.drawContours(mask, contours, index, 0, cv.FILLED)
        
        # Generate output
        masked_img = thresh_img.copy()
        masked_img[mask.astype(np.bool)] = 255 
        # Threshold to throw away noise
        print(f'masked img shape {masked_img.shape}')
        zero_threshold = masked_img.shape[0]*masked_img.shape[1]/10e3
        if np.count_nonzero(masked_img == 0) > zero_threshold:
            segmented_symbols.append(masked_img)
            #cv.imwrite(f"{save_dir}/{img_name}_Contour_{index}.png", masked_img)
    return segmented_symbols

    
    
if __name__ == '__main__':

    file_path = pathlib.Path(__file__)
    img_dir = file_path.parents[1] / 'dataset' / 'segmentation'
    saving_dir = file_path.parents[0] / 'segmented'
    # Erase previous contents of the directory
    for file_path in saving_dir.iterdir():
        os.unlink(file_path) 
    # Find all images to segment
    for img_path in img_dir.iterdir():
        img = cv.imread(str(img_path.absolute()))
        
        symbols = contour_split(img, img_name=img_path.parts[-1], 
                                   save_dir=saving_dir)
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # ax.imshow(thresh_img)
        # plt.axis("off")