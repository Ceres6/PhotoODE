# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 12:24:30 2021

@author: Carlos
"""
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import os

def show_image(im, title):
    # axes for debugging purposes
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.imshow(im, cmap='gray')

# Define global padding
pad = [3] * 4

file_path = pathlib.Path(__file__)
img_dir = file_path.parents[1] / 'dataset' / 'segmentation'
saving_dir = file_path.parents[0] / 'segmented'
# Erase previous contents of the directory
structure = np.ones((3,3), dtype='uint8')

for file_path in saving_dir.iterdir():
    os.unlink(file_path) 
# Find all images to segment
for img_path in img_dir.iterdir():
    if 'root' in str(img_path):
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
        show_image(img, "original image")
        
"""Function that splits radical from radicand"""
crop_imgs = []
_, img_inv = cv.threshold(img, 80, 255, cv.THRESH_BINARY_INV)
if int(cv.__version__.split('.')[0]) < 4:
    _, contours, _ = cv.findContours(img_inv, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

else:
    contours, _ = cv.findContours(img_inv, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

print(f'there are {len(contours)} contours')
bound_rects = [None]*len(contours)

for i, c in enumerate(contours):
    print(f'type of contour {i} is: {type(c)}')
    bound_rects[i] = cv.boundingRect(c)
rect_areas = [w*h for (x, y, w, h) in bound_rects ]
# Find biggest box
#boundRect.sort(reverse=True, key=lambda tup: tup[2]*tup[3])
big_rect_idx = np.argsort(rect_areas)[-1]

# x, y, w, h = bound_rects[big_rect_idx]
# cv.rectangle(img,(x,y),(x+w,y+h),(100,100,100),2)
# show_image(img, "biggest rect")

index = big_rect_idx
contour = contours[index]

mask = np.ones(img.shape)
mask = cv.drawContours(mask, contours, index, 0, cv.FILLED)
counter = cv.drawContours(img, contours, index, 255, cv.FILLED)
show_image(counter, "counter")
# Generate output
masked_img = img.copy()
masked_img[mask.astype(np.bool)] = 255 
show_image(masked_img, "radical alone")
print(masked_img)
# Threshold to throw away noise
#segmented_symbols.append(masked_img)