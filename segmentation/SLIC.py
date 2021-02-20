import cv2 as cv
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import pathlib

def superpixel_split(thresh_img, save_dir, n_segments=100, sigma=5.0, compactness=5, order=0,):
    segments = slic(thresh_img, n_segments=n_segments, compactness=compactness, sigma=sigma)
    for (i, segVal) in enumerate(np.unique(segments)):
        # construct a mask for the segment
        # print (f"[x] inspecting segment {i}")
        mask = np.ones(thresh_img.shape, dtype="uint8") * 255
        mask[segments == segVal] = 0
        # show the masked region
        # cv.imwrite(f"Test_Mask_{i}.png", mask)
        masked_img = cv.bitwise_or(thresh_img, mask)
        zero_threshold = masked_img.shape[0]*masked_img.shape[1]/10e3
        if np.count_nonzero(masked_img == 0) > zero_threshold:
            cv.imwrite(f"{save_dir}/Applied_Mask_{i*10**order}.png", masked_img)
    
file_path = pathlib.Path(__file__)
img_dir = file_path.parents[1] / 'dataset' / 'segmentation'
saving_dir = file_path.parents[0] / 'segmented'

for img_path in img_dir.iterdir():
    img = cv.imread(str(img_path.absolute()))

    ret, thresh_img = cv.threshold(img, 80, 255, cv.THRESH_BINARY)


    # titles = ['Original Image', 'BINARY']
    # images = [img, thresh_img]
    
    # for i in range(2):
    #     plt.subplot(2,3,i+1)
    #     plt.imshow(images[i],'gray',vmin=0,vmax=255)
    #     plt.title(titles[i])
    #     plt.xticks([])
    #     plt.yticks([])
    
    # plt.subplot(2,3,3)
    # plt.imshow(mark_boundaries(thresh_img, segments))
    # plt.title(titles[i])
    # plt.xticks([])
    # plt.yticks([])
    
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.imshow(mark_boundaries(thresh_img, segments, color=(255,0,0)))
    # plt.axis("off")
    
    # loop over the unique segment values
    print(f'saving directory is: {saving_dir}')
    superpixel_split(thresh_img, save_dir=saving_dir)
    
    for idx, img_path in enumerate(saving_dir.iterdir()):
        img = cv.imread(str(img_path.absolute()))
        superpixel_split(img, save_dir=saving_dir, order=2 * (idx + 1))
