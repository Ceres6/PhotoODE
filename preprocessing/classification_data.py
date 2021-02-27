import os
import pathlib
import sys
from datetime import datetime
from typing import Dict

import numpy as np
import cv2 as cv

base_dir = pathlib.Path(__file__).parents[1]
sys.path.append(str(base_dir))
from preprocessing.image_edition import image_to_square, resize_threshold

# to run: python ./preprocessing/classification_data.py
# path to dataset directory


crohme_dir = base_dir / 'dataset' / 'crohme_by_class'
nist_dir = base_dir / 'dataset' / 'nist_by_class'
# array to compress images
array_of_images = []
# dictionary to store label encoding
label_dict: Dict[int, str] = {}
label_array = []
# classes not wanted in the clasifier
label_exceptions = [
        '!', '{', '}', '=', 'ascii_124', 'div', 'exists', 'forall', 'geq', 
        'gt', 'in', 'infty', 'int', 'ldots', 'leq', 'lim', 'lt', 'neq', 'pm', 
        'rightarrow', 'sum', 'times'
    ]
idx = -1
subdirs_crohme = [subdir for subdir in crohme_dir.iterdir() 
           if subdir.parts[-1] not in label_exceptions]

for subdir_crohme in subdirs_crohme:
    # stores name of the label
    label = subdir_crohme.parts[-1]
    # encoding  and storage of label
    # prime class considered as ,
    if label != 'prime':
        idx += 1
        label_dict[idx] = label.strip('_').lower()
        index = idx
    else:
        label = ','
        index = [key for (key, value) in label_dict.items() if value == ','][0]

    print(f'saving images of crohme class {label}')
    for file in subdir_crohme.iterdir():
        # converts image to array and stores it with its label and rescales to 32x32
        img = cv.imread(str(file), 0)
        img2 = resize_threshold(img, (32, 32))
        # im = Image.open(file).resize((32, 32), Image.LANCZOS)
        single_array = np.array(img2)
        array_of_images.append(single_array)
        label_array.append(index)


for subdir_nist in nist_dir.iterdir():
    label = subdir_nist.parts[-1].strip('_').lower()
    if label in label_dict.values():
        index = list(label_dict.keys())[list(label_dict.values()).index(label)]
    else:
        idx += 1
        label_dict[idx] = label.strip('_').lower()
        index = idx
        
    print(f'saving images of nist class {label}')
    for file in subdir_nist.iterdir():
        if file.is_dir():
            continue
        # converts image to array and stores it with its label and rescales to 32x32
        img = cv.imread(str(file), 0)
        try:
            img2 = resize_threshold(img, (32, 32))
        except Exception as e:
            print(f"conflicting image is: {file}")
            raise(e)
        # im = Image.open(file).resize((32, 32), Image.LANCZOS)
        single_array = np.array(img2)
        array_of_images.append(single_array)
        label_array.append(index)
        
label_array = np.array(label_array)

# Saves compressed images
now = datetime.now()
date_string = now.strftime("%Y_%m_%d_%H_%M_%S")

np.savez_compressed(base_dir / 'dataset' / f"classification_set_{date_string}.npz",
                    database=array_of_images, label=label_array)

# TODO: store label_dict automatically
