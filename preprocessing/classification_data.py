from PIL import Image
import os
import numpy as np
import pathlib

# to run: python ./preprocessing/classification_data.py
# path to dataset directory
file_path = pathlib.Path(__file__)
main_dir = file_path.parents[1] / 'dataset'
classification_dir = main_dir / 'classification'

# array to compress images
array_of_images = []
# dictionary to store label encoding
label_dict = {}
label_array = []
# classes not wanted in the clasifier
label_exceptions = [
        '!', '{', '}', '=', 'ascii_124', 'div', 'exists', 'forall', 'geq', 'gt', 'in', 'infty', 'int', 'ldots', 'leq',
        'lim', 'lt', 'neq', 'pm', 'rightarrow', 'sum', 'times'
    ]
idx = -1
subdirs = [subdir for subdir in classification_dir.iterdir() if subdir.parts[-1] not in label_exceptions]
for subdir in subdirs:
    # stores name of the label
    label = subdir.parts[-1]
    # encoding  and storage of label
    # prime class considered as ,
    if label != 'prime':
        idx += 1
        label_dict[idx] = label
        index = idx
    else:
        label = ','
        index = [key for (key, value) in label_dict.items() if value == ','][0]

    print(f'saving images of class {label}')
    for file in subdir.iterdir():
        # converts image to array and stores it with its label and rescales to 32x32
        im = Image.open(file).resize((32, 32), Image.LANCZOS)
        single_array = np.array(im)
        array_of_images.append(single_array)
        label_array.append(index)

label_array = np.array(label_array)

# Saves compressed images
np.savez_compressed(main_dir / "classification_set.npz", database=array_of_images, label=label_array)
