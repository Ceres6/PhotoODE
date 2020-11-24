# add your imports here
from __future__ import division
from __future__ import print_function
# own imports
import ntpath

# image resize 'added'
# from test_resize import test_resize

# custom classes imports
import kerasmodel as km
import segmentor2 as seg

# Separate ImgPred and SymPred to files
from SymPred import SymPred
from ImgPred import ImgPred

# original imports
from sys import argv
from glob import glob
from scipy import misc
import numpy as np
import random

"""
add whatever you think it's essential here
"""
# load keras model and label encoder
model, le = km.KerasModel.load()


def predict(image_path):
	"""
	Add your code here
	"""
	"""
	#Don't forget to store your prediction into ImgPred
	img_prediction = ImgPred(...)
	"""
	sym_preds = []
	inv_transform = []
	print("Predicting " + ntpath.basename(image_path))
	imgs, rects = seg.process(image_path)
	print(f'len(imgs): {len(imgs)}')
	for i in range(0, len(imgs)):
		# rectangle  i coordinates
		# recti = (rects[i][0], rects[i][1], rects[i][2], rects[i][3])
		# print(f'recti: {recti}')
		# argument for inverse_transform
		prediction = model.predict(imgs[i].reshape([-1, 32, 32, 1]))
		print(f'prediction {i}: {prediction}, of type: {type(prediction)}')
		itarg = np.array(np.argmax(prediction))
		print(f'itarg {i}: {itarg} of type {type(itarg)}')
		inv_transform.append(itarg)
	inv_trans = le.inverse_transform(inv_transform)   #itarg)
	print(f'inv_trans: {inv_trans} of type {type(inv_trans)}')
	# sym_pred = SymPred(inv_trans, *recti)
	# sym_preds.append(sym_pred)
	# print(str(sym_pred))

	# img_pred = ImgPred(ntpath.basename(image_path).split('.')[0], sym_preds)
	seg.cropped_imgs = []
	seg.cropped_rects = []
	return inv_trans  # img_pred

# no se por que esto estaba aqui
# return img_prediction


if __name__ == '__main__':
	image_folder_path = argv[1]
	isWindows_flag = False
	if len(argv) == 3:
		isWindows_flag = True
	if isWindows_flag:
		image_paths = glob(image_folder_path + '\\*png')
	else:
		image_paths = glob(image_folder_path + '/*png')
	# results = []
	# for image_path in image_paths:
	# 	impred = predict(image_path)
	#	results.append(impred)
	# lo voy a poner bonito
	results = [predict(im) for im in image_paths]

	with open('predictions.txt', 'w') as fout:
		for res in results:
			fout.write(str(res))

