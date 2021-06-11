import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

from utils import *
from keras.models import Model
from sklearn.manifold import TSNE
from sklearn.linear_model import Ridge
from keras.applications.vgg16 import VGG16
from matplotlib.ticker import NullFormatter
from sklearn.decomposition import TruncatedSVD

# ========================================================= #
# FILL IN THE FOLLOWING TO TEST MY CLASSIFIER
# dims
num_imgs = # num imgs in test data
height = 50
width = 50
channels = 3
desired_dims = (height, width) # or 200x200

dump_path = # path containing the classifier i provided, it's called 200_clf_4000.sav
test_path = # path to test image frames
gt_path = # test to json file containing correct labels of speed
json_name = # name of json file for test data
# ========================================================= #

def load_frames_from_dir(imgs_path, desired_dims):
	# get test frame names
	included_extenstions = ['jpg']
	file_names = [fn for fn in os.listdir(img_path) 
							if any(fn.endswith(ext) for ext in included_extenstions)]

	# store total number of images
	num_imgs = len(file_names)

	# initialize img array which will hold frames
	X = np.empty((num_imgs, height, width, channels), dtype='float32')
	
	# loop and convert jpg to numpy array
	for i in range(num_imgs):
		filepath = os.path.join(imgs_path, file_names[i])
		img = preprocess_img(filepath, desired_dims=(height, width))
		X[i] = img

	return X

def load_labels_json(gt_path, json_name):
	with open(gt_path + json_name) as f:
		data = json.load(f)

	# convert to numpy array
	data = np.asarray(data)

	# extract speed
	y = data[:, 1]

	return y

def main():
	# load X_test
	test_data = load_frames_from_dir(test_path, desired_dims)

	# load labels
	y_test = load_labels_json(gt_path, json_name)

	# instantiate vgg model
	print("Loading VGG16 model...")
	base_model = VGG16(weights='imagenet', include_top=False)
	model = Model(input=base_model.input, output=base_model.get_layer('block2_pool').output)

	# compute test features
	print("Computing VGG16 features...")
	test_features = model.predict(test_data)
	test_features = np.reshape(test_features, [test_features.shape[0], -1])

	# load trained linear regression model
	print("Loading linear regression model...")
	clf = pickle.load(open(dump_path + "200_clf_4000.sav", "rb"))

	print("Predicting...")
	predictions = clf.predict(test_features)
	mse = np.mean((predictions - y_test) ** 2)
	print("MSE: {}".format(mse))

if __name__ == '__main__':
	main()
