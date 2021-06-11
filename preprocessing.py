import os
import json
import numpy as np

from utils import *
from sklearn.utils import shuffle

# path params
gt_path = '/Users/kevin/Desktop/speed_challenge/data/'
out_path = '/Users/kevin/Desktop/speed_challenge/data/dump/'
img_path = '/Users/kevin/Desktop/speed_challenge/data/frames/'

# img params
height = 50
width = 50
channels = 3

# create mask for splitting (test will be 10% of data)
train_mask = np.arange(7754)
test_mask = np.arange(7754, 7754+862)

# grab all file names
included_extensions = ['jpg']
file_names = [fn for fn in os.listdir(img_path) 
                if any(fn.endswith(ext) for ext in included_extensions)]

# store total number of images
num_imgs = len(file_names)

# initialize img array which will hold frames
X = np.empty((num_imgs, height, width, channels), dtype='float32')

print("\nConverting JPGs to numpy array...")
print("\n")

# loop and convert jpg to numpy array
for i in range(num_imgs):
    if i % 1000 == 0:
        print("\tProcessing img {}".format(i))
    filepath = os.path.join(img_path, file_names[i])
    img = preprocess_img(filepath, desired_dims=(height, width))
    X[i] = img

print("\nConverting groundTruth labels to numpy array...")

# this part is for the groundTruth labels
with open(gt_path + 'drive.json') as f:
    data = json.load(f)

# convert to numpy array
data = np.asarray(data)

# extract speed
y = data[:, 1]

# shuffle
print("\nShuffling the data...")
X, y = shuffle(X, y, random_state=42)

# split into train and test
print("\nSplitting X into train and test...")
X_train = X[train_mask]
X_test = X[test_mask]

print("\nWriting X_train as HDF5...")
write_hdf5(X_train, out_path + "X_train_50.hdf5")

print("\nWriting X_test as HDF5...")
write_hdf5(X_test, out_path + "X_test_50.hdf5")

# split into train and test
print("\nSplitting y into train and test...")
y_train = y[train_mask]
y_test = y[test_mask]

# add dimension for scikit-learn api
y_train = np.expand_dims(y_train, axis=1)
y_test = np.expand_dims(y_test, axis=1)
print("\n\ty_train shape: {}".format(y_train.shape))
print("\n\ty_test shape: {}".format(y_test.shape))

print("\nWriting y_train as HDF5...")
write_hdf5(y_train, out_path + "y_train_50.hdf5")

print("\nWriting y_test as HDF5...")
write_hdf5(y_test, out_path + "y_test_50.hdf5")

print("\nDone!")
