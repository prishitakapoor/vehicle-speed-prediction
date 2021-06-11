import h5py
import numpy as np
from PIL import Image
from keras.applications import vgg16

def write_hdf5(arr, outfile):
	"""
	Write an numpy array to a file in HDF5 format.
	"""
	with h5py.File(outfile, "w", libver='latest') as f:
		f.create_dataset("image", data=arr, dtype=arr.dtype)

def load_hdf5(infile):
	"""
	Load a numpy array stored in HDF5 format into a numpy array.
	"""
	with h5py.File(infile, "r", libver='latest') as hf:
		return hf["image"][:]

def img_to_array(data_path, desired_size=None):
	"""
	Util function for loading RGB image into 3D numpy array.
	Returns array of shape (H, W, C)

	References
	----------
	- adapted from keras preprocessing/image.py
	"""
	img = Image.open(data_path)
	img = img.convert('RGB')
	if desired_size:
		img = img.resize((desired_size[1], desired_size[0]))
	x = np.asarray(img, dtype='float32')
	return x

def preprocess_img(img_path, desired_dims):
	"""
	Loads image using img_to_array, expands it to 4D tensor
	of shape (1, H, W, C), preprocesses it for use in the
	VGG16 network and resequeezes it to a 3D tensor.

	References
	----------
	- adapted from keras preprocessing/image.py
	"""
	img = img_to_array(data_path=img_path, desired_size=desired_dims)
	img = np.expand_dims(img, axis=0)
	img = vgg16.preprocess_input(img)
	img = np.squeeze(img)
	return img

def array_to_img(x):
	"""
	Util function for converting 4D numpy array to numpy array.

	Returns PIL RGB image.

	References
	----------
	- adapted from keras preprocessing/image.py
	"""
	x = np.asarray(x)
	x += max(-np.min(x), 0)
	x_max = np.max(x)
	if x_max != 0:
		x /= x_max
	x *= 255
	return Image.fromarray(x.astype('uint8'), 'RGB')
