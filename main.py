import time
import pickle
import numpy as np
from utils import *
from keras.models import Model
from keras.applications.vgg16 import VGG16
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD

# ensuring reproducibility
np.random.seed(1400)

# path params
dump_path = '/Users/kevin/Desktop/speed_challenge/data/dump/'

def load_data(path):

    # load training data
    X_train = load_hdf5(path + "X_train_50.hdf5")
    y_train = load_hdf5(path + "y_train_50.hdf5")

    # load test data
    X_test = load_hdf5(path + "X_test_200.hdf5")
    y_test = load_hdf5(path + "y_test_200.hdf5")

    return X_train, y_train, X_test, y_test

def get_vgg_features(dump_path):
    """
    Computes the features of the frame images from
    the 'block2_pool' layer of the VGG16 network.

    Writes them as HDF5 when done.
    """

    # load the data
    X_train, y_train, X_test, y_test = load_data(dump_path)

    # dimension sanity check
    print("Train samples: {}".format(X_train.shape))
    print("Test samples: {}".format(X_test.shape))

    base_model = VGG16(weights='imagenet', include_top=False)

    # # print layer names
    # for i, layer in enumerate(model.layers):
    #     print(i, layer.name)

    model = Model(input=base_model.input, output=base_model.get_layer('block2_pool').output)

    print("Computing train features...")
    train_features = model.predict(X_train)
    print("Train Features Shape: {}".format(train_features.shape))
    train_features = np.reshape(train_features, [train_features.shape[0], -1])

    print("Computing test features...")
    test_features = model.predict(X_test)
    print("Test Features Shape: {}".format(test_features.shape))
    test_features = np.reshape(test_features, [test_features.shape[0], -1])

    print("writing as HDF5...")
    write_hdf5(train_features, dump_path + "train_features_200.hdf5")
    write_hdf5(test_features, dump_path + "test_features_200.hdf5")

    return 

def main():

    # load the data
    X_train, y_train, X_test, y_test = load_data(dump_path)

    # compute vgg features and write to hdf5
    get_vgg_features(dump_path)
    
    print("Loading features from HDF5 files...")
    # train_features = load_hdf5(dump_path + "train_features_200.hdf5")
    test_features = load_hdf5(dump_path + "test_features_200.hdf5")
    y_train = load_hdf5(dump_path + "y_train_50.hdf5")
    y_test = load_hdf5(dump_path + "y_test_50.hdf5")

    print("Training Linear Regression model...")
    clf = Ridge(alpha=5.0)
    tic = time.time()
    clf.fit(train_features, y_train)
    toc = time.time()
    print("Time elapsed: {} seconds".format(toc - tic))
    print("Predicting...")
    predictions = clf.predict(test_features)
    mse = np.mean((predictions - y_test) ** 2)
    print("MSE: {}".format(mse))

    # dump model
    print("Dumping model...")
    pickle.dump(clf, open(dump_path + "50_clf_all.sav", "wb"))

    clf = pickle.load(open(dump_path + "200_clf_4000.sav", "rb"))
    print("Predicting...")
    predictions = clf.predict(test_features)
    mse = np.mean((predictions - y_test) ** 2)
    print("MSE: {}".format(mse))

    # print("Visualizing TSNE embedding...")
    # tic = time.time()
    # X_reduced = TruncatedSVD(n_components=50, random_state=0).fit_transform(train_features)
    # toc = time.time()
    # print("SVD time: {} seconds".format(toc - tic))
    # model = TSNE(n_components=2, perplexity=40, verbose=2, random_state=0)
    # tic = time.time()
    # Y = model.fit_transform(X_reduced)
    # toc = time.time()
    # print("TSNE time: {} seconds".format(toc - tic))
    # write_hdf5(Y, dump_path + "tsne_50.hdf5")

    # # visualize
    # fig = plt.figure(figsize=(10, 10))
    # ax = plt.axes(frameon=False)
    # plt.setp(ax, xticks=(), yticks=())
    # plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9, wspace=0.0, hspace=0.0)
    # plt.scatter(Y[:, 0], Y[:, 1], c=y_train, cmap=plt.cm.Spectral)
    # plt.savefig('tsne.png')

if __name__ == '__main__':
    main()
