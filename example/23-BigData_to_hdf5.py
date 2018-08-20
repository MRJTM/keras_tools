# USAGE
# python 23-BigData_to_hdf5.py

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from preprocessing import AspectAwarePreprocessor
from tools import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import json
import cv2
import os

"""--------------------------------相关配置--------------------------------------------"""
# define the paths to the images directory
IMAGES_PATH = "../../datasets/kaggle_dogs_vs_cats/train"

# since we do not have validation data or access to the testing
# labels we need to take a number of images from the training
# data and use them instead
NUM_CLASSES = 2
NUM_VAL_IMAGES = 1250 * NUM_CLASSES
NUM_TEST_IMAGES = 1250 * NUM_CLASSES

# define the path to the output training, validation, and testing
# HDF5 files
TRAIN_HDF5 = "../../datasets/kaggle_dogs_vs_cats/hdf5/train.hdf5"
VAL_HDF5 = "../../datasets/kaggle_dogs_vs_cats/hdf5/val.hdf5"
TEST_HDF5 = "../../datasets/kaggle_dogs_vs_cats/hdf5/test.hdf5"


# define the path to the dataset mean
DATASET_MEAN = "../saved_models/dogs_vs_cats_mean.json"

# define the path to the output directory used for storing plots,
# classification reports, etc.
OUTPUT_PATH = "../output"

"""-----------------------------读入图片路径，划分数据集---------------------------------------"""
# grab the paths to the images
trainPaths = list(paths.list_images(IMAGES_PATH))
trainLabels = [p.split(os.path.sep)[-1].split(".")[0]
	for p in trainPaths]
le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)

# perform stratified sampling from the training set to build the
# testing split from the training data
split = train_test_split(trainPaths, trainLabels,
	test_size=NUM_TEST_IMAGES, stratify=trainLabels,
	random_state=42)
(trainPaths, testPaths, trainLabels, testLabels) = split

# perform another stratified sampling, this time to build the
# validation data
split = train_test_split(trainPaths, trainLabels,
	test_size=NUM_VAL_IMAGES, stratify=trainLabels,
	random_state=42)
(trainPaths, valPaths, trainLabels, valLabels) = split

# construct a list pairing the training, validation, and testing
# image paths along with their corresponding labels and output HDF5
# files
datasets = [
	("train", trainPaths, trainLabels, TRAIN_HDF5),
	("val", valPaths, valLabels, VAL_HDF5),
	("test", testPaths, testLabels, TEST_HDF5)]


"""----------------------------------开始写入HDF5文件------------------------------------------"""
# initialize the image pre-processor and the lists of RGB channel
# averages
aap = AspectAwarePreprocessor(256, 256)
(R, G, B) = ([], [], [])

# loop over the dataset tuples
for (dType, paths, labels, outputPath) in datasets:
	# create HDF5 writer
	print("[INFO] building {}...".format(outputPath))
	writer = HDF5DatasetWriter((len(paths), 256, 256, 3), outputPath)

	# initialize the progress bar
	widgets = ["Building Dataset: ", progressbar.Percentage(), " ",
		progressbar.Bar(), " ", progressbar.ETA()]
	pbar = progressbar.ProgressBar(maxval=len(paths),
		widgets=widgets).start()

	# loop over the image paths
	for (i, (path, label)) in enumerate(zip(paths, labels)):
		# load the image and process it
		image = cv2.imread(path)
		image = aap.preprocess(image)

		# if we are building the training dataset, then compute the
		# mean of each channel in the image, then update the
		# respective lists
		if dType == "train":
			(b, g, r) = cv2.mean(image)[:3]
			R.append(r)
			G.append(g)
			B.append(b)

		# add the image and label # to the HDF5 dataset
		writer.add([image], [label])
		pbar.update(i)

	# close the HDF5 writer
	pbar.finish()
	writer.close()

# construct a dictionary of averages, then serialize the means to a
# JSON file
print("[INFO] serializing means...")
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
f = open(DATASET_MEAN, "w")
f.write(json.dumps(D))
f.close()