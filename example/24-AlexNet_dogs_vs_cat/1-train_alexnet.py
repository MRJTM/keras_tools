# USAGE
# python 1-train_alexnet.py

# import the necessary packages
# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")

# import the necessary packages
from preprocessing import ImageToArrayPreprocessor
from preprocessing import SimplePreprocessor
from preprocessing import PatchPreprocessor
from preprocessing import MeanPreprocessor
from tools import TrainingMonitor
from tools import HDF5DatasetGenerator
from net.alexnet import AlexNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import json
import os

"""--------------------------------相关配置--------------------------------------------"""
# 配置项目顶层路径
project_root = "../../"

# define the path to the output training, validation, and testing
# HDF5 files
TRAIN_HDF5 = project_root + "../datasets/kaggle_dogs_vs_cats/hdf5/train.hdf5"
VAL_HDF5 = project_root + "../datasets/kaggle_dogs_vs_cats/hdf5/val.hdf5"
TEST_HDF5 = project_root + "../datasets/kaggle_dogs_vs_cats/hdf5/test.hdf5"

# path to the output model file
MODEL_PATH = project_root + "saved_models//alexnet_dogs_vs_cats.model"

# define the path to the dataset mean
DATASET_MEAN = project_root + "saved_models/dogs_vs_cats_mean.json"

# define the path to the output directory
OUTPUT_PATH = project_root + "output"

"""----------------------------------初始化-------------------------------------------"""
# 构建data genetor
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
                         width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                         horizontal_flip=True, fill_mode="nearest")

# 载入均值文件
means = json.loads(open(DATASET_MEAN).read())

# initialize the image preprocessors
sp = SimplePreprocessor(227, 227)  # 图像resize到(227,227)
pp = PatchPreprocessor(227, 227)  # 随机切去一个(227,227)的patch
mp = MeanPreprocessor(means["R"], means["G"], means["B"])  # 去掉均值
iap = ImageToArrayPreprocessor()

# initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(TRAIN_HDF5, 64, aug=aug,
                                preprocessors=[pp, mp, iap], classes=2)
valGen = HDF5DatasetGenerator(VAL_HDF5, 64,
                              preprocessors=[sp, mp, iap], classes=2)

# 模型编译
print("[INFO] compiling model...")
opt = Adam(lr=1e-3)
model = AlexNet.build(width=227, height=227, depth=3,
                      classes=2, reg=0.0002)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# construct the set of callbacks
path = os.path.sep.join([OUTPUT_PATH, "{}.png".format(
    os.getpid())])
callbacks = [TrainingMonitor(path)]

"""-------------------------------训练和保存模型--------------------------------------"""
model.fit_generator(
    trainGen.generator(),
    steps_per_epoch=trainGen.numImages // 128,
    validation_data=valGen.generator(),
    validation_steps=valGen.numImages // 128,
    epochs=75,
    max_queue_size=128 * 2,
    callbacks=callbacks, verbose=1)

# save the model to file
print("[INFO] serializing model...")
model.save(MODEL_PATH, overwrite=True)

# close the HDF5 datasets
trainGen.close()
valGen.close()
