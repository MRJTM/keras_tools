# USAGE
# python 20-rank_accuracy.py --db ../datasets/flowers17/hdf5/features.hdf5
#	--model ../chapter03-feature_extraction/flowers17.cpickle
# python 20-rank_accuracy.py --db ../datasets/caltech101/hdf5/features.hdf5 \
#	--model ../chapter03-feature_extraction/caltech101.cpickle

# import the necessary packages
from tools.ranked import rank5_accuracy
import argparse
import pickle
import h5py

# construct the argument parse and parse the arguments
db_name="animals"
model_path='../saved_models/{}.cpickle'.format(db_name)
db_path='../../datasets/{}/hdf5/features.hdf5'.format(db_name)

# load the pre-trained model
print("[INFO] loading pre-trained model...")
model = pickle.loads(open(model_path, "rb").read())

# 导入h5格式的数据集
db = h5py.File(db_path, "r")
i = int(db["labels"].shape[0] * 0)

# 在测试集上预测，获得rank-1和rank-5的
print("[INFO] predicting...")
preds = model.predict_proba(db["features"][i:])
(rank1, rank5) = rank5_accuracy(preds, db["labels"][i:])

# display the rank-1 and rank-5 accuracies
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
print("[INFO] rank-5: {:.2f}%".format(rank5 * 100))

# close the database
db.close()