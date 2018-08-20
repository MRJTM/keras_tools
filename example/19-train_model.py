# USAGE
# python 4-train_model.py --db ../datasets/animals/hdf5/features.hdf5 \
#	--model animals.cpickle
# python 4-train_model.py --db ../datasets/caltech-101/hdf5/features.hdf5 \
#	--model caltech101.cpickle
# python 4-train_model.py --db ../datasets/flowers17/hdf5/features.hdf5 \
#	--model flowers17.cpickle

# import the necessary packages
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import argparse
import pickle
import h5py

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--db", required=True,
# 	help="path HDF5 database")
# ap.add_argument("-m", "--model", required=True,
# 	help="path to output model")
# ap.add_argument("-j", "--jobs", type=int, default=-1,
# 	help="# of jobs to run when tuning hyperparameters")
# args = vars(ap.parse_args())

"""用CPU多线程去training一个sklearn提供的简单分类器"""
if __name__=='__main__':

    db_name='caltech-101'
    if db_name=="animals":
        db_path = '../../datasets/animals/hdf5/features.hdf5'
    elif db_name=="caltech-101":
        db_path = '../../datasets/caltech-101/hdf5/features.hdf5'
    else:
        db_path ='../../datasets/flowers17/hdf5/features.hdf5'

    model_name = '../saved_models/{}.cpickle'.format(db_name)
    jobs = 8

    # 打开hdf5的特征数据文件，确定75%为训练集
    db = h5py.File(db_path, "r")
    i = int(db["labels"].shape[0] * 0.75)

    # define the set of parameters that we want to tune then start a
    # grid search where we evaluate our model for each value of C
    print("[INFO] tuning hyperparameters...")
    params = {"C": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}
    model = GridSearchCV(LogisticRegression(), params, cv=3,
        n_jobs=jobs)
    model.fit(db["features"][:i], db["labels"][:i])
    print("[INFO] best hyperparameters: {}".format(model.best_params_))

    # evaluate the model
    print("[INFO] evaluating...")
    preds = model.predict(db["features"][i:])
    print(classification_report(db["labels"][i:], preds,
        target_names=db["label_names"]))

    # serialize the model to disk
    print("[INFO] saving model...")
    f = open(model_name, "wb")
    f.write(pickle.dumps(model.best_estimator_))
    f.close()

    # close the database
    db.close()