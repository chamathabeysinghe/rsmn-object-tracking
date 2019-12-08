import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras import Sequential, Model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import os
from data import dnn_input
from model.RandomKernelModel import RandomKernelModel
from data import classifier_input
from sklearn.linear_model import RidgeClassifierCV
from sklearn.decomposition import KernelPCA
from sklearn.externals import joblib
import seaborn as sn
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score

train_data = dnn_input.get_processed_frames(os.path.abspath('./data/videos/frames_train/'))
test_data = dnn_input.get_processed_frames(os.path.abspath('./data/videos/frames_test/'))


def test_kernel_sizes(size):
    model = RandomKernelModel.build_model((240, 320, 6), (10, 4), kernel_count=size)
    output_train = model.predict(train_data, batch_size=16)
    output_test = model.predict(test_data, batch_size=16)

    X_train, y_train = classifier_input.get_processed_classifier_input(output_train, normalized=True)
    X_train_unbalanced, y_train_unbalanced = classifier_input.get_processed_classifier_input(output_train,
                                                                                             normalized=True,
                                                                                             balanced=False)
    X_test, y_test = classifier_input.get_processed_classifier_input(output_test, normalized=True)
    X_test_proper, y_test_proper = classifier_input.get_processed_classifier_input(output_test, normalized=True,
                                                                                   balanced=False)

    X_test_proper = np.asarray(X_test_proper)
    y_test_proper = np.asarray(y_test_proper)

    np.save('X_train', X_train)
    np.save('y_train', y_train)
    np.save('X_train_unbalanced', X_train_unbalanced)
    np.save('y_train_unbalanced', y_train_unbalanced)
    np.save('X_test', X_test)
    np.save('y_test', y_test)
    np.save('X_test_proper', X_test_proper)
    np.save('y_test_proper', y_test_proper)

    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    X_train_unbalanced = np.load('X_train_unbalanced.npy')
    y_train_unbalanced = np.load('y_train_unbalanced.npy')
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    X_test_proper = np.load('X_test_proper.npy')
    y_test_proper = np.load('y_test_proper.npy')

    min_max_scaler = MinMaxScaler(feature_range=(-.5, .5))
    normalizer = Normalizer()

    X_train_cp = min_max_scaler.fit_transform(X_train)
    X_test_cp = min_max_scaler.transform(X_test)
    X_test_proper_cp = min_max_scaler.transform(X_test_proper)
    X_train_unbalanced_cp = min_max_scaler.transform(X_train_unbalanced)

    X_train_cp = normalizer.fit_transform(X_train)
    X_test_cp = normalizer.transform(X_test)
    X_test_proper_cp = normalizer.transform(X_test_proper)
    X_train_unbalanced_cp = normalizer.transform(X_train_unbalanced)

    clf = LGBMClassifier(n_estimators=2000)
    clf.fit(X_train_cp, y_train)

    balanced_validation_score = clf.score(X_test_cp, y_test)
    proper_validation_score = clf.score(X_test_proper_cp, y_test_proper)
    y_predict_proper = clf.predict(X_test_proper_cp)

    data = confusion_matrix(y_test_proper, y_predict_proper)

    precision, recall, _, _ = precision_recall_fscore_support(y_test_proper, y_predict_proper, average='binary')
    f1 = f1_score(y_test_proper, y_predict_proper)

    print('{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(int(balanced_validation_score * 1000) / 1000,
                                                          int(proper_validation_score * 1000) / 1000,
                                                          int(precision * 1000) / 1000, int(recall * 1000) / 1000,
                                                          int(f1 * 1000) / 1000))


for size in [5, 10, 20, 40, 50, 75, 100, 200, 400, 800]:
    print(size)
    test_kernel_sizes(size)
