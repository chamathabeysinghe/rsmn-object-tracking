import numpy as np
import os
from data import dnn_input
from model.RandomKernelModel import RandomKernelModel
from data import classifier_input
from sklearn.linear_model import RidgeClassifierCV
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import Normalizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


def get_dnn_output(train_data, test_data, kernel_count=400):
    model = RandomKernelModel.build_model((240, 320, 6), (10, 4), kernel_count=kernel_count)

    output_train = []
    for video_data in train_data:
        output_train.append(model.predict(video_data, batch_size=16))

    output_test = []
    for video_data in test_data:
        output_test.append(model.predict(video_data, batch_size=16))

    return output_train, output_test


def get_classifier_input(output_train, output_test, normalize=True, pca=True, pca_components=200, train_balanced=True, test_balanced=False, train_positions_data=None, test_positions_data=None):
    X_train_balanced, y_train_balanced = classifier_input.get_processed_classifier_input_for_multiple_videos(output_train, positions_data=train_positions_data, normalized=True)
    X_train_unbalanced, y_train_unbalanced = classifier_input.get_processed_classifier_input_for_multiple_videos(output_train, positions_data=train_positions_data, normalized=True, balanced=False)
    X_test_balanced, y_test_balanced = classifier_input.get_processed_classifier_input_for_multiple_videos(output_test, positions_data=test_positions_data, normalized=True)
    X_test_unbalanced, y_test_unbalanced = classifier_input.get_processed_classifier_input_for_multiple_videos(output_test, positions_data=test_positions_data, normalized=True, balanced=False)

    if train_balanced:
        X_train = X_train_balanced
        y_train = y_train_balanced
    else:
        X_train = X_train_unbalanced
        y_train = y_train_unbalanced

    if test_balanced:
        X_test = X_test_balanced
        y_test = y_test_balanced
    else:
        X_test = X_test_unbalanced
        y_test = y_test_unbalanced

    if normalize:
        normalizer = Normalizer()
        X_train = normalizer.fit_transform(X_train)
        X_test = normalizer.transform(X_test)

    if pca:
        pca = PCA(n_components=pca_components, whiten=False, random_state=2019)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    return X_train, y_train, X_test, y_test


def get_classifier(clf_name):
    svmClf = SVC(gamma='auto')
    advancedSvmClf = SVC(gamma='auto', kernel='rbf', probability=True, class_weight='balanced', C=120000000)
    ridgeClf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
    randomForestClf = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=0)
    gradientBoostClf = GradientBoostingClassifier(n_estimators=200, max_depth=7, random_state=0)
    xgbClf = XGBClassifier(n_estimators=500, max_depth=5)
    adaBoostClf = AdaBoostClassifier()
    lgbmClf = LGBMClassifier(n_estimators=1000)

    if clf_name == 'svm':
        return svmClf
    elif clf_name == 'advanced_svm':
        return advancedSvmClf
    elif clf_name == 'ridge':
        return ridgeClf
    elif clf_name == 'random_forest':
        return randomForestClf
    elif clf_name == 'gradient_boosting':
        return gradientBoostClf
    elif clf_name == 'xgb':
        return xgbClf
    elif clf_name == 'ada_boost':
        return adaBoostClf
    elif clf_name == 'lgbm':
        return lgbmClf


def evaluate_classifier(clf, X, y):
    score = clf.score(X, y)
    y_predict = clf.predict(X)

    precision, recall, _, _ = precision_recall_fscore_support(y, y_predict, average='binary')
    f1 = f1_score(y, y_predict)

    data = confusion_matrix(y, y_predict)

    return [score, int(precision * 1000) / 1000, int(recall * 1000) / 1000, int(f1 * 1000) / 1000]


def run_tests():
    train_data = dnn_input.get_processed_frames_for_multiple_videos(os.path.abspath('./data/train/'))
    test_data = dnn_input.get_processed_frames_for_multiple_videos(os.path.abspath('./data/test/'))

    train_roi_data = dnn_input.get_rois_for_multiple_videos(os.path.abspath('./data/train/'))
    test_roi_data = dnn_input.get_rois_for_multiple_videos(os.path.abspath('./data/test/'))

    kernel_sizes = [50, 75, 100, 200, 400, 800]
    # kernel_sizes = [100, 200, 400, 800]

    for size in kernel_sizes:
        output_train, output_test = get_dnn_output(train_data=train_data, test_data=test_data, kernel_count=size)
        X_train, y_train, X_test, y_test = get_classifier_input(output_train, output_test, normalize=True, pca=True, pca_components=4, train_positions_data=train_roi_data, test_positions_data=test_roi_data)
        clf = get_classifier('lgbm')
        clf.fit(X_train, y_train)
        result = evaluate_classifier(clf, X_test, y_test)
        print('{:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(*result))

    for size in kernel_sizes:
        print('{} Kernels + Normalizer + PCA (8 Components) + LGBM Classifier (n_estimators=20) + Position Data'.format(size))


run_tests()
