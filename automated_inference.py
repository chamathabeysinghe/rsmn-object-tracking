import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras import Sequential, Model
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from model.RandomKernelModel import RandomKernelModel
from sklearn.linear_model import RidgeClassifierCV
from sklearn.decomposition import KernelPCA
import numpy as np
import os
from data import dnn_input
from data import classifier_input
from data import view_tracking_video
from sklearn.externals import joblib
from copy import copy, deepcopy
from scipy.optimize import linear_sum_assignment


model = RandomKernelModel.build_model((240, 320, 6),(10, 4), kernel_count=400)
model.load_weights('./checkpoints/saved_model_withoutpca_moredata')

# test_data = dnn_input.get_processed_frames(os.path.abspath('./data/videos/frames_test/'))
# output_test = model.predict(test_data, batch_size=16)


def inference_one_video(path):
    test_data = dnn_input.get_processed_frames(os.path.abspath(path))

    frames = dnn_input.get_frames(os.path.abspath(path), start=1, end=46)
    rois = dnn_input.get_processed_frames(os.path.abspath(path), relative=False)[1]

    output_test = model.predict(test_data, batch_size=16)

    '''
        Have created a new function get_processed_classifier_input_inference because using this we can easily index ROIs
         than the previous function (which I am thinking to remove). y_test_proper is a list for all the rois combinations 
         in all frames. (10*10*46). After reshaping this into (46,100) we can index position (i,j) as i - is the frame. 
         j / n_rois = current frames roi index
         j % n_rois = future frames roi index 
    '''
    X_test_proper, y_test_proper = classifier_input.get_processed_classifier_input_inference(output_test, normalized=True, balanced=False, do_shuffle=False)

    normalizer = joblib.load('./checkpoints/normalizer.joblib.pkl')
    X_test_proper_cp = normalizer.transform(X_test_proper)

    pca = joblib.load('./checkpoints/pca.joblib.pkl')
    X_pca_test_proper = pca.transform(X_test_proper_cp)


    clf = joblib.load('./checkpoints/lgbmclassifier.joblib.pkl')

    print(clf.score(X_pca_test_proper, y_test_proper))

    y_predict_proper = clf.predict(X_pca_test_proper)
    y_predict_proper_prob = clf.predict_proba(X_pca_test_proper)[:, 1]

    y_test_proper = np.asarray(y_test_proper).reshape((46, 100))
    y_test_proper_visualize = np.asarray(y_test_proper).reshape((46, 10, 10))

    y_predict_proper = np.asarray(y_predict_proper).reshape((46, 100))
    y_predict_proper_visualize = np.asarray(y_predict_proper).reshape((46, 10, 10))

    y_predict_proper_prob_visualize = np.asarray(y_predict_proper_prob).reshape((46, 10, 10))

    frame_roi_colors = np.zeros((rois.shape[0], rois.shape[1], 3))
    frame_roi_colors_2 = np.zeros((rois.shape[0], rois.shape[1], 3))
    frame_roi_colors_multiple = np.zeros((rois.shape[0], rois.shape[1]))
    frame_roi_colors_original = np.zeros((rois.shape[0], rois.shape[1], 3))

    colors = [
        (250, 0, 0),
        (0, 250, 0),
        (0, 0, 250),
        (250, 250, 0),
        (250, 0, 250),
        (0, 250, 250),
        (250, 250, 250),
        (100, 0, 50),
        (0, 50, 200),
        (70, 0, 200)
    ]

    frame_roi_colors[0, :, :] = np.asarray(colors)
    frame_roi_colors_2[0, :, :] = np.asarray(colors)

    for i in range(rois.shape[0]):
        frame_roi_colors_original[i, :, :] = np.asarray(colors)


    tracked_frames_original = view_tracking_video.add_tracks_with_colors(deepcopy(frames), rois, frame_roi_colors_original)
    # view_tracking_video.visualize_sequence(tracked_frames)

    # for frame_index in range(0, y_predict_proper.shape[0] - 1):
    #     for roi_match_index in range(y_predict_proper.shape[1]):
    #
    #         if y_predict_proper[frame_index, roi_match_index] == 1:
    #             current_frame_roi_index = int(roi_match_index / 10)
    #             future_frame_roi_index = roi_match_index % 10
    #             frame_roi_colors[frame_index+1, future_frame_roi_index] = frame_roi_colors[frame_index, current_frame_roi_index]
    #
    #
    # tracked_frames = view_tracking_video.add_tracks_with_colors(frames, rois, frame_roi_colors)
    # view_tracking_video.visualize_sequence(tracked_frames)

    """
    for frame_index in range(0, y_predict_proper.shape[0] - 1):
    for roi_match_index in range(y_predict_proper.shape[1]):

        if y_predict_proper[frame_index, roi_match_index] == 1:
            current_frame_roi_index = int(roi_match_index / 10)
            future_frame_roi_index = roi_match_index % 10
            frame_roi_colors_2[frame_index+1, future_frame_roi_index] = frame_roi_colors_original[frame_index, current_frame_roi_index]
            frame_roi_colors_multiple[frame_index+1, future_frame_roi_index] += 1

    tracked_frames = view_tracking_video.add_tracks_with_single_colors(deepcopy(frames), rois, frame_roi_colors_2, frame_roi_colors_multiple)
    view_tracking_video.visualize_two_sequences(tracked_frames_original, tracked_frames)
    # view_tracking_video.visualize_sequence(tracked_frames)
    """

    reshape_y_predict_proper_prob = y_predict_proper_prob.reshape((46, 10, 10))
    reshape_y_predict_proper_prob = -1 * reshape_y_predict_proper_prob
    frame_optimization = np.zeros((46, 10, 10))

    for frame_index in range(0, 45):
        pd.DataFrame(reshape_y_predict_proper_prob[frame_index]).to_csv(
            './data/view/probs/{}.csv'.format(frame_index + 1))
        row_ind, col_ind = linear_sum_assignment(reshape_y_predict_proper_prob[frame_index])
        for f_roi in range(10):
            p_roi = col_ind[f_roi]
            frame_optimization[frame_index, p_roi, f_roi] = 1
            frame_roi_colors_2[frame_index + 1, f_roi] = frame_roi_colors_original[frame_index, p_roi]

    print('Done')
inference_one_video('./data/test/4')
