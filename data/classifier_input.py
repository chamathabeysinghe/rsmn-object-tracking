import numpy as np
from random import shuffle


def get_same_records(features_current, features_future, balanced):
    records = []
    for i in range(len(features_current)):
        for _ in range(9):  # TODO Change 9 to dyncamic value
            records.append((np.concatenate([features_current[i], features_future[i]], axis=2), 1))
            if not balanced:
                break
    return records


def get_different_records(features_current, features_future):
    records = []
    for i in range(len(features_current)):
        for j in range(len(features_future)):
            if i == j:
                continue
            records.append((np.concatenate([features_current[i], features_future[j]], axis=2), 0))
    return records


def normalize_output(dnn_output):
    embedding_for_current = dnn_output[0] / 100 - .5
    embedding_for_future = dnn_output[1] / 100 - .5
    return embedding_for_current, embedding_for_future


def get_processed_classifier_input(dnn_output, position_data=None, normalized=False, balanced=True, do_shuffle=True):

    if not normalized:
        embedding_for_current = dnn_output[0]
        embedding_for_future = dnn_output[1]
        n_samples = embedding_for_current.shape[0]
    else:
        embedding_for_current, embedding_for_future = normalize_output(dnn_output)
        n_samples = embedding_for_current.shape[0]

    if position_data is not None:
        rois_for_current = position_data[0]
        rois_for_future = position_data[1]

    dataset = []
    for i in range(n_samples):
        features_current = embedding_for_current[i]
        features_future = embedding_for_future[i]

        if position_data is not None:
            roi_current = rois_for_current[i].reshape((10, 1, 1, 4))
            rois_future = rois_for_future[i].reshape((10, 1, 1, 4))

            # TODO change bellow 4 lines, this is to test how using
            features_current = np.concatenate((features_current, roi_current), axis=3)
            features_future = np.concatenate((features_future, rois_future), axis=3)
            #
            # features_current = roi_current
            # features_future = rois_future

        records_same = get_same_records(features_current, features_future, balanced)
        records_diff = get_different_records(features_current, features_future)
        dataset += records_same
        dataset += records_diff

    if do_shuffle:
        shuffle(dataset)

    X = []
    y = []
    for item in dataset:
        X.append(item[0].flatten())
        y.append(item[1])

    return X, y


def get_processed_classifier_input_for_multiple_videos(dnn_outputs, positions_data=None, normalized=False, balanced=True, do_shuffle=True):
    X = []
    y = []
    for i in range(len(dnn_outputs)):
        dnn_output = dnn_outputs[i]

        position_data = None
        if positions_data is not None:
            position_data = positions_data[i]

        X_video, y_video = get_processed_classifier_input(dnn_output, normalized=normalized, balanced=balanced,
                                                          do_shuffle=do_shuffle, position_data=position_data)
        X += X_video
        y += y_video

    return X, y


def get_processed_classifier_input_inference(dnn_output, position_data=None, normalized=False, balanced=True, do_shuffle=True):

    if not normalized:
        embedding_for_current = dnn_output[0]
        embedding_for_future = dnn_output[1]
        n_samples = embedding_for_current.shape[0]
        n_rois = embedding_for_current.shape[1]
    else:
        embedding_for_current, embedding_for_future = normalize_output(dnn_output)
        n_samples = embedding_for_current.shape[0]
        n_rois = embedding_for_current.shape[1]

    if position_data is not None:
        rois_for_current = position_data[0]
        rois_for_future = position_data[1]

    dataset = []
    for i in range(n_samples):
        features_current = embedding_for_current[i]
        features_future = embedding_for_future[i]

        if position_data is not None:
            roi_current = rois_for_current[i].reshape((10, 1, 1, 4))
            rois_future = rois_for_future[i].reshape((10, 1, 1, 4))

            features_current = np.concatenate((features_current, roi_current), axis=3)
            features_future = np.concatenate((features_future, rois_future), axis=3)

        for x in range(n_rois):
            for y in range(n_rois):
                if x == y:
                    dataset.append((np.concatenate([features_current[x], features_future[y]], axis=2), 1))
                else:
                    dataset.append((np.concatenate([features_current[x], features_future[y]], axis=2), 0))

    if do_shuffle:
        shuffle(dataset)

    X = []
    y = []
    for item in dataset:
        X.append(item[0].flatten())
        y.append(item[1])

    return X, y


# train_roi_data = np.load('../weights/multiple_train_video_rois.npy')
# output_train_dnn = np.load('../weights/multiple_train_video_dnn_output.npy')
# #
# # get_processed_classifier_input_for_multiple_videos(output_train_dnn, positions_data=train_roi_data, normalized=True)
#
# get_processed_classifier_input_inference(output_train_dnn[0], normalized=True, balanced=False, do_shuffle=False)
