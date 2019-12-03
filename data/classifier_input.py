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


def get_processed_classifier_input(dnn_output, normalized=False, balanced=True, do_shuffle=True):

    if not normalized:
        embedding_for_current = dnn_output[0]
        embedding_for_future = dnn_output[1]
        n_samples = embedding_for_current.shape[0]
    else:
        embedding_for_current, embedding_for_future = normalize_output(dnn_output)
        n_samples = embedding_for_current.shape[0]

    dataset = []
    for i in range(n_samples):
        features_current = embedding_for_current[i]
        features_future = embedding_for_future[i]
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


def get_processed_classifier_input_inference(dnn_output, normalized=False, balanced=True, do_shuffle=True):

    if not normalized:
        embedding_for_current = dnn_output[0]
        embedding_for_future = dnn_output[1]
        n_samples = embedding_for_current.shape[0]
        n_rois = embedding_for_current.shape[1]
    else:
        embedding_for_current, embedding_for_future = normalize_output(dnn_output)
        n_samples = embedding_for_current.shape[0]
        n_rois = embedding_for_current.shape[1]

    dataset = []
    for i in range(n_samples):
        features_current = embedding_for_current[i]
        features_future = embedding_for_future[i]

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