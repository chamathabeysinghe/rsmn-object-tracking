import numpy as np


def get_same_records(features_current, features_future):
    records = []
    for i in range(len(features_current)):
        records.append((np.concatenate([features_current[i], features_future[i]], axis=2), 1))
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
    embedding_for_current = dnn_output[0] / 160
    embedding_for_future = dnn_output[1] / 160
    return embedding_for_current, embedding_for_future


def get_processed_classifier_input(dnn_output, normalized=False):

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
        records_same = get_same_records(features_current, features_future)
        records_diff = get_different_records(features_current, features_future)
        dataset += records_same
        dataset += records_diff

    X = []
    y = []
    for item in dataset:
        X.append(item[0].flatten())
        y.append(item[1])

    return X, y

