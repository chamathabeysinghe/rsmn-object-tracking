import numpy as np
from PIL import Image
import pandas as pd
import os


def get_processed_frames(path):
    frames = []
    for i in range(20):
        img = np.asarray(Image.open(os.path.join(path, 'frame_{}.png'.format(i))))
        frames.append(img)
    image_size = frames[0].shape
    image_height = image_size[0]
    image_width = image_size[1]
    df_data = pd.read_csv(os.path.join(path, 'data.csv')).values

    current_frames = []
    current_rois = []
    future_frames = []
    future_rois = []

    def get_rois_for_frame(row):
        roi_row = []
        for j in range(1, len(row)):
            roi = eval(row[j])
            roi[0] /= image_height
            roi[1] /= image_width
            roi[2] /= image_height
            roi[3] /= image_width
            roi_row.append(roi)
        return roi_row

    for i in range(1, len(frames) - 2):
        current_frames.append(np.concatenate([frames[i], frames[i - 1]], axis=2))
        current_rois.append(get_rois_for_frame(df_data[i]))

        future_frames.append(np.concatenate([frames[i + 2], frames[i + 1]], axis=2))
        future_rois.append(get_rois_for_frame(df_data[i + 1]))

    current_frames = np.asarray(current_frames)
    current_rois = np.asarray(current_rois)
    future_frames = np.asarray(future_frames)
    future_rois = np.asarray(future_rois)

    return current_frames, current_rois, future_frames, future_rois

