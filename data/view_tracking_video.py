import numpy as np
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv


def get_frames(path):
    frames = []
    rois_in_frame = []
    for i in range(20):
        img = np.array(Image.open(os.path.join(path, 'frame_{}.png'.format(i))))
        frames.append(img)

    df_data = pd.read_csv(os.path.join(path, 'data.csv')).values

    def get_rois_for_frame(row):
        roi_row = []
        for j in range(1, len(row)):
            roi = eval(row[j])
            # roi[0] /= image_height
            # roi[1] /= image_width
            # roi[2] /= image_height
            # roi[3] /= image_width
            roi_row.append(roi)
        return roi_row

    for i in range(len(frames)):
        rois_in_frame.append(get_rois_for_frame(df_data[i]))

    return frames, rois_in_frame


def visualize_sequence(video):
    for frame in video:
        # cv.imshow('frame', frame)
        # cv.waitKey(100)
        plt.imshow(frame)
        plt.pause(.5)


def add_tracks(video, rois_in_frame):
    tracked_frames = []
    for i in range(len(video)):
        print(i)
        frame = video[i]
        rois = rois_in_frame[i]
        print(type(frame))
        print(frame.shape)
        for roi in rois:
            print(roi)
            x0, y0, x1, y1 = roi
            frame[x0, y0:y1, :] = 0
            frame[x1, y0:y1, :] = 0
            frame[x0:x1, y0, :] = 0
            frame[x0:x1, y1, :] = 0
        tracked_frames.append(frame)
    return tracked_frames


frames, rois_in_frame = get_frames('./test')
tracked_frames = add_tracks(frames, rois_in_frame)
visualize_sequence(frames)
