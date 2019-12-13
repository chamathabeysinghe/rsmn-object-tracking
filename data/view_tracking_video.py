import numpy as np
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
import matplotlib


def get_frames(path):
    frames = []
    rois_in_frame = []
    for i in range(49):
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
    plt.rcParams["figure.figsize"] = (30, 10)

    for frame in video:
        # cv.imshow('frame', frame)
        # cv.waitKey(100)
        plt.imshow(frame)
        plt.pause(.1)


def write_to_files(frames, parent_path):
    for index, frame in enumerate(frames):
        path = os.path.join(parent_path, 'frame_{}.png'.format(index))
        image = Image.fromarray(frame)
        image.save(path)


def visualize_two_sequences(original, predicted):
    plt.rcParams["figure.figsize"] = (60, 20)

    for i in range(len(original)):
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(original[i])
        axarr[1].imshow(predicted[i])
        plt.pause(.1)



def add_tracks(video, rois_in_frame):
    tracked_frames = []
    for i in range(len(video)):
        frame = video[i]
        rois = rois_in_frame[i]
        for roi in rois:
            x0, y0, x1, y1 = roi
            frame[x0, y0:y1, :] = 0
            frame[x1, y0:y1, :] = 0
            frame[x0:x1, y0, :] = 0
            frame[x0:x1, y1, :] = 0
        tracked_frames.append(frame)
    return tracked_frames


def add_tracks_with_colors(video, rois_in_frame, colors):
    tracked_frames = []
    for i in range(len(video)):
        frame = video[i]
        rois = rois_in_frame[i]
        for j in range(len(rois)):
            roi = rois[j]
            color = colors[i, j]
            x0, y0, x1, y1 = roi
            frame[x0, y0:y1, :] = color
            frame[x1, y0:y1, :] = color
            frame[x0:x1, y0, :] = color
            frame[x0:x1, y1, :] = color
        tracked_frames.append(frame)
    return tracked_frames


def add_tracks_with_single_colors(video, rois_in_frame, colors, frame_roi_colors_multiple):
    tracked_frames = []
    for i in range(len(video)):
        frame = video[i]
        rois = rois_in_frame[i]
        for j in range(len(rois)):
            if i > 0 and frame_roi_colors_multiple[i, j] != 1:
                continue

            roi = rois[j]
            color = colors[i, j]
            x0, y0, x1, y1 = roi
            frame[x0, y0:y1, :] = color
            frame[x1, y0:y1, :] = color
            frame[x0:x1, y0, :] = color
            frame[x0:x1, y1, :] = color
        tracked_frames.append(frame)
    return tracked_frames

# frames, rois_in_frame = get_frames('./videos/frames_train')
# tracked_frames = add_tracks(frames, rois_in_frame)
# visualize_sequence(frames)
