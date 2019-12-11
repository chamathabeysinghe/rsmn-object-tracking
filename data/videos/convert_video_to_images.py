"""
Extract frames from videos and save as images
"""

import cv2
import os
import glob


def convert_video_to_frames(read_path, save_path):
    vidcap = cv2.VideoCapture(read_path)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite("{}/frame_{}.png".format(save_path, count), image)     # save frame as JPEG file
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1


files = sorted(glob.glob('./annotations/out_*.mp4'))
indexes = [int(x[x.rfind('_')+1:x.rfind('.')]) for x in files]

for x in zip(indexes, files):
    save_path = './frames/{}'.format(x[0])
    os.makedirs(save_path, exist_ok=True)
    read_path = x[1]
    convert_video_to_frames(read_path, save_path)
