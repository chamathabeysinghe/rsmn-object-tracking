"""
Extract frames from videos and save as images
"""

import cv2
import os


def convert_file(file, frames_per_video=50):
    vidcap = cv2.VideoCapture(file)
    success, image = vidcap.read()
    count = 0
    while success:
        image = cv2.resize(image, (320, 180))
        dir = './frames/{}'.format(count // frames_per_video)
        file = 'frame_{}.png'.format(count % frames_per_video)
        os.makedirs(dir, exist_ok=True)
        cv2.imwrite(os.path.join(dir, file), image)
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1
        if count >= 300:
            break

convert_file('./annotations/video1.MOV')

