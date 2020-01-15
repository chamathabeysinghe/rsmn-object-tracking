"""
Convert VitBat CSV file as we need.
"""

import pandas as pd
import glob
import os


def convert_csv(read_path, save_path):
    lines = []

    frames_count = 0
    rois_per_frame = 0

    with open(read_path) as fp:
        for cnt, line in enumerate(fp):
            if cnt < 3:
                continue
            line = [int(round(float(x))) for x in line.strip('\n').split('\t')]

            rois_per_frame = max(rois_per_frame, line[1])
            frames_count = line[0]
            lines.append(line)

    data_array = [[[-1, -1, -1, -1] for _ in range(rois_per_frame)] for _ in range(frames_count)]

    for line in lines:
        frame_index = line[0]
        roi_index = line[1]
        square = [line[3], line[2], line[3] + line[5], line[2] + line[4]]
        data_array[frame_index - 1][roi_index - 1] = square

    df = pd.DataFrame(data_array)
    print(df.head())
    df.to_csv(save_path)


files = sorted(glob.glob('./annotations/out_*_IndividualStates.txt'))
indexes = [int(x[x.index('_')+1:x.rfind('_')]) for x in files]

for x in zip(indexes, files):
    print(x)
    save_path = './frames/{}/data.csv'.format(x[0])
    # os.makedirs(save_path, exist_ok=True)
    read_path = x[1]
    convert_csv(read_path, save_path)


