"""
Convert VitBat CSV file as we need.
"""

import pandas as pd


def convert_annotations(file, frames_per_video=50, video_sets=6):
    lines = []

    frames_count = 0
    rois_per_frame = 0

    with open(file) as fp:
        for cnt, line in enumerate(fp):
            if cnt < 3:
                continue
            line = [int(round(float(x))) for x in line.strip('\n').split('\t')]
            line[2] /= 6
            line[3] /= 6
            line[4] /= 6
            line[5] /= 6
            line = [int(round(x)) for x in line]
            rois_per_frame = max(rois_per_frame, line[1])
            frames_count = line[0]
            lines.append(line)

    data = [[[[-1, -1, -1, -1] for _ in range(rois_per_frame)] for _ in range(frames_per_video)] for _ in range(video_sets)]

    for line in lines:
        frame_index = (line[0] - 1) % frames_per_video
        video_index = (line[0] - 1) // frames_per_video
        roi_index = line[1] - 1

        if video_index >= video_sets:
            break

        square = [line[3], line[2], line[3] + line[5], line[2] + line[4]]
        data[video_index][frame_index][roi_index] = square

    for i in range(video_sets):
        df = pd.DataFrame(data[i])
        df.to_csv('frames/{}/data.csv'.format(i))


convert_annotations('./annotations/annotations_1.txt')

