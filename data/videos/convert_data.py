import pandas as pd
filepath = 'test_coordinates.txt'
lines = []

frames_count = 0
rois_per_frame = 0

with open(filepath) as fp:
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
df.to_csv('frames_test/data.csv')

