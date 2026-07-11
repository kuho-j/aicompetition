import os
result = []

with open('filepaths_bev.txt') as f:
    for line in f:
        imgpath = line.split(' ')[-1]
        
        if os.path.isfile(imgpath):
            result.append(line)

with open('filepaths_bev.txt') as f:
    for line in result:
        f.write(line)
