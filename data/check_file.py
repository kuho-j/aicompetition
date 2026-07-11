import os
with open('filepaths_bev.txt') as f:
    for num_line, line in enumerate(f):
        imgpath = line.split(' ')[-1]
        
        if not os.path.isfile(imgpath):
            print(f'line {num_line + 1} path {imgpath} do not exists')