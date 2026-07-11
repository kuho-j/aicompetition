from make_filename import make_filename

with open('data/filename.txt', 'r') as f:
    for line in f:
        paths = make_filename(line)

        add_txt = ''
        for idx, img_path in enumerate(paths[0]):
            add_txt = add_txt + f'train cam{idx+1} {img_path}\n'
        
        with open('data/filename_bev_gird.txt', 'a') as file:
            file.write(add_txt)