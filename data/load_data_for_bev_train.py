import cv2

grid_dict : dict[str, list[tuple]]= {
    'trainClose cam1' : [],
    'trainClose cam2' : [],
    'trainClose cam3' : [],
    'trainClose cam4' : [],
    'trainClose cam5' : [],
    'trainMiddle cam1' : [],
    'trainMiddle cam2' : [],
    'trainMiddle cam3' : [],
    'trainMiddle cam4' : [],
    'trainMiddle cam5' : [],
    'trainLong cam1' : [],
    'trainLong cam2' : [],
    'trainLong cam3' : [],
    'trainLong cam4' : [],
    'trainLong cam5' : [],
    'background cam0' : [],
    'backgroundwhite cam0' : [],
    'background cam1' : [],
    'backgroundwhite cam1' : [],
    'background cam2' : [],
    'backgroundwhite cam2' : [],
    'background cam3' : [],
    'backgroundwhite cam3' : [],
    'background cam4' : [],
    'backgroundwhite cam4' : [],
}

def load_data_bev(fileinfo : str):
    '''
    fileinfo : line of data/filepaths_bev.txt
    '''
    fileinfo = fileinfo.split(' ')
    grid_type = ' '.join(fileinfo[:2])
    img_path = fileinfo[-1]
    
    img = cv2.imread(img_path)

    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # (H, W, C) -> (C, H, W)
    img = img.transpose(2, 0, 1)
    
    return {
        'image' : img,
        'grid_points' : grid_dict[grid_type]
    }