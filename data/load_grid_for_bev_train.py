
grid_dict : dict[str, list[tuple]]= {
    'trainClose cam1' : [],
    'trainClose cam2' : [],
    'trainClose cam3' : [],
    'trainClose cam4' : [],
    'trainClose cam5' : [],
    'trainMiddle cam1' : [],
    'traiMiddle cam2' : [],
    'traiMiddle cam3' : [],
    'traiMiddle cam4' : [],
    'traiMiddle cam5' : [],
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


def load_grid(info : str):
    '''
    info : 
    
    e.g.
    train cam1
    -> it means the photo is from the train dataset in the view of cam1
    '''
    return grid_dict[info]
