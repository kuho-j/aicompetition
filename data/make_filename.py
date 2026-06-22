
path = '/home/aicompetition43/Dataset/1.competition_trainset/1_dataset/'
path_1 = '/home/aicompetition43/prototype_1/data/2dlabel_dataset/'
cam = ['cam1', 'cam2', 'cam3', 'cam4', 'cam5']
f_type = ['.jpg', '.txt']

def make_filename(in_str : str):
    '''
    return:
    [[cam1.jpg, cam2.jpg, cam3.jpg, cam4.jpg, cam5.jpg],
     [cam1.txt, cam2.txt, cam3.txt, cam4.txt, cam5.txt],
     ().txt]
    '''

    info = in_str.split() # {close/middle/long} {num1} {num2}

    result =  [[path + ('_'.join([info[0], 'CAPP', cam[j], info[1], info[2]])) + f_type[i] for j in range(5)] for i in range(2)]
    result.append(path_1 + '_'.join(info) + '.txt')

    return result


if __name__ == '__main__':
    print(make_filename('1 2 3'))
    



