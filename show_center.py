from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
    filename = ''
    fileinfo = filename.split('_')
    img_filename = '../Dataset/1.competition_trainset/1_dataset/'+fileinfo[0]+'_CAPP_cam1_'+fileinfo[1]+'_'+fileinfo[2]+'.jpg'
    image = Image.open(img_filename)
    img_copy = np.array(image.convert('RGC')).copy()
    
    with open('data/'+filename+'.txt') as file:
        for line in file:
            cx, cy = map(float, line.split()[1:])
            cv2.circle(img_copy, (cx, cy), 5, (255, 0, 0), -1)

    plt.imshow(img_copy)
    plt.show()

if __name__ == '__main__':
    main()