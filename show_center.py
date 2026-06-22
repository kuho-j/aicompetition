from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
    filename = 'Close_10_13924'
    fileinfo = filename.split('_')
    img_filename = '../Dataset/1.competition_trainset/1_dataset/'+fileinfo[0]+'_CAPP_cam1_'+fileinfo[1]+'_'+fileinfo[2]+'.jpg'
    image = Image.open(img_filename)
    img_copy = np.array(image.convert('RGB')).copy()

    # test code...
    for i in range(1, 6):
        print(f'cam{i}:\n')
        with open('../Dataset/1.competition_trainset/1_dataset/'+fileinfo[0]+'_CAPP_cam'+str(i)+'_'+fileinfo[1]+'_'+fileinfo[2]+'.txt', 'r') as file:
            print(file.read())
    
    with open('data/2dlabel_dataset/'+filename+'.txt', 'r') as file:
        for line in file:
            cx, cy = map(float, line.split()[1:])
            cx = int(cx * 640)
            cy = int(cy * 480)

            cx = max(cx, 0)
            cx = min(640, cx)

            cy = max(cy, 0)
            cy = min(480, cy)

            cv2.circle(img_copy, (cx, cy), 5, (255, 0, 0), 1)

    plt.imshow(img_copy)
    plt.show()

if __name__ == '__main__':
    main()
