from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
    image = Image.open('../Dataset/1.competition_trainset/1_dataset/Close_CAPP_cam5_10_12682.jpg')
    img_copy = np.array(image.convert('RGB')).copy()
    with open('../Dataset/1.competition_trainset/1_dataset/Close_CAPP_cam5_10_12682.txt', 'r') as file:
        for line in file:
            cx, cy, w, h = map(float, line.split()[1:])
            x1 = 640 * (cx + 0.5 * w)
            x2 = 640 * (cx - 0.5 * w)
            y1 = 480 * (cy + 0.5 * h)
            y2 = 480 * (cy - 0.5 * h)
            
            cv2.rectangle(img_copy, (int(x1), int(y1)), (int(x2), int(y2)), color=(255, 0, 0))

    plt.imshow(img_copy)
    plt.show()

if __name__ == '__main__':
    main()
