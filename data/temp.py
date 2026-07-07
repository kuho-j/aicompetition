import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt

from data.make_filename import make_filename

'''
yeild homography matrices that convert train dataset view to given video view
'''
'''
with open('data/homography_matrix_for_train_dataset.pkl', 'rb') as f:
    H_train_dict = pickle.load(f)

with open('data/homography_matrix.pkl', 'rb') as f:
    H_video_dict = pickle.load(f)

# H(train -> video) = H-1(train -> plane) * H(video -> plane)
# train - video
# cam1 - cam2
# cam2 - cam4
# cam3 - cam0
# cam4 - cam3
# cam5 - cam1

H_cam1 = np.linalg.inv(H_train_dict[0]) @ H_video_dict[2]
H_cam2 = np.linalg.inv(H_train_dict[1]) @ H_video_dict[4]
H_cam3 = np.linalg.inv(H_train_dict[2]) @ H_video_dict[0]
H_cam4 = np.linalg.inv(H_train_dict[3]) @ H_video_dict[3]
H_cam5 = np.linalg.inv(H_train_dict[4]) @ H_video_dict[1]

H = {0 : H_cam1,
     1 : H_cam2,
     2 : H_cam3,
     3 : H_cam4,
     4 : H_cam5}

with open('data/homography_matrix_train_to_video.pkl', 'wb') as f:
    pickle.dump(H, f)
'''

'''
show an image with converted view by given homogrphy matrices
'''

homography_path = 'homography_matrix_train_to_video.pkl'
file_info = ''
img_path_list = make_filename(file_info)[0]


with open(homography_path, 'rb') as f:
    homography_dict = pickle.load(f)

img_list = []

for img_path in img_path_list:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_list.append(img)

row, col, ch = 480, 640, 3

cvt_img_list = [cv2.warpPerspective(img_list[i], homography_dict[i], (col, row)) for i in range(0, 5)]

for i in range(1, 6):
    plt.subplot(5, 2, 2*i-1)
    plt.title(f'original cam{i}')
    plt.imshow(img_list[i-1])
    plt.axis('off')

    plt.subplot(5, 2, 2*i)
    plt.title(f'converted cam{i}')
    plt.imshow(cvt_img_list[i-1])
    plt.axis('off')

plt.tight_layout()
plt.show()