import os
import pickle
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from make_filename import make_filename

class Cam:
    def __init__(self, H):
        self.H = H # homography matrix

    def transform_to_2d(self, cx, cy, w, h):
        img_w = 640
        img_h = 480

        px, py, pw, ph = cx * img_w, cy * img_h, w * img_w, h * img_h

        estimate_px = px
        estimate_py = py + 0.5 * ph

        point = np.array([[estimate_px, estimate_py]], dtype = np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(point, self.H)

        return transformed.reshape(-1)[0], transformed.reshape(-1)[1]

def load_yolo_labels(txt_path):
    ''' (class, cx, cy, w, h) '''
    labels = []
    if not os.path.exists(txt_path):
        return labels

    with open(txt_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) == 5:
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
                labels.append((cls, cx, cy, w, h))
            else:
                print(f'error: unknown format {txt_path}')

    return labels

def cluster(txtfile_lst, cam_lst):
    projected_lst = []

    for cam_idx, txt_path in enumerate(txtfile_lst):
        labels = load_yolo_labels(txt_path)
        for label in labels:
            projected_point = cam_lst[cam_idx].transform_to_2d(*label[1:])
            projected_lst.append([label[0], projected_point[0], projected_point[1]])

    projected_lst = np.array(projected_lst)
    unique_classes = np.unique(projected_lst[:, 0])
    result = []


    for cls in unique_classes:
        
        cls_mask = projected_lst[:, 0] == cls
        cls_points = projected_lst[cls_mask][:, 1:]

        if len(cls_points) == 0:
            continue

        clustering = DBSCAN(eps=75, min_samples=1).fit(cls_points)
        cluster_labels = clustering.labels_

        unique_clusters = np.unique(cluster_labels)
        valid_clusters = [cid for cid in unique_clusters if cid != -1]

        if not valid_clusters:
            continue

        best_cluster_id = None
        max_element_count = -1

        for cluster_id in valid_clusters:

            element_count = np.sum(cluster_labels == cluster_id)

            if element_count > max_element_count:
                max_element_count = element_count
                best_cluster_id = cluster_id

        if best_cluster_id is not None:

            cluster_mask = cluster_labels == cluster_id
            matched_points = cls_points[cluster_mask]

            mean_x = np.mean(matched_points[:, 0])
            mean_y = np.mean(matched_points[:, 1])

            result.append([int(cls), round(mean_x, 2) / 640, round(mean_y, 2) / 480])
    
    return result


def main():
    with open('homography_matrix_for_train_dataset.pkl', 'rb') as f:
        H = pickle.load(f)
    
    cam1 = Cam(H[0])
    cam2 = Cam(H[1])
    cam3 = Cam(H[2])
    cam4 = Cam(H[3])
    cam5 = Cam(H[4])
    cam_lst = [cam1, cam2, cam3, cam4, cam5]

    with open('filename.txt', 'r') as file:
        for line in file:
            filenames = make_filename(line)
            txtfile_lst = filenames[1]
            new_filename = filenames[2]

            new_labeling = cluster(txtfile_lst, cam_lst)

            with open(new_filename, 'w') as file:
                for l_new in new_labeling:
                    file.write(' '.join(map(str,l_new)) + '\n')



if __name__ == '__main__':
    main()

