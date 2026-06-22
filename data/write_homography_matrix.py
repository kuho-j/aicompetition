import numpy as np
import cv2
import pickle

class MultiViewPlaneIntegrator:
    def __init__(self):
        # 각 카메라별 가상 평면 변환 행렬(Homography)을 저장할 딕셔너리
        self.homography_matrices = {}
        
    def calibrate_camera_view(self, cam_idx, src_points, dst_points):
        """
        각 시점의 이미지 점들과 가상 평면의 실제 격자 좌표를 매칭하여 변환 행렬을 생성합니다.
        Args:
            cam_idx (int): 카메라 번호 (0 ~ 4)
            src_points (list or np.array): 이미지 상에서 찾은 컬러 점 좌표 [[u, v], ...] (최소 4개 이상)
            dst_points (list or np.array): 가상 평면계에서의 실제 목표 좌표 [[x, y], ...]
        """
        src_pts = np.array(src_points, dtype=np.float32)
        dst_pts = np.array(dst_points, dtype=np.float32)
        
        # 호모그래피 변환 행렬 계산
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        self.homography_matrices[cam_idx] = H
        print(f"[Cam {cam_idx}] 캘리브레이션 완료 (변환 행렬 생성됨)")
        print(f"변환 행혈:\n{H}")

    def transform_labels_to_plane(self, cam_idx, object_labels):
        """
        특정 시점 카메라의 물체 라벨(좌표)을 가상 평면 좌표로 변환합니다.
        Args:
            cam_idx (int): 카메라 번호
            object_labels (np.array): 이미지 상의 물체 위치 [[u1, v1], [u2, v2], ...]
        Returns:
            np.array: 가상 평면 위의 [X, Y] 좌표 리스트
        """
        if cam_idx not in self.homography_matrices:
            raise ValueError(f"카메라 {cam_idx}의 캘리브레이션이 먼저 수행되어야 합니다.")
            
        H = self.homography_matrices[cam_idx]
        
        # OpenCV perspectiveTransform 사용을 위해 데이터 구조 변경 (N, 1, 2)
        pts = np.array(object_labels, dtype=np.float32).reshape(-1, 1, 2)
        
        # 가상 평면으로 좌표 투영
        transformed_pts = cv2.perspectiveTransform(pts, H)
        
        # 다시 원래 포맷 (N, 2)로 되돌려 반환
        return transformed_pts.reshape(-1, 2)

# --- 사용 예시 (실행 단계) ---
if __name__ == "__main__":
    integrator = MultiViewPlaneIntegrator()
    
    # points from the each camera view
    ''' this is for the background images
    image_points = {0 : [[237, 156],
                         [262, 200],
                         [131, 191],
                         [286, 152]],
                    1 : [[304, 167],
                         [284, 210],
                         [337, 209],
                         [354, 162]],
                    2 : [[282, 152],
                         [280, 248],
                         [395, 250],
                         [392, 155]],
                    3 : [[256, 186],
                         [263, 240],
                         [314, 258],
                         [309, 197]],
                    4 : [[343, 197],
                         [336, 261],
                         [386, 244],
                         [393, 189]]}
    '''

    ''' this is for the test dataset background images '''
    image_points = {0 : [[310, 355],
                         [362, 355],
                         [35, 204],
                         [595, 201]],
                    1 : [[255, 207],
                         [472, 185],
                         [384, 334],
                         [520, 327]],
                    2 : [[429, 150],
                         [624, 187],
                         [343, 260],
                         [446, 299]],
                    3 : [[422, 204],
                         [207, 182],
                         [285, 350],
                         [154, 315]],
                    4 : [[206, 193],
                         [11, 217],
                         [309, 319],
                         [189, 334]],
                    }




                         
    
    # coordinates of virtual plane
    virtual_plane_points = [
        [284, 160],
        [284, 240],
        [355, 240],
        [355, 160]
    ]

    ''' this is for train_dataset '''
    virtual_plane_points = {0 : [[310, 320],
                                 [330, 320],
                                 [0, 0],
                                 [640, 0]],
                            1 : [[640, 0],
                                 [640, 480],
                                 [330, 160],
                                 [310, 320]],
                            2 : [[640, 0],
                                 [640, 480],
                                 [330, 160],
                                 [310, 320]],
                            3 : [[0, 0],
                                 [0, 480],
                                 [330, 160],
                                 [310, 320]],
                            4 : [[0, 0],
                                 [0, 480],
                                 [330, 160],
                                 [310, 320]],
                            }







    
    for i in range(5):
        integrator.calibrate_camera_view(cam_idx=i, src_points=image_points[i], dst_points=virtual_plane_points[i])
    
    # save the dictionary
    with open('homography_matrix_for_train_dataset.pkl', 'wb') as f:
        pickle.dump(integrator.homography_matrices, f)

    # 2. 이후 AI가 2번 카메라 이미지 내에서 새로운 물체(라벨)를 탐지했을 때
    # 물체의 바닥면 중심 좌표가 [300, 400]이라고 가정
    detected_objects_cam2 = np.array([[300, 400]])
    
    # 3. 통합 가상 평면 좌표로 변환
    integrated_coordinates = integrator.transform_labels_to_plane(cam_idx=2, object_labels=detected_objects_cam2)
    
    print("\n=== 변환 결과 ===")
    print(f"카메라 2 이미지 내 물체 좌표: {detected_objects_cam2[0]}")
    print(f"통합 가상 평면 내 변환 좌표 (X, Y): {integrated_coordinates[0]}")
