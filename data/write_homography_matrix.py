from pathlib import Path
import pickle

import cv2
import numpy as np


class SingleViewPlaneIntegrator:
    def __init__(self):
        self.homography_matrices = {}

    @staticmethod
    def _valid_point_pairs(image_points, plane_points):
        if len(image_points) != len(plane_points):
            raise ValueError(
                f"image_points and plane_points must have the same length: "
                f"{len(image_points)} != {len(plane_points)}"
            )

        src_points = []
        dst_points = []
        valid_indices = []

        for idx, (image_point, plane_point) in enumerate(zip(image_points, plane_points)):
            if image_point is None:
                continue
            if len(image_point) != 2 or len(plane_point) != 2:
                raise ValueError(f"Point at index {idx} must be [x, y] or None.")

            src_points.append(image_point)
            dst_points.append(plane_point)
            valid_indices.append(idx)

        if len(src_points) < 4:
            raise ValueError(
                f"At least 4 non-None image points are required, got {len(src_points)}."
            )

        return (
            np.asarray(src_points, dtype=np.float32),
            np.asarray(dst_points, dtype=np.float32),
            valid_indices,
        )

    @staticmethod
    def _robust_homography(src_pts, dst_pts, reproj_threshold=5.0):
        # With only 4 points, no extra correspondence exists for outlier rejection.
        if len(src_pts) == 4:
            h, _ = cv2.findHomography(src_pts, dst_pts, method=0)
            if h is None:
                raise RuntimeError("Failed to estimate a homography from 4 points.")
            inlier_mask = np.ones((4, 1), dtype=np.uint8)
            return h, inlier_mask

        method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
        h, inlier_mask = cv2.findHomography(
            src_pts,
            dst_pts,
            method=method,
            ransacReprojThreshold=reproj_threshold,
            maxIters=10000,
            confidence=0.995,
        )

        if h is None or inlier_mask is None:
            h, inlier_mask = cv2.findHomography(
                src_pts,
                dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=reproj_threshold,
                maxIters=10000,
                confidence=0.995,
            )

        if h is None or inlier_mask is None:
            raise RuntimeError("Failed to estimate a homography from the provided points.")

        inlier_mask = inlier_mask.astype(bool).reshape(-1)
        if np.count_nonzero(inlier_mask) >= 4:
            refined_h, _ = cv2.findHomography(src_pts[inlier_mask], dst_pts[inlier_mask], method=0)
            if refined_h is not None:
                h = refined_h

        return h, inlier_mask.reshape(-1, 1).astype(np.uint8)

    def calibrate_camera_view(self, cam_idx, image_points, plane_points, reproj_threshold=5.0):
        """
        Estimate one view's homography from image points to virtual plane points.

        Args:
            cam_idx (int): Camera/view id to save in the output dict.
            image_points (list): Image coordinates. Each item is [u, v] or None.
            plane_points (list): Matching virtual plane coordinates [[x, y], ...].
            reproj_threshold (float): RANSAC/USAC reprojection threshold in pixels.
        """
        src_pts, dst_pts, valid_indices = self._valid_point_pairs(image_points, plane_points)
        h, inlier_mask = self._robust_homography(src_pts, dst_pts, reproj_threshold)

        if abs(h[2, 2]) > 1e-12:
            h = h / h[2, 2]

        self.homography_matrices[cam_idx] = h

        inlier_indices = [
            valid_indices[i] for i, is_inlier in enumerate(inlier_mask.reshape(-1)) if is_inlier
        ]
        print(f"[Cam {cam_idx}] calibration complete")
        print(f"valid point indices: {valid_indices}")
        print(f"inlier point indices: {inlier_indices}")
        print(f"homography:\n{h}")

    def transform_labels_to_plane(self, cam_idx, object_labels):
        if cam_idx not in self.homography_matrices:
            raise ValueError(f"Camera {cam_idx} must be calibrated first.")

        h = self.homography_matrices[cam_idx]
        pts = np.asarray(object_labels, dtype=np.float32).reshape(-1, 1, 2)
        transformed_pts = cv2.perspectiveTransform(pts, h)
        return transformed_pts.reshape(-1, 2)

    def transform_plane_to_image(self, cam_idx, plane_points):
        if cam_idx not in self.homography_matrices:
            raise ValueError(f"Camera {cam_idx} must be calibrated first.")

        h_inv = np.linalg.inv(self.homography_matrices[cam_idx])
        pts = np.asarray(plane_points, dtype=np.float32).reshape(-1, 1, 2)
        transformed_pts = cv2.perspectiveTransform(pts, h_inv)
        return transformed_pts.reshape(-1, 2)

    def transform_virtual_grid_to_image(
        self,
        cam_idx,
        virtual_grid_points=((1, 1), (0, 1), (1, 0), (0, 0)),
        plane_size=(640, 480),
    ):
        plane_w, plane_h = plane_size
        plane_points = [
            [grid_x * plane_w, grid_y * plane_h]
            for grid_x, grid_y in virtual_grid_points
        ]
        return self.transform_plane_to_image(cam_idx, plane_points)


MultiViewPlaneIntegrator = SingleViewPlaneIntegrator


if __name__ == "__main__":
    integrator = SingleViewPlaneIntegrator()

    # Select exactly one view to calibrate.
    cam_idx = 0

    # About 9 points on the shared virtual plane. Keep this list fixed, then put
    # either [u, v] or None at the same index in image_points below.
    virtual_plane_points = [
        [0, 0],
        [320, 0],
        [640, 0],
        [0, 240],
        [320, 240],
        [640, 240],
        [0, 480],
        [320, 480],
        [640, 480],
    ]

    # Image points for only cam_idx. None means that virtual-plane point is not
    # visible or has not been manually annotated in this image.

    # trainClose cam1
    trainClose_cam1 = [
        [36, 204],
        [316, 196],
        [595, 201],
        None,
        [310, 304],
        None,
        None,
        [320, 504],
        None,
    ]

    # trainClose cam2
    trainClose_cam2 = [
        None,
        [254, 351],
        [253, 204],
        None,
        [448, 334],
        [368, 193],
        None,
        [623, 312],
        [472, 184]
    ]

    trainClose_cam3 = [
        None,
        [271, 231],
        [431, 150],
        None,
        [295, 276],
        [522, 162],
        None,
        [564, 339],
        [622, 186]
    ]

    trainClose_cam4 = [
        [421, 204],
        [416, 362],
        None,
        [311, 189],
        [222, 331],
        None,
        [208, 181],
        [47, 304],
        None,
    ]

    trainClose_cam5 = [
        [206, 191],
        [409, 297],
        None,
        [118, 201],
        [240, 327],
        None,
        [6, 215],
        [61, 357],
        None
    ]

    trainLong_cam1 = [
        None,
        [270, 189],
        [518, 240],
        None,
        [287, 308],
        [556, 311],
        None,
        [312, 501],
        [599, 413]
    ]

    trainLong_cam2 = trainClose_cam2
    trainLong_cam3 = trainClose_cam3
    trainLong_cam4 = trainClose_cam4

    trainLong_cam5 = [
        [219, 184],
        [432, 276],
        None,
        [130, 200],
        [268, 331],
        None,
        [25, 228],
        [98, 380],
        None
    ]

    trainMiddle_cam1 = [
        [28, 182],
        [306, 168],
        [588, 184],
        None,
        [310, 307],
        None,
        None,
        [317, 478],
        None
    ]

    trainMiddle_cam2 = trainLong_cam2
    trainMiddle_cam3 = trainLong_cam3
    trainMiddle_cam4 = trainLong_cam4
    trainMiddle_cam5 = trainLong_cam5

    backgroundwhite_cam0 = [
        [103, 60],
        [305, 146],
        [418, 203],
        [44, 206],
        [324, 272],
        [447, 298],
        [32, 464],
        [362, 432],
        None
    ]

    backgroundwhite_cam1 = [
        [214, 185],
        [339, 129],
        [510, 26],
        [181, 272],
        [312, 252],
        [584, 204],
        None,
        [274, 424],
        [589, 461],
    ]

    backgroundwhite_cam2 = [
        [40, 50],
        [337, 15],
        [614, 62],
        [19, 248],
        [333, 268],
        [631, 253],
        [59, 434],
        None,
        [603, 422],
    ]

    backgroundwhite_cam3 = [
        [79, 88],
        [189, 61],
        [556, 68],
        [91, 206],
        [207, 245],
        [504, 331],
        [115, 307],
        None,
        [439, 462]
    ]

    backgroundwhite_cam4 = [
        [57, 39],
        [391, 97],
        [502, 139],
        [60, 333],
        [335, 284],
        [475, 255],
        [128, 482],
        None,
        [448, 346],
    ]

    background_cam0 = [
        [9, 109],
        [226, 90],
        [348, 100],
        [14, 254],
        [290, 197],
        [413, 173],
        [129, 502],
        None,
        None
    ]

    background_cam1 = [
        [223, 125],
        [353, 98],
        [573, 76],
        [182, 212],
        [305, 210],
        [588, 219],
        None,
        None,
        [548, 473]
    ]

    background_cam2 = [
        [44, 37],
        None,
        [619, 48],
        [23, 236],
        [337, 248],
        [635, 241],
        [61, 419],
        [342, 468],
        [608, 400]
    ]
    background_cam3 = [
        [148, 94],
        [263, 64],
        None,
        [168, 213],
        [289, 248],
        [576, 342],
        [191, 306],
        None,
        [507, 473]
    ]

    background_cam4 = [
        [40, 46],
        [379, 66],
        [496, 99],
        [75, 330],
        [362, 252],
        [479, 214],
        [146, 473],
        None,
        [460, 310]
    ]

    image_points = {
        'trainClose_cam1' : trainClose_cam1,
        'trainClose_cam2' : trainClose_cam2,
        'trainClose_cam3' : trainClose_cam3,
        'trainClose_cam4' : trainClose_cam4,
        'trainClose_cam5' : trainClose_cam5,
        'trainLong_cam1' : trainLong_cam1,
        'trainLong_cam2' : trainLong_cam2,
        'trainLong_cam3' : trainLong_cam3,
        'trainLong_cam4' : trainLong_cam4,
        'trainLong_cam5' : trainLong_cam5,
        'trainMiddle_cam1' : trainMiddle_cam1,
        'trainMiddle_cam2' : trainMiddle_cam2,
        'trainMiddle_cam3' : trainMiddle_cam3,
        'trainMiddle_cam4' : trainMiddle_cam4,
        'trainMiddle_cam5' : trainMiddle_cam5,
        'backgroundwhite_cam0' : backgroundwhite_cam0,
        'backgroundwhite_cam1' : backgroundwhite_cam1,
        'backgroundwhite_cam2' : backgroundwhite_cam2,
        'backgroundwhite_cam3' : backgroundwhite_cam3,
        'backgroundwhite_cam4' : backgroundwhite_cam4,
        'background_cam0' : background_cam0,
        'background_cam1' : background_cam1,
        'background_cam2' : background_cam2,
        'background_cam3' : background_cam3,
        'background_cam4' : background_cam4,
    }

    virtual_grid_points = [(0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
                           (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
                           (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)]

    calibration_results = {}
    for key, points in image_points.items():
        integrator.calibrate_camera_view(
            cam_idx=key,
            image_points=points,
            plane_points=virtual_plane_points,
            reproj_threshold=5.0,
        )
        grid_image_points = integrator.transform_virtual_grid_to_image(
            cam_idx=key,
            virtual_grid_points=virtual_grid_points,
            plane_size=(640, 480),
        )
        calibration_results[key] = {
            "homography": integrator.homography_matrices[key],
            "grid_points": grid_image_points,
        }

        print(f"\n=== {key} virtual grid points in image ===")
        for grid_point, image_point in zip(virtual_grid_points, grid_image_points):
            print(f"{grid_point} -> {image_point}")

    output_path = Path(__file__).with_name("homography_matrix_and_bev_grids.pkl")
    with output_path.open("wb") as f:
        pickle.dump(calibration_results, f)

    print(f"\nSaved {len(calibration_results)} calibration results to {output_path}")
