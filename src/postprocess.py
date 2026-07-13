import torch


def apply_homography_to_points(
    points: torch.Tensor,
    homography: torch.Tensor,
    invert: bool = False,
) -> torch.Tensor:
    """
    Apply a homography to 2D points.

    points:
        [N, 2] or [B, N, 2]
    homography:
        [3, 3] or [B, 3, 3]

    Coordinates are used as-is. Use image_centers_to_bev() when converting the
    detector's normalized [0, 1] image centers with GridToBEVLayer homographies.
    """
    original_dim = points.dim()
    if original_dim == 2:
        points = points.unsqueeze(0)
    elif original_dim != 3:
        raise ValueError(f"points must have shape [N, 2] or [B, N, 2], got {tuple(points.shape)}")

    if points.shape[-1] != 2:
        raise ValueError(f"points last dimension must be 2, got {tuple(points.shape)}")

    homography = torch.as_tensor(homography, device=points.device, dtype=points.dtype)
    if homography.dim() == 2:
        if homography.shape != (3, 3):
            raise ValueError(f"homography must have shape [3, 3], got {tuple(homography.shape)}")
        homography = homography.unsqueeze(0).expand(points.shape[0], -1, -1)
    elif homography.dim() == 3:
        if homography.shape[1:] != (3, 3):
            raise ValueError(f"homography must have shape [B, 3, 3], got {tuple(homography.shape)}")
        if homography.shape[0] == 1 and points.shape[0] != 1:
            homography = homography.expand(points.shape[0], -1, -1)
        elif homography.shape[0] != points.shape[0]:
            raise ValueError(
                f"homography batch size must be 1 or {points.shape[0]}, got {homography.shape[0]}"
            )
    else:
        raise ValueError(f"homography must have shape [3, 3] or [B, 3, 3], got {tuple(homography.shape)}")

    if invert:
        homography = torch.linalg.inv(homography)

    ones = torch.ones(*points.shape[:2], 1, device=points.device, dtype=points.dtype)
    points_h = torch.cat([points, ones], dim=-1)
    transformed = points_h @ homography.transpose(1, 2)
    transformed_xy = transformed[..., :2] / _safe_denominator(transformed[..., 2:])

    if original_dim == 2:
        return transformed_xy.squeeze(0)
    return transformed_xy


def image_centers_to_bev(
    centers: torch.Tensor,
    homography: torch.Tensor,
    output_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """
    Convert detector center coordinates to BEV coordinates.

    centers are normalized image/heatmap coordinates in [0, 1], matching
    decode_predictions(). homography is the GridToBEVLayer matrix that maps BEV
    normalized coordinates [-1, 1] to image-feature normalized coordinates
    [-1, 1]. The inverse is applied here.

    If output_size=(height, width) is supplied, returned centers are BEV pixel
    coordinates ordered as (x, y). Otherwise they remain normalized [0, 1].
    """
    centers = torch.as_tensor(centers, dtype=torch.float32)
    if centers.numel() == 0:
        return centers.clone()

    image_norm = centers * 2.0 - 1.0
    bev_norm = apply_homography_to_points(image_norm, homography, invert=True)
    bev_centers = (bev_norm + 1.0) * 0.5

    if output_size is None:
        return bev_centers

    height, width = output_size
    scale = bev_centers.new_tensor([width, height])
    return bev_centers * scale


def transform_detections_to_bev(
    model_outputs: dict | None = None,
    detections: list[dict[str, torch.Tensor]] | None = None,
    homography: torch.Tensor | None = None,
    centers: torch.Tensor | None = None,
    classes: torch.Tensor | None = None,
    scores: torch.Tensor | None = None,
    output_size: tuple[int, int] | None = None,
) -> list[dict[str, torch.Tensor]] | dict[str, torch.Tensor]:
    """
    Transform model detections, or explicitly supplied centers, into BEV space.

    Usage:
        transform_detections_to_bev(model_outputs=outputs)
        transform_detections_to_bev(homography=H, centers=centers, classes=classes)
    """
    if model_outputs is not None:
        if detections is None:
            detections = model_outputs.get("detections")
        if homography is None:
            homography = model_outputs.get("homography")

    if homography is None:
        raise ValueError("homography is required.")

    if centers is not None:
        bev_centers = image_centers_to_bev(
            centers,
            homography,
            output_size=output_size,
        )
        result = {"centers": bev_centers}
        if classes is not None:
            result["classes"] = classes
        if scores is not None:
            result["scores"] = scores
        return result

    if detections is None:
        raise ValueError("detections or centers are required.")

    homography_batch = torch.as_tensor(homography)
    transformed: list[dict[str, torch.Tensor]] = []
    for batch_idx, detection in enumerate(detections):
        if homography_batch.dim() == 3:
            h = homography_batch[min(batch_idx, homography_batch.shape[0] - 1)]
        else:
            h = homography_batch

        bev_centers = image_centers_to_bev(
            detection["centers"],
            h,
            output_size=output_size,
        )
        transformed_detection = {
            key: value
            for key, value in detection.items()
            if key != "centers"
        }
        transformed_detection["centers"] = bev_centers
        transformed.append(transformed_detection)

    return transformed


def _safe_denominator(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    sign = torch.where(x < 0, -torch.ones_like(x), torch.ones_like(x))
    return torch.where(x.abs() < eps, sign * eps, x)

# -*- coding: utf-8 -*-
"""
=========================================================================
5개 카메라 예측 결과를 합쳐서 라벨별 최종 상품 개수를 내는 통합 모듈.
=========================================================================

포함된 것:
    1. merge_5_predictions()     - 한 프레임의 5-view 예측을 합쳐 개수 산출
    2. TemporalSmoother          - 여러 프레임에 걸친 단순 안정화 (초기 실험용)
    3. ObjectTracker             - 물체별 상태 추적 (TRACKED/OCCLUDED/REMOVED)
                                   가려짐과 진짜 제거를 구분
    4. LiftAwareTracker (설계만)  - 물체가 잠깐 들렸다 놓이는 상황을 처리
                                   (2초 히스토리 기반, 실측 데이터로 튜닝 필요)

전제:
    - 각 카메라 예측의 centers는 이미 시점변환된 정규화 좌표(0~1)임
    - 5대 카메라, 정지 매대, 손님이 상품을 집어감

미해결/TODO 이슈들 (실제 예측 데이터 확보 후 처리):
    [1] jitter_tolerance, stable_position_frames 등 파라미터의 실측값
    [2] 프레임 fps 확정
    [3] 모델이 contact_points(접촉점 지도좌표)도 함께 리턴하는지 팀 확인
    [4] cam1 호모그래피 커버리지 이슈 (약 0.1% 샘플이 캔버스 밖)
        - 지금은 zero_max 리스트로 자동 스킵
    [5] 손 감지(hand detection) 모듈이 있는지, 없다면 대체 신호 확정

작성 이력:
    - 오늘: 기본 merge + camera-constraint clustering + 시간축 스무딩 + 상태 트래커
    - 다음: 실측 데이터로 파라미터 튜닝, LiftAwareTracker 완성
"""

import numpy as np
from collections import defaultdict


# =========================================================================
# 유틸
# =========================================================================
def _to_numpy(x):
    """torch tensor / numpy / list 전부 numpy로 통일."""
    if x is None:
        return np.zeros(0)
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


def _euclid(p, q):
    return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5


# =========================================================================
# 1. 카메라별 사전 정제
# =========================================================================
def _preprocess_camera(scores, classes, centers, cam_idx,
                       score_threshold, class_thresholds,
                       intra_nms_distance, fov_mask):
    """
    한 카메라의 예측 정제: score 필터, 시야 마스크, 내부 NMS.
    Returns: list of (class_id, x, y, cam_idx, score)
    """
    scores = _to_numpy(scores).astype(float)
    classes = _to_numpy(classes).astype(int)
    centers = _to_numpy(centers).astype(float)

    if len(scores) == 0:
        return []

    # score 필터 (클래스별 threshold 우선)
    keep = np.ones(len(scores), dtype=bool)
    for i, cls in enumerate(classes):
        thresh = class_thresholds.get(int(cls), score_threshold) \
            if class_thresholds else score_threshold
        if scores[i] < thresh:
            keep[i] = False

    # 시야 마스크 필터
    if fov_mask is not None:
        h, w = fov_mask.shape
        for i, (cx, cy) in enumerate(centers):
            if not keep[i]:
                continue
            px, py = int(cx * w), int(cy * h)
            if 0 <= px < w and 0 <= py < h:
                if fov_mask[py, px] == 0:
                    keep[i] = False
            else:
                keep[i] = False

    survived = []
    for i in np.where(keep)[0]:
        survived.append((int(classes[i]), float(centers[i, 0]),
                         float(centers[i, 1]), cam_idx, float(scores[i])))

    # 같은 카메라 안 내부 NMS
    if intra_nms_distance is not None and intra_nms_distance > 0:
        survived = _intra_camera_nms(survived, intra_nms_distance)

    return survived


def _intra_camera_nms(points, distance):
    """같은 카메라 안 클래스별로 너무 가까운 점들은 score 높은 것만 남김."""
    if len(points) <= 1:
        return points
    by_cls = defaultdict(list)
    for p in points:
        by_cls[p[0]].append(p)

    kept = []
    for cls, plist in by_cls.items():
        plist = sorted(plist, key=lambda p: -p[4])  # score 내림차순
        used = [False] * len(plist)
        for i in range(len(plist)):
            if used[i]:
                continue
            kept.append(plist[i])
            for j in range(i + 1, len(plist)):
                if used[j]:
                    continue
                if _euclid((plist[i][1], plist[i][2]),
                           (plist[j][1], plist[j][2])) <= distance:
                    used[j] = True
    return kept


# =========================================================================
# 2. 카메라 제약 클러스터링 (핵심)
# =========================================================================
def _cluster_with_camera_constraint(points, threshold):
    """
    거리 기반 그리디 클러스터링, "한 클러스터에 같은 카메라 최대 1개" 강제.
    (fuse_multiview_counts.py와 동일한 로직)
    Returns: list of clusters (각 cluster는 원본 인덱스의 list)
    """
    n = len(points)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    parent = list(range(n))
    root_cams = [{points[i][2]} for i in range(n)]

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # 후보 엣지 (다른 카메라 + threshold 이내), 거리순 정렬
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if points[i][2] == points[j][2]:
                continue
            d = _euclid((points[i][0], points[i][1]),
                        (points[j][0], points[j][1]))
            if d <= threshold:
                edges.append((d, i, j))
    edges.sort()

    # 그리디 병합 (카메라 집합 겹치면 거부 = 전이적 병합 방지)
    for d, i, j in edges:
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        if root_cams[ri] & root_cams[rj]:
            continue
        parent[rj] = ri
        root_cams[ri] |= root_cams[rj]

    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)
    return list(clusters.values())


# =========================================================================
# 3. 메인 함수: 한 프레임 5-view 합치기
# =========================================================================
def merge_5_predictions(
    predictions,
    score_threshold: float = 0.3,
    distance_threshold: float = 0.04,
    min_cameras_per_object: int = 1,
    intra_camera_nms_distance: float = 0.02,
    class_thresholds: dict = None,
    fov_masks: list = None,
    return_details: bool = False,
):
    """
    5개(또는 그 이하) 카메라의 한 프레임 예측을 합쳐서 라벨별 개수를 반환.

    Args:
        predictions: list of dict.
            각 원소는 {"scores", "classes", "centers"}.
            centers는 시점변환된 정규화 좌표(0~1).
        score_threshold: 기본 score 임계값.
        distance_threshold: 클러스터링 거리 임계값 (정규화 스케일).
        min_cameras_per_object: 이 값 미만 카메라만 본 클러스터는 카운트에서 제외.
        intra_camera_nms_distance: 같은 카메라 안 중복 검출 흡수 거리.
        class_thresholds: {class_id: threshold} 특정 클래스 별도 임계값.
        fov_masks: list of 2D numpy array (각 카메라 시야 마스크).
        return_details: True면 (counts, details) 튜플 반환.

    Returns:
        기본: {class_id: count}
        return_details=True: (counts, details) — details는 클러스터별 상세
    """
    if class_thresholds is None:
        class_thresholds = {}

    # 1. 카메라별 정제
    all_points = []
    for cam_idx, pred in enumerate(predictions):
        fov = fov_masks[cam_idx] if fov_masks and cam_idx < len(fov_masks) else None
        pts = _preprocess_camera(
            pred.get("scores"), pred.get("classes"), pred.get("centers"),
            cam_idx, score_threshold, class_thresholds,
            intra_camera_nms_distance, fov,
        )
        all_points.extend(pts)

    if not all_points:
        return ({}, {}) if return_details else {}

    # 2. 클래스별 그룹화
    by_class = defaultdict(list)
    for cls, x, y, cam_idx, score in all_points:
        by_class[cls].append((x, y, cam_idx, score))

    # 3. 클래스별 클러스터링
    counts = {}
    details = {}
    for cls, pts in by_class.items():
        clusters = _cluster_with_camera_constraint(pts, distance_threshold)

        valid_clusters = []
        for c in clusters:
            unique_cams = set(pts[i][2] for i in c)
            if len(unique_cams) >= min_cameras_per_object:
                valid_clusters.append(c)

        counts[cls] = len(valid_clusters)

        if return_details:
            details[cls] = []
            for c in valid_clusters:
                cluster_pts = [pts[i] for i in c]
                total_w = sum(p[3] for p in cluster_pts)
                if total_w > 0:
                    rep_x = sum(p[0] * p[3] for p in cluster_pts) / total_w
                    rep_y = sum(p[1] * p[3] for p in cluster_pts) / total_w
                else:
                    rep_x = np.mean([p[0] for p in cluster_pts])
                    rep_y = np.mean([p[1] for p in cluster_pts])
                max_conf = max(p[3] for p in cluster_pts)
                details[cls].append({
                    "center": (rep_x, rep_y),
                    "confidence": max_conf,
                    "n_cameras": len(set(p[2] for p in cluster_pts)),
                    "cameras": sorted(set(p[2] for p in cluster_pts)),
                })

    return (counts, details) if return_details else counts


# =========================================================================
# 4. 시간축 안정화 (단순 버전)
# =========================================================================
class TemporalSmoother:
    """
    여러 프레임 결과를 부드럽게 이어줌.
    반짝 나오는 오검출 필터, 잠깐 놓친 물체 유지.
    (물체 정체성은 추적하지 않음 - 그건 ObjectTracker의 역할)
    """
    def __init__(self, window_size: int = 5, min_appearances: int = 3):
        self.window = []
        self.window_size = window_size
        self.min_appearances = min_appearances

    def update(self, counts: dict):
        self.window.append(counts)
        if len(self.window) > self.window_size:
            self.window.pop(0)

        appearances = defaultdict(int)
        count_sums = defaultdict(list)
        for frame_counts in self.window:
            for cls, cnt in frame_counts.items():
                appearances[cls] += 1
                count_sums[cls].append(cnt)

        stable = {}
        for cls, n_appear in appearances.items():
            if n_appear >= self.min_appearances:
                stable[cls] = int(np.median(count_sums[cls]))
        return stable


# =========================================================================
# 5. 상태 기반 트래커 (물체 정체성 추적)
# =========================================================================
class ObjectTracker:
    """
    각 물체에 track ID를 부여하고 상태(TRACKED/OCCLUDED/REMOVED) 관리.
    "안 보임" != "제거됨"을 구분.

    사용:
        tracker = ObjectTracker()
        for frame in video:
            _, details = merge_5_predictions(preds, return_details=True)
            counts = tracker.update(details, hand_present=frame_has_hand)
    """

    STATE_TRACKED = "tracked"
    STATE_OCCLUDED = "occluded"
    STATE_REMOVED = "removed"

    def __init__(
        self,
        match_distance: float = 0.05,
        occlusion_grace_frames: int = 30,  # 이 프레임 넘게 안 보이면 REMOVED 후보
        removal_needs_hand: bool = True,   # True면 손 지나갔던 경우에만 REMOVED
    ):
        self.tracks = {}
        self._next_id = 0
        self.match_distance = match_distance
        self.occlusion_grace = occlusion_grace_frames
        self.removal_needs_hand = removal_needs_hand

    def update(self, details: dict, hand_present: bool = False):
        # 관측 평탄화
        observations = []
        for cls, obj_list in details.items():
            for obj in obj_list:
                x, y = obj["center"]
                observations.append((cls, x, y, obj["confidence"]))

        # 기존 track과 매칭 (그리디)
        matched_obs = set()
        for track_id, tr in self.tracks.items():
            if tr["state"] == self.STATE_REMOVED:
                continue
            best_j, best_d = -1, self.match_distance
            for j, (cls, x, y, conf) in enumerate(observations):
                if j in matched_obs or cls != tr["class_id"]:
                    continue
                d = _euclid((x, y), tr["last_center"])
                if d < best_d:
                    best_d, best_j = d, j
            if best_j >= 0:
                cls, x, y, conf = observations[best_j]
                tr["last_center"] = (x, y)
                tr["last_conf"] = conf
                tr["missed_frames"] = 0
                tr["state"] = self.STATE_TRACKED
                tr["hand_seen_during_occlusion"] = False
                matched_obs.add(best_j)
            else:
                tr["missed_frames"] += 1
                if tr["state"] == self.STATE_TRACKED:
                    tr["state"] = self.STATE_OCCLUDED
                if hand_present:
                    tr["hand_seen_during_occlusion"] = True
                if self._should_mark_removed(tr):
                    tr["state"] = self.STATE_REMOVED

        # 매칭 안 된 관측 = 새 track
        for j, (cls, x, y, conf) in enumerate(observations):
            if j in matched_obs:
                continue
            self.tracks[self._next_id] = {
                "class_id": cls,
                "last_center": (x, y),
                "last_conf": conf,
                "missed_frames": 0,
                "state": self.STATE_TRACKED,
                "hand_seen_during_occlusion": False,
            }
            self._next_id += 1

        # 카운트 (TRACKED + OCCLUDED, REMOVED 제외)
        counts = defaultdict(int)
        for tr in self.tracks.values():
            if tr["state"] != self.STATE_REMOVED:
                counts[tr["class_id"]] += 1
        return dict(counts)

    def _should_mark_removed(self, tr):
        if tr["missed_frames"] > self.occlusion_grace:
            if not self.removal_needs_hand:
                return True
            if tr.get("hand_seen_during_occlusion"):
                return True
        return False


# =========================================================================
# 6. 들림-인식 트래커 (설계, 실측 튜닝 대기)
# =========================================================================
class LiftAwareTracker(ObjectTracker):
    """
    ObjectTracker 확장판.
    물체가 잠깐 들렸다 놓이는 상황을 흡수:
      - 최근 2초 위치 히스토리 유지
      - 값이 튀었다가 원래 자리 근처로 돌아오면 "현실 세계 변수"로 판단, 개수 유지
      - 2초 지나도 원래 자리에 안 돌아오면 진짜 이동/제거로 확정

    ⚠️ 이 클래스는 설계만 되어 있고 실측 파라미터 튜닝이 필요함.
    실제 예측 데이터가 확보된 뒤 아래 값들을 조정할 것:
      - history_seconds: 얼마나 오래 기억할지 (감으로 2초, 실제 손 개입 시간으로 재조정)
      - fps: 실제 영상 프레임 속도
      - jitter_tolerance: "같은 자리"로 볼 거리 (실측 튐 크기 관찰 후 결정)
      - stable_position_frames: 안정 판정에 필요한 프레임 수

    가능하다면 모델이 접촉점(contact_point)도 리턴하게 하면
    "중심점 vs 접촉점 차이"로 들림 상태를 더 확실히 감지 가능.
    """

    def __init__(
        self,
        history_seconds: float = 2.0,
        fps: int = 30,
        jitter_tolerance: float = 0.05,
        stable_position_frames: int = 15,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.history_frames = int(history_seconds * fps)
        self.jitter_tolerance = jitter_tolerance
        self.stable_position_frames = stable_position_frames

    def update(self, details, hand_present=False):
        # TODO: super().update() 확장해서 아래 로직 삽입
        # 1. 매칭된 track마다 position_history에 이번 위치 append
        # 2. 히스토리가 history_frames 넘으면 앞에서 pop
        # 3. 최근 stable_position_frames 중앙값 vs 그 이전 중앙값 비교
        #    - 두 중앙값이 jitter_tolerance 안 → 튐 있었어도 원래 자리, 개수 유지
        #    - 크게 다름 → 진짜 이동, stable_position 업데이트
        # 4. 흔들림 상태에서 REMOVED로 넘어가는 로직은 부모 클래스에 위임
        raise NotImplementedError("실측 데이터 확보 후 구현 및 튜닝 필요")
