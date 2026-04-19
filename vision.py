"""
엔드이펙터 카메라 + OWL-ViT 객체 인식 모듈
- PyBullet computeViewMatrix / getCameraImage 로 RGB+Depth 취득
- OWL-ViT로 텍스트 쿼리 기반 객체 검출
- 2D bbox + depth → 3D 월드 좌표 변환 (deprojection)
"""

import math
import numpy as np
import pybullet as p
from PIL import Image
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# ── 카메라 파라미터 ────────────────────────────────────────────────────────────
IMG_W, IMG_H = 320, 320
FOV          = 60.0          # 수직 화각 (degree)
NEAR, FAR    = 0.01, 5.0

CONF_HIGH    = 0.5           # 이 이상이면 바로 사용
CONF_LOW     = 0.3           # 이 미만이면 OBJECT_NOT_FOUND

# OWL-ViT 모델 (최초 호출 시 로드)
_processor: OwlViTProcessor | None = None
_model: OwlViTForObjectDetection | None = None


def _load_model():
    global _processor, _model
    if _model is None:
        print("[Vision] OWL-ViT 모델 로딩 중...")
        _processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        _model     = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        _model.eval()
        print("[Vision] 모델 로딩 완료")


# ── 카메라 내부 파라미터 ──────────────────────────────────────────────────────
def _get_intrinsics():
    """FOV와 이미지 크기로부터 핀홀 카메라 내부 파라미터 계산"""
    fy = IMG_H / (2.0 * math.tan(math.radians(FOV / 2.0)))
    fx = fy   # 정사각 픽셀 가정
    cx = IMG_W / 2.0
    cy = IMG_H / 2.0
    return fx, fy, cx, cy


# ── 엔드이펙터 카메라 이미지 취득 ─────────────────────────────────────────────
def capture_ee_camera(robot_id: int, ee_link: int):
    """
    엔드이펙터 링크 위치/방향으로 카메라 뷰 행렬을 계산하고
    RGB 이미지 (H,W,3 uint8) 와 선형 depth 버퍼 (H,W float32) 를 반환.
    """
    link_state = p.getLinkState(robot_id, ee_link, computeForwardKinematics=True)
    ee_pos     = list(link_state[4])
    ee_orn     = link_state[5]

    # EE 로컬 +Z 방향 (그리퍼 손가락이 향하는 방향) = 카메라 광축
    rot_mat = p.getMatrixFromQuaternion(ee_orn)
    forward = [rot_mat[2], rot_mat[5], rot_mat[8]]    # +Z_local in world
    up      = [rot_mat[0], rot_mat[3], rot_mat[6]]    # +X_local in world

    target = [ee_pos[i] + forward[i] * 0.5 for i in range(3)]

    view_mat = p.computeViewMatrix(ee_pos, target, up)
    proj_mat = p.computeProjectionMatrixFOV(FOV, IMG_W / IMG_H, NEAR, FAR)

    _, _, rgba, depth_buf, _ = p.getCameraImage(
        IMG_W, IMG_H,
        viewMatrix=view_mat,
        projectionMatrix=proj_mat,
        renderer=p.ER_TINY_RENDERER,
    )

    rgb = np.array(rgba, dtype=np.uint8).reshape(IMG_H, IMG_W, 4)[:, :, :3]

    # depth_buf 는 [0,1] NDC → 선형 미터 단위로 변환
    depth_buf = np.array(depth_buf, dtype=np.float32).reshape(IMG_H, IMG_W)
    depth_m   = FAR * NEAR / (FAR - (FAR - NEAR) * depth_buf)

    return rgb, depth_m, view_mat


# ── OWL-ViT 검출 ──────────────────────────────────────────────────────────────
def detect_object(rgb: np.ndarray, text_query: str):
    """
    RGB 이미지에서 text_query 에 해당하는 객체를 검출.
    반환: (cx_px, cy_px, confidence, box_xyxy) 또는 None
    """
    _load_model()

    pil_img = Image.fromarray(rgb)
    inputs  = _processor(text=[[text_query]], images=pil_img, return_tensors="pt")

    with torch.no_grad():
        outputs = _model(**inputs)

    # 후처리: 원본 이미지 크기 기준 좌표로 변환
    target_sizes = torch.tensor([[IMG_H, IMG_W]])
    results = _processor.post_process_grounded_object_detection(
        outputs, threshold=0.1, target_sizes=target_sizes
    )[0]

    if len(results["scores"]) == 0:
        return None

    best_idx  = results["scores"].argmax().item()
    score     = results["scores"][best_idx].item()
    box       = results["boxes"][best_idx].tolist()   # [x1, y1, x2, y2]

    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0

    return cx, cy, score, box


# ── OWL-ViT 다중 쿼리 동시 검출 ──────────────────────────────────────────────
def detect_multiple(rgb: np.ndarray, text_queries: list) -> dict:
    """
    여러 텍스트 쿼리를 한 번의 추론으로 동시에 검출.
    반환: {query: (cx, cy, score, box)} — 미검출이면 해당 키 값 None
    """
    _load_model()

    pil_img = Image.fromarray(rgb)
    inputs  = _processor(text=[text_queries], images=pil_img, return_tensors="pt")

    with torch.no_grad():
        outputs = _model(**inputs)

    target_sizes = torch.tensor([[IMG_H, IMG_W]])
    results = _processor.post_process_grounded_object_detection(
        outputs, threshold=0.1, target_sizes=target_sizes
    )[0]

    detected = {q: None for q in text_queries}

    for score, label_idx, box in zip(results["scores"], results["labels"], results["boxes"]):
        score = score.item()
        label = text_queries[label_idx.item()]
        box   = box.tolist()
        if detected[label] is None or score > detected[label][2]:
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            detected[label] = (cx, cy, score, box)

    return detected


# ── Deprojection: 픽셀 + depth → 월드 좌표 ───────────────────────────────────
def pixel_to_world(cx_px: float, cy_px: float,
                   depth_m: np.ndarray, view_mat) -> np.ndarray | None:
    """
    픽셀 좌표 (cx_px, cy_px) 와 depth 맵으로부터 월드 3D 좌표를 계산.
    """
    ix, iy = int(round(cx_px)), int(round(cy_px))
    ix = max(0, min(IMG_W - 1, ix))
    iy = max(0, min(IMG_H - 1, iy))

    z = depth_m[iy, ix]
    if z <= 0.0 or z >= FAR:
        return None   # DEPTH_INVALID

    fx, fy, c_cx, c_cy = _get_intrinsics()

    # 카메라 좌표계
    x_cam = (cx_px - c_cx) * z / fx
    y_cam = (cy_px - c_cy) * z / fy
    z_cam = z

    # view_mat (column-major 16개 값) → 4×4 행렬
    V = np.array(view_mat).reshape(4, 4).T   # row-major 변환

    # 카메라 → 월드 변환 (view 역행렬)
    V_inv = np.linalg.inv(V)
    p_cam = np.array([x_cam, y_cam, z_cam, 1.0])
    p_world = V_inv @ p_cam

    return p_world[:3]


# ── 통합 파이프라인 ────────────────────────────────────────────────────────────
class VisionResult:
    def __init__(self, world_pos, confidence, box_xyxy, status):
        self.world_pos  = world_pos    # np.ndarray [x,y,z] 또는 None
        self.confidence = confidence
        self.box_xyxy   = box_xyxy
        self.status     = status       # "OK" | "LOW_CONF" | "GT_FALLBACK" | "NOT_FOUND" | "DEPTH_INVALID"

    def __repr__(self):
        if self.world_pos is not None:
            pos = [round(v, 4) for v in self.world_pos]
        else:
            pos = None
        return (f"VisionResult(status={self.status}, "
                f"conf={self.confidence:.3f}, world_pos={pos})")


def find_object(robot_id: int, ee_link: int, text_query: str,
                gt_body_id: int | None = None) -> VisionResult:
    """
    엔드이펙터 카메라로 text_query 객체를 탐색하고 3D 월드 좌표를 반환.

    gt_body_id: PyBullet body ID. 신뢰도가 CONF_LOW~CONF_HIGH 구간일 때
                해당 body의 GT 좌표로 폴백한다. None이면 폴백 없이 LOW_CONF 반환.
    """
    rgb, depth_m, view_mat = capture_ee_camera(robot_id, ee_link)

    det = detect_object(rgb, text_query)
    if det is None:
        print(f"[Vision] OBJECT_NOT_FOUND: '{text_query}'을(를) 찾을 수 없습니다.")
        return VisionResult(None, 0.0, None, "NOT_FOUND")

    cx, cy, conf, box = det
    print(f"[Vision] 검출: '{text_query}'  conf={conf:.3f}  box={[round(v,1) for v in box]}")

    if conf < CONF_LOW:
        print("[Vision] OBJECT_NOT_FOUND: 신뢰도가 너무 낮습니다 (< 0.3).")
        return VisionResult(None, conf, box, "NOT_FOUND")

    if conf < CONF_HIGH:
        print(f"[Vision] ⚠ 신뢰도 낮음 ({conf:.3f}, {CONF_LOW}~{CONF_HIGH}).")
        if gt_body_id is not None:
            gt_pos, _ = p.getBasePositionAndOrientation(gt_body_id)
            world_pos = np.array(gt_pos, dtype=np.float64)
            print(f"[Vision] GT 폴백 적용 (body_id={gt_body_id}): "
                  f"{[round(v, 4) for v in world_pos]}")
            return VisionResult(world_pos, conf, box, "GT_FALLBACK")
        print("[Vision] gt_body_id 미제공 — LOW_CONF로 비전 좌표 사용.")
        status = "LOW_CONF"
    else:
        status = "OK"

    world_pos = pixel_to_world(cx, cy, depth_m, view_mat)
    if world_pos is None:
        print("[Vision] DEPTH_INVALID: 유효한 깊이값을 읽을 수 없습니다.")
        return VisionResult(None, conf, box, "DEPTH_INVALID")

    print(f"[Vision] 월드 좌표: {[round(v,4) for v in world_pos]}")
    return VisionResult(world_pos, conf, box, status)