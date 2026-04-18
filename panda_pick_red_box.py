"""
Franka Emika Panda - 다중 물체 pick & place
PyBullet 시뮬레이션 + OWL-ViT 비전 + NL 파서 + 장애물 회피
"""

import pybullet as p
import pybullet_data
import time
import math
from vision import find_object, CONF_HIGH
from nl_parser import parse_command

# ── 상수 ──────────────────────────────────────────────────────────────────────
PANDA_URDF       = "franka_panda/panda.urdf"
EE_LINK          = 11       # panda_hand 링크 인덱스
FINGER_JOINT_1   = 9
FINGER_JOINT_2   = 10
NUM_JOINTS       = 12

# 작업 높이
HOVER_HEIGHT     = 0.20   # 물체 상단 위 호버 높이
GRASP_HEIGHT     = 0.005  # 물체 상단 위 파지 높이
LIFT_HEIGHT      = 0.35   # 들어올릴 높이

# 장애물 회피 거리
OBSTACLE_THRESH  = 0.12   # 목적지 반경 (m) 이내면 장애물로 판정
OBSTACLE_OFFSET  = 0.15   # 장애물을 x, y 각각 이만큼 이동

# ── 씬에 배치할 물체 정의 ─────────────────────────────────────────────────────
OBJECTS_DEF = [
    {
        "name":   "red box",
        "pos":    [-0.5, 0.2, 0.025],
        "half_z": 0.05,
        "shape":  "box",
        "half":   [0.04, 0.04, 0.05],
        "color":  [1, 0, 0, 1],
        "mass":   0.3,
    },
    {
        "name":   "blue cylinder",
        "pos":    [0.4, -0.2, 0.05],
        "half_z": 0.05,
        "shape":  "cylinder",
        "radius": 0.04,
        "h_ext":  0.05,
        "color":  [0, 0.3, 1, 1],
        "mass":   0.2,
    },
    {
        "name":   "green sphere",
        "pos":    [0.3, 0.45, 0.04],
        "half_z": 0.04,
        "shape":  "sphere",
        "radius": 0.04,
        "color":  [0, 0.8, 0, 1],
        "mass":   0.15,
    },
]

# IK 반복 횟수
IK_ITER          = 100

MAX_FORCE        = 87.0

# PID 프리셋 테이블
PID_PRESETS = {
    "careful": dict(kp=0.3,  kd=0.8,  max_vel=0.05),
    "slow":    dict(kp=0.5,  kd=0.5,  max_vel=0.10),
    "normal":  dict(kp=0.8,  kd=0.3,  max_vel=0.20),
    "fast":    dict(kp=1.5,  kd=0.1,  max_vel=0.40),
}
# 강제 적용 프리셋 (페이즈별)
PID_CAREFUL = PID_PRESETS["careful"]
PID_SLOW    = PID_PRESETS["slow"]

# 각 페이즈 스텝 수
STEPS_APPROACH   = 1500
STEPS_DESCEND    = 1000
STEPS_LIFT       = 1200
STEPS_TRANSIT    = 1500
STEPS_PLACE_DESC = 1000
STEPS_HOLD       = 60

PLACE_Y_OFFSET   = 0.30   # 목적지 Y 오프셋 (30cm)


# ── 초기화 ────────────────────────────────────────────────────────────────────
def init_sim():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(0)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.4,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0.4, 0, 0.2],
    )


def load_plane():
    p.loadURDF("plane.urdf")


def load_panda():
    robot = p.loadURDF(
        PANDA_URDF,
        basePosition=[0, 0, 0],
        useFixedBase=True,
    )
    # 홈 자세 (관절각 라디안)
    home = [0, -math.pi/4, 0, -3*math.pi/4, 0, math.pi/2, math.pi/4, 0.04, 0.04, 0, 0, 0]
    for i in range(NUM_JOINTS):
        p.resetJointState(robot, i, home[i])
    return robot


def load_objects() -> dict:
    """
    OBJECTS_DEF에 정의된 물체를 모두 로드.
    반환: {name: {"body_id": int, "half_z": float, "pos": [x,y,z]}}
    """
    registry = {}
    for obj in OBJECTS_DEF:
        name  = obj["name"]
        pos   = obj["pos"]
        color = obj["color"]
        mass  = obj["mass"]

        if obj["shape"] == "box":
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=obj["half"])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=obj["half"], rgbaColor=color)
        elif obj["shape"] == "cylinder":
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=obj["radius"], height=obj["h_ext"] * 2)
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=obj["radius"], length=obj["h_ext"] * 2, rgbaColor=color)
        elif obj["shape"] == "sphere":
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=obj["radius"])
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=obj["radius"], rgbaColor=color)
        else:
            continue

        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos,
        )
        registry[name] = {"body_id": body_id, "half_z": obj["half_z"], "pos": list(pos)}
        print(f"[Load] '{name}' → body_id={body_id}  pos={pos}")

    return registry


# ── IK ───────────────────────────────────────────────────────────────────────
def solve_ik(robot, target_pos, target_orn=None):
    if target_orn is None:
        # 엔드이펙터를 아래로 향하게 (쿼터니언)
        target_orn = p.getQuaternionFromEuler([math.pi, 0, 0])
    joints = p.calculateInverseKinematics(
        robot, EE_LINK, target_pos, target_orn,
        maxNumIterations=IK_ITER,
        residualThreshold=1e-5,
    )
    return joints


# ── 관절 구동 ─────────────────────────────────────────────────────────────────
def drive_joints(robot, target_joints, steps, pid=None):
    if pid is None:
        pid = PID_PRESETS["normal"]
    for _ in range(steps):
        for j in range(7):   # 7-DOF 암 관절만
            p.setJointMotorControl2(
                robot, j,
                p.POSITION_CONTROL,
                targetPosition=target_joints[j],
                positionGain=pid["kp"],
                velocityGain=pid["kd"],
                maxVelocity=pid["max_vel"],
                force=MAX_FORCE,
            )
        p.stepSimulation()
        time.sleep(1.0 / 240.0)


def set_gripper(robot, width, steps=120):
    for _ in range(steps):
        p.setJointMotorControl2(robot, FINGER_JOINT_1, p.POSITION_CONTROL,
                                targetPosition=width, force=40)
        p.setJointMotorControl2(robot, FINGER_JOINT_2, p.POSITION_CONTROL,
                                targetPosition=width, force=40)
        p.stepSimulation()
        time.sleep(1.0 / 240.0)


# ── 페이즈 출력 ───────────────────────────────────────────────────────────────
def log(phase, msg=""):
    print(f"[{phase}] {msg}")


# ── 360도 스캔 ────────────────────────────────────────────────────────────────
SCAN_POSE = [0, -0.4, 0, -2.0, 0, 1.8, math.pi/4, 0.04, 0.04, 0, 0, 0]
SCAN_STEP_DEG  = 20    # 몇 도마다 OWL-ViT 탐색할지
SCAN_SIM_STEPS = 80    # 각 회전 구간 시뮬레이션 스텝 수


def move_to_scan_pose(robot):
    """스캔에 유리한 자세로 이동 (팔 뻗고 약간 아래를 향함)"""
    for _ in range(600):
        for j in range(NUM_JOINTS):
            p.setJointMotorControl2(robot, j, p.POSITION_CONTROL,
                                    targetPosition=SCAN_POSE[j],
                                    positionGain=0.5, velocityGain=0.5,
                                    maxVelocity=0.3, force=MAX_FORCE)
        p.stepSimulation()
        time.sleep(1.0 / 240.0)


def scan_360(robot, text_query):
    """
    Joint 0을 360도 회전하며 SCAN_STEP_DEG마다 OWL-ViT 탐색.
    처음 신뢰도 OK 결과를 반환, 못 찾으면 None.
    """
    start_angle = p.getJointState(robot, 0)[0]
    total_steps = int(360 / SCAN_STEP_DEG)

    for i in range(total_steps):
        target_angle = start_angle + math.radians(SCAN_STEP_DEG * (i + 1))
        log("Scan", f"{SCAN_STEP_DEG * (i+1)}° 회전 중...")

        # Joint 0만 목표 각도로 구동
        for _ in range(SCAN_SIM_STEPS):
            p.setJointMotorControl2(robot, 0, p.POSITION_CONTROL,
                                    targetPosition=target_angle,
                                    positionGain=0.8, velocityGain=0.5,
                                    maxVelocity=0.5, force=MAX_FORCE)
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

        result = find_object(robot, EE_LINK, text_query)
        if result.status in ("OK", "LOW_CONF"):
            log("Scan", f"객체 발견! conf={result.confidence:.2f}  pos={[round(v,3) for v in result.world_pos]}")
            return result

    return None


# ── 공통 pick & place ────────────────────────────────────────────────────────
def pick_and_place(robot, body_id, pick_pos, half_z, place_dest, pid, label="물체"):
    """
    단일 물체를 pick_pos에서 집어 place_dest로 이동.
    pick_pos : 물체 중심 [x, y, z]
    half_z   : 물체 z 방향 반높이 (top_z = center_z + half_z)
    place_dest: [x, y, z]
    """
    orn_down = p.getQuaternionFromEuler([math.pi, 0, 0])
    bx, by, bz = pick_pos
    top_z = bz + half_z

    log(f"{label} Ph1", f"APPROACH  [{bx:.3f}, {by:.3f}, {top_z + HOVER_HEIGHT:.3f}]")
    joints = solve_ik(robot, [bx, by, top_z + HOVER_HEIGHT], orn_down)
    set_gripper(robot, 0.04, steps=60)
    drive_joints(robot, joints, STEPS_APPROACH, pid=pid)

    w1 = p.getJointState(robot, FINGER_JOINT_1)[0]
    w2 = p.getJointState(robot, FINGER_JOINT_2)[0]
    if (w1 + w2) < 0.035:
        set_gripper(robot, 0.04, steps=120)

    log(f"{label} Ph3", f"DESCEND   [{bx:.3f}, {by:.3f}, {top_z + GRASP_HEIGHT:.3f}]  [CAREFUL]")
    joints = solve_ik(robot, [bx, by, top_z + GRASP_HEIGHT], orn_down)
    drive_joints(robot, joints, STEPS_DESCEND, pid=PID_CAREFUL)

    log(f"{label} Ph4", "GRASP")
    set_gripper(robot, 0.01, steps=180)
    constraint = p.createConstraint(
        robot, EE_LINK, body_id, -1,
        p.JOINT_FIXED, [0, 0, 0], [0, 0, 0.05], [0, 0, 0],
    )
    for _ in range(STEPS_HOLD):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    log(f"{label} Ph5", f"LIFT      [{bx:.3f}, {by:.3f}, {LIFT_HEIGHT:.3f}]  [SLOW]")
    joints = solve_ik(robot, [bx, by, LIFT_HEIGHT], orn_down)
    drive_joints(robot, joints, STEPS_LIFT, pid=PID_SLOW)

    dx, dy, dz = place_dest
    log(f"{label} Ph6", f"TRANSIT   [{dx:.3f}, {dy:.3f}, {LIFT_HEIGHT:.3f}]")
    joints = solve_ik(robot, [dx, dy, LIFT_HEIGHT], orn_down)
    drive_joints(robot, joints, STEPS_TRANSIT, pid=pid)

    # dz = 목적지 바닥 높이, EE는 물체 상단 + GRASP_HEIGHT 위치까지 내려야 바닥에 닿음
    place_ee_z = dz + half_z + GRASP_HEIGHT
    log(f"{label} Ph7", f"PLACE     [{dx:.3f}, {dy:.3f}, {place_ee_z:.3f}]  [CAREFUL]")
    joints = solve_ik(robot, [dx, dy, place_ee_z], orn_down)
    drive_joints(robot, joints, STEPS_PLACE_DESC, pid=PID_CAREFUL)

    log(f"{label} Ph8", "RELEASE")
    p.removeConstraint(constraint)
    set_gripper(robot, 0.04, steps=120)
    joints = solve_ik(robot, [dx, dy, LIFT_HEIGHT], orn_down)
    drive_joints(robot, joints, 800)


# ── 장애물 감지 ───────────────────────────────────────────────────────────────
def find_obstacle(dest, objects, target_name):
    """
    dest [x,y,z] 주변 OBSTACLE_THRESH 이내에 target 외 다른 물체가 있으면
    (name, info_dict) 반환, 없으면 None.
    """
    dx, dy = dest[0], dest[1]
    for name, info in objects.items():
        if name == target_name:
            continue
        ox, oy, _ = p.getBasePositionAndOrientation(info["body_id"])[0]
        dist = math.sqrt((ox - dx) ** 2 + (oy - dy) ** 2)
        if dist < OBSTACLE_THRESH:
            log("Obstacle", f"'{name}'이(가) 목적지 근처에 있습니다 (dist={dist:.3f}m)")
            return name, info
    return None


# ── 메인 시퀀스 ───────────────────────────────────────────────────────────────
def run():
    # ── PyBullet GUI 먼저 실행, 로봇 + 물체 로드 ─────────────────────────────
    init_sim()
    load_plane()
    robot   = load_panda()
    objects = load_objects()   # {name: {body_id, half_z, pos}}

    for _ in range(120):
        p.stepSimulation()

    # ── 자연어 명령 입력 및 파싱 ─────────────────────────────────────────────
    print("=" * 60)
    print(" Franka Panda NL2C — 자연어 명령을 입력하세요")
    print(" 예) 빨간 네모를 0.3 0.2 0.025 로 빠르게 옮겨줘")
    print(" 물체 목록:", ", ".join(objects.keys()))
    print("=" * 60)
    command = input(">> ").strip()

    try:
        parsed = parse_command(command)
    except Exception as e:
        print(f"[오류] 명령 파싱 실패: {e}")
        p.disconnect()
        return

    obj_query = parsed["object"]
    dest      = parsed["destination"]
    speed     = parsed["speed_level"]
    pid_user  = PID_PRESETS[speed]

    print(f"\n[명령 확인]  객체={obj_query}  목적지={dest}  속도={speed}")

    # ── 스캔 자세 → 360° 회전 탐색 ───────────────────────────────────────────
    log("Scan", "스캔 자세로 이동 중...")
    move_to_scan_pose(robot)
    log("Scan", f"360° 회전 스캔 시작 ('{obj_query}')")
    result = scan_360(robot, obj_query)

    if result is None:
        print(f"[오류] '{obj_query}'을(를) 찾지 못했습니다.")
        p.disconnect()
        return

    if result.status == "LOW_CONF":
        ans = input(f"  신뢰도 {result.confidence:.2f} — 계속 진행하시겠습니까? (y/n): ")
        if ans.strip().lower() != "y":
            p.disconnect()
            return

    # 비전 좌표 or GT 폴백
    if result.confidence >= CONF_HIGH:
        pick_pos = list(result.world_pos)
        log("Vision", f"OWL-ViT 좌표: {[round(v,3) for v in pick_pos]}  (conf={result.confidence:.2f})")
    else:
        target_info = objects.get(obj_query)
        if target_info:
            pick_pos = list(p.getBasePositionAndOrientation(target_info["body_id"])[0])
        else:
            pick_pos = list(result.world_pos)
        log("Vision", f"GT 폴백: {[round(v,3) for v in pick_pos]}")

    # target half_z (OWL-ViT 결과면 objects에서 찾고, 없으면 기본값)
    target_info = objects.get(obj_query, {})
    half_z = target_info.get("half_z", 0.05)
    body_id = target_info.get("body_id")

    # body_id 미확인 시 가장 가까운 물체로 추정
    if body_id is None:
        min_dist, body_id, half_z = float("inf"), None, 0.05
        for info in objects.values():
            pos, _ = p.getBasePositionAndOrientation(info["body_id"])
            d = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos, pick_pos)))
            if d < min_dist:
                min_dist, body_id, half_z = d, info["body_id"], info["half_z"]

    # ── 장애물 감지 및 회피 ───────────────────────────────────────────────────
    obstacle = find_obstacle(dest, objects, obj_query)
    if obstacle:
        obs_name, obs_info = obstacle
        obs_pos = list(p.getBasePositionAndOrientation(obs_info["body_id"])[0])
        obs_dest = [obs_pos[0] + OBSTACLE_OFFSET, obs_pos[1] + OBSTACLE_OFFSET, obs_pos[2]]
        log("Obstacle", f"'{obs_name}' → [{obs_dest[0]:.3f}, {obs_dest[1]:.3f}]로 회피 이동")
        pick_and_place(robot, obs_info["body_id"], obs_pos, obs_info["half_z"],
                       obs_dest, PID_PRESETS["normal"], label=obs_name)

    # ── 목표 물체 pick & place ────────────────────────────────────────────────
    log("Task", f"'{obj_query}' pick & place 시작")
    pick_and_place(robot, body_id, pick_pos, half_z, dest, pid_user, label=obj_query)

    final_pos, _ = p.getBasePositionAndOrientation(body_id)
    log("완료", f"최종 위치: {[round(v,3) for v in final_pos]}")
    print(f"\n→ TASK_COMPLETE  ('{obj_query}'을(를) {dest}에 내려놓았습니다.)")

    print("시뮬레이션 종료하려면 Ctrl+C를 누르세요.")
    try:
        while True:
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
    except KeyboardInterrupt:
        pass
    finally:
        p.disconnect()


if __name__ == "__main__":
    run()
