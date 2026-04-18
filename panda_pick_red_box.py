"""
Franka Emika Panda - 빨간 네모 박스 pick & place
PyBullet 시뮬레이션 + OWL-ViT 비전 모듈
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

# 박스 목표 위치
BOX_POS          = [-0.5,  0.2,  0.025]   # 바닥에 놓인 상태 (높이 0.05 → 중심 0.025)
BOX_SIZE         = [0.04, 0.04, 0.05]    # 반치수 (half-extents)

# 작업 높이
HOVER_HEIGHT     = 0.20   # 박스 중심 위 이 높이에서 하강 시작
GRASP_HEIGHT     = 0.005  # 박스 표면 위 이 높이까지 하강
LIFT_HEIGHT      = 0.35   # 들어올릴 최종 높이

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


def load_red_box():
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=BOX_SIZE)
    vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=BOX_SIZE,
        rgbaColor=[1, 0, 0, 1],   # 빨간색
    )
    box = p.createMultiBody(
        baseMass=0.3,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=BOX_POS,
    )
    return box


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


# ── 메인 시퀀스 ───────────────────────────────────────────────────────────────
def run():
    # ── PyBullet GUI 먼저 실행, 로봇 로드 ────────────────────────────────────
    init_sim()
    load_plane()
    robot = load_panda()
    box   = load_red_box()

    # 시뮬레이션 안정화 (로봇이 화면에 정지 상태로 표시됨)
    for _ in range(120):
        p.stepSimulation()

    # ── 자연어 명령 입력 및 파싱 ─────────────────────────────────────────────
    print("=" * 55)
    print(" Franka Panda NL2C — 자연어 명령을 입력하세요")
    print(" 예) 빨간 네모를 0.3 0.2 0.05 로 빠르게 옮겨줘")
    print("=" * 55)
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

    print(f"\n[명령 확인]")
    print(f"  객체   : {obj_query}")
    print(f"  목적지 : {dest}")
    print(f"  속도   : {speed}  (Kp={pid_user['kp']}, Kd={pid_user['kd']}, max_vel={pid_user['max_vel']})")
    print()

    # ── 스캔 자세로 이동 ──────────────────────────────────────────────────────
    log("Scan", "스캔 자세로 이동 중...")
    move_to_scan_pose(robot)

    # ── 360도 회전 스캔으로 박스 탐색 ────────────────────────────────────────
    log("Scan", f"360° 회전 스캔 시작 ('{obj_query}')")
    result = scan_360(robot, obj_query)

    if result is None:
        print("[오류] 360° 스캔 완료 후에도 빨간 박스를 찾지 못했습니다.")
        p.disconnect()
        return

    if result.status == "LOW_CONF":
        ans = input(f"  신뢰도 {result.confidence:.2f} — 계속 진행하시겠습니까? (y/n): ")
        if ans.strip().lower() != "y":
            p.disconnect()
            return

    # 비전 결과 사용, 신뢰도 낮으면 GT 폴백
    if result.confidence >= CONF_HIGH:
        bx, by, bz = result.world_pos
        log("Vision", f"OWL-ViT 좌표: [{bx:.3f}, {by:.3f}, {bz:.3f}]  (conf={result.confidence:.2f})")
    else:
        box_pos, _ = p.getBasePositionAndOrientation(box)
        bx, by, bz = box_pos
        log("Vision", f"GT 좌표 폴백: [{bx:.3f}, {by:.3f}, {bz:.3f}]")

    top_z = bz + BOX_SIZE[2]

    orn_down = p.getQuaternionFromEuler([math.pi, 0, 0])

    # ── Phase 1 : APPROACH (박스 위 호버) ──────────────────────────────────
    log("Phase 1", f"APPROACH → [{bx:.3f}, {by:.3f}, {top_z + HOVER_HEIGHT:.3f}]")
    hover_pos = [bx, by, top_z + HOVER_HEIGHT]
    joints    = solve_ik(robot, hover_pos, orn_down)
    set_gripper(robot, 0.04, steps=60)
    drive_joints(robot, joints, STEPS_APPROACH, pid=pid_user)

    # ── Phase 2 : GRIPPER OPEN CONFIRM ────────────────────────────────────
    log("Phase 2", "GRIPPER_OPEN_CONFIRM")
    w1 = p.getJointState(robot, FINGER_JOINT_1)[0]
    w2 = p.getJointState(robot, FINGER_JOINT_2)[0]
    if (w1 + w2) < 0.035:
        print("  ⚠ 그리퍼가 충분히 열리지 않았습니다. 강제 개방합니다.")
        set_gripper(robot, 0.04, steps=120)

    # ── Phase 3 : DESCEND (박스 표면까지 하강) ────────────────────────────
    log("Phase 3", f"DESCEND → [{bx:.3f}, {by:.3f}, {top_z + GRASP_HEIGHT:.3f}]  [CAREFUL 강제]")
    grasp_pos = [bx, by, top_z + GRASP_HEIGHT]
    joints    = solve_ik(robot, grasp_pos, orn_down)
    drive_joints(robot, joints, STEPS_DESCEND, pid=PID_CAREFUL)

    # ── Phase 4 : GRASP ───────────────────────────────────────────────────
    log("Phase 4", "GRASP → 그리퍼 폐쇄")
    set_gripper(robot, 0.01, steps=180)   # 박스 두께에 맞게 닫기
    # 고정 제약으로 안정적 파지 시뮬레이션
    constraint = p.createConstraint(
        robot, EE_LINK, box, -1,
        p.JOINT_FIXED, [0, 0, 0], [0, 0, 0.05], [0, 0, 0],
    )
    for _ in range(STEPS_HOLD):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)
    log("Phase 4", "파지 성공 ✅")

    # ── Phase 5 : LIFT ────────────────────────────────────────────────────
    log("Phase 5", f"LIFT → [{bx:.3f}, {by:.3f}, {LIFT_HEIGHT:.3f}]  [SLOW 최소]")
    lift_pos = [bx, by, LIFT_HEIGHT]
    joints   = solve_ik(robot, lift_pos, orn_down)
    drive_joints(robot, joints, STEPS_LIFT, pid=PID_SLOW)

    # ── Phase 6 : TRANSIT (목적지 위로 이동) ─────────────────────────────
    dest_x, dest_y, dest_z = dest[0], dest[1], dest[2]
    log("Phase 6", f"TRANSIT → [{dest_x:.3f}, {dest_y:.3f}, {LIFT_HEIGHT:.3f}]")
    transit_pos = [dest_x, dest_y, LIFT_HEIGHT]
    joints      = solve_ik(robot, transit_pos, orn_down)
    drive_joints(robot, joints, STEPS_TRANSIT, pid=pid_user)

    # ── Phase 7 : PLACE_DESCEND (박스 내려놓기) ──────────────────────────
    log("Phase 7", f"PLACE_DESCEND → [{dest_x:.3f}, {dest_y:.3f}, {dest_z:.3f}]  [CAREFUL 강제]")
    place_pos = [dest_x, dest_y, dest_z]
    joints    = solve_ik(robot, place_pos, orn_down)
    drive_joints(robot, joints, STEPS_PLACE_DESC, pid=PID_CAREFUL)

    # ── Phase 8 : RELEASE ────────────────────────────────────────────────
    log("Phase 8", "RELEASE → 제약 해제 + 그리퍼 개방")
    p.removeConstraint(constraint)
    set_gripper(robot, 0.04, steps=120)

    # 후퇴 (엔드이펙터 위로 올림)
    retract_pos = [dest_x, dest_y, LIFT_HEIGHT]
    joints      = solve_ik(robot, retract_pos, orn_down)
    drive_joints(robot, joints, 800)

    # 결과 확인
    box_final, _ = p.getBasePositionAndOrientation(box)
    log("완료", f"박스 최종 위치: {[round(v,3) for v in box_final]}")
    print(f"\n→ TASK_COMPLETE  ('{obj_query}'을(를) [{dest_x:.3f}, {dest_y:.3f}, {dest_z:.3f}]에 내려놓았습니다.)")

    # 화면 유지
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
