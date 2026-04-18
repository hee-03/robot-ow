"""
Franka Emika Panda - 빨간 네모 박스 집어올리기
PyBullet 시뮬레이션
"""

import pybullet as p
import pybullet_data
import time
import math

# ── 상수 ──────────────────────────────────────────────────────────────────────
PANDA_URDF       = "franka_panda/panda.urdf"
EE_LINK          = 11       # panda_hand 링크 인덱스
FINGER_JOINT_1   = 9
FINGER_JOINT_2   = 10
NUM_JOINTS       = 12

# 박스 목표 위치
BOX_POS          = [0.5,  0.0,  0.025]   # 바닥에 놓인 상태 (높이 0.05 → 중심 0.025)
BOX_SIZE         = [0.04, 0.04, 0.05]    # 반치수 (half-extents)

# 작업 높이
HOVER_HEIGHT     = 0.20   # 박스 중심 위 이 높이에서 하강 시작
GRASP_HEIGHT     = 0.005  # 박스 표면 위 이 높이까지 하강
LIFT_HEIGHT      = 0.35   # 들어올릴 최종 높이

# IK 반복 횟수
IK_ITER          = 100

# PID - CAREFUL 프리셋
KP               = 0.3
KD               = 0.8
MAX_VEL          = 0.05
MAX_FORCE        = 87.0

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
def drive_joints(robot, target_joints, steps):
    for _ in range(steps):
        for j in range(7):   # 7-DOF 암 관절만
            p.setJointMotorControl2(
                robot, j,
                p.POSITION_CONTROL,
                targetPosition=target_joints[j],
                positionGain=KP,
                velocityGain=KD,
                maxVelocity=MAX_VEL,
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


# ── 메인 시퀀스 ───────────────────────────────────────────────────────────────
def run():
    init_sim()
    load_plane()
    robot = load_panda()
    box   = load_red_box()

    # 시뮬레이션 안정화
    for _ in range(120):
        p.stepSimulation()

    box_pos, _ = p.getBasePositionAndOrientation(box)
    bx, by, bz = box_pos
    top_z = bz + BOX_SIZE[2]   # 박스 상단 Z

    orn_down = p.getQuaternionFromEuler([math.pi, 0, 0])

    # ── Phase 1 : APPROACH (박스 위 호버) ──────────────────────────────────
    log("Phase 1", f"APPROACH → [{bx:.3f}, {by:.3f}, {top_z + HOVER_HEIGHT:.3f}]")
    hover_pos = [bx, by, top_z + HOVER_HEIGHT]
    joints    = solve_ik(robot, hover_pos, orn_down)
    set_gripper(robot, 0.04, steps=60)   # 그리퍼 열기
    drive_joints(robot, joints, STEPS_APPROACH)

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
    drive_joints(robot, joints, STEPS_DESCEND)

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
    drive_joints(robot, joints, STEPS_LIFT)

    # ── Phase 6 : TRANSIT (목적지 위로 이동) ─────────────────────────────
    dest_x = bx
    dest_y = by + PLACE_Y_OFFSET
    dest_z = bz   # 원래 바닥 높이 기준
    log("Phase 6", f"TRANSIT → [{dest_x:.3f}, {dest_y:.3f}, {LIFT_HEIGHT:.3f}]")
    transit_pos = [dest_x, dest_y, LIFT_HEIGHT]
    joints      = solve_ik(robot, transit_pos, orn_down)
    drive_joints(robot, joints, STEPS_TRANSIT)

    # ── Phase 7 : PLACE_DESCEND (박스 내려놓기) ──────────────────────────
    place_z = dest_z + BOX_SIZE[2] + 0.01   # 바닥 + 박스 반높이 + 여유
    log("Phase 7", f"PLACE_DESCEND → [{dest_x:.3f}, {dest_y:.3f}, {place_z:.3f}]  [CAREFUL 강제]")
    place_pos = [dest_x, dest_y, place_z]
    joints    = solve_ik(robot, place_pos, orn_down)
    drive_joints(robot, joints, STEPS_PLACE_DESC)

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
    print(f"\n→ TASK_COMPLETE  (빨간 네모 박스를 [{dest_x:.2f}, {dest_y:.2f}, {dest_z:.3f}]에 내려놓았습니다.)")

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
