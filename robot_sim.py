"""
Franka Panda 시뮬레이션 백그라운드 스레드 클래스
Streamlit 연동용 - PyBullet DIRECT 모드로 실행
"""

import math
import queue
import threading
import time

import numpy as np
import pybullet as p
import pybullet_data

# ── 상수 (panda_pick_red_box.py와 동일) ─────────────────────────────────────
PANDA_URDF     = "franka_panda/panda.urdf"
EE_LINK        = 11
FINGER_JOINT_1 = 9
FINGER_JOINT_2 = 10
NUM_JOINTS     = 12

HOVER_HEIGHT    = 0.20
GRASP_HEIGHT    = 0.005
LIFT_HEIGHT     = 0.35
OBSTACLE_THRESH = 0.12
OBSTACLE_OFFSET = 0.15

OBJECTS_DEF = [
    {
        "name":  "red box",
        "pos":   [-0.5, 0.2, 0.025],
        "half_z": 0.05,
        "shape": "box",
        "half":  [0.04, 0.04, 0.05],
        "color": [1, 0, 0, 1],
        "mass":  0.3,
    },
    {
        "name":  "blue cylinder",
        "pos":   [0.4, -0.4, 0.05],
        "half_z": 0.05,
        "shape": "cylinder",
        "radius": 0.04,
        "h_ext": 0.05,
        "color": [0, 0.3, 1, 1],
        "mass":  0.2,
    },
    {
        "name":  "green sphere",
        "pos":   [0.3, 0.45, 0.04],
        "half_z": 0.04,
        "shape": "sphere",
        "radius": 0.04,
        "color": [0, 0.8, 0, 1],
        "mass":  0.15,
    },
]

IK_ITER   = 100
MAX_FORCE = 87.0

PID_PRESETS = {
    "careful": dict(kp=0.3, kd=0.8, max_vel=0.05),
    "slow":    dict(kp=0.5, kd=0.5, max_vel=0.10),
    "normal":  dict(kp=0.8, kd=0.3, max_vel=0.20),
    "fast":    dict(kp=1.5, kd=0.1, max_vel=0.40),
}
PID_CAREFUL = PID_PRESETS["careful"]
PID_SLOW    = PID_PRESETS["slow"]

STEPS_APPROACH   = 1500
STEPS_DESCEND    = 1000
STEPS_LIFT       = 1200
STEPS_TRANSIT    = 1500
STEPS_PLACE_DESC = 1000
STEPS_HOLD       = 60

SCAN_POSE       = [0, -0.4, 0, -2.0, 0, 1.8, math.pi / 4, 0.04, 0.04, 0, 0, 0]
SCAN_STEP_DEG   = 10
SCAN_SIM_STEPS  = 200
JOINT_LIMIT_DEG = 160

# 오버뷰 카메라 설정
CAM_W  = 640
CAM_H  = 480
CAM_FOV = 60.0

# N 시뮬레이션 스텝마다 프레임 캡처 (240Hz / 12 ≈ 20fps)
FRAME_INTERVAL = 12


class RobotSimulation:
    """PyBullet DIRECT 모드 시뮬레이션을 백그라운드 스레드에서 실행하는 클래스"""

    def __init__(self):
        self._lock      = threading.Lock()
        self._frame     = None          # 최신 RGB 프레임 (numpy uint8)
        self._ee_frame  = None          # EE 카메라 프레임
        self._cmd_queue = queue.Queue()
        self._logs      = []
        self._status    = "초기화 전"
        self._running   = False
        self._thread    = None
        self._robot     = None
        self._objects   = {}
        self._step_cnt  = 0
        self._busy      = False         # 명령 실행 중 플래그

    # ── 외부 인터페이스 ───────────────────────────────────────────────────────

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._main_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def is_running(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()

    def is_busy(self) -> bool:
        return self._busy

    def send_command(self, cmd: str):
        self._cmd_queue.put(cmd)

    def get_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def get_ee_frame(self):
        with self._lock:
            return self._ee_frame.copy() if self._ee_frame is not None else None

    def get_logs(self) -> list:
        with self._lock:
            return list(self._logs)

    def get_status(self) -> str:
        with self._lock:
            return self._status

    def get_object_positions(self) -> dict:
        """현재 물체 위치 반환 (스레드 안전)"""
        if not self._robot:
            return {}
        try:
            result = {}
            for name, info in self._objects.items():
                pos, _ = p.getBasePositionAndOrientation(info["body_id"])
                result[name] = [round(v, 3) for v in pos]
            return result
        except Exception:
            return {}

    # ── 내부 유틸 ─────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        ts    = time.strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}"
        print(entry)
        with self._lock:
            self._logs.append(entry)
            if len(self._logs) > 60:
                self._logs.pop(0)

    def _set_status(self, status: str):
        with self._lock:
            self._status = status

    def _capture_overview(self):
        """고정 오버뷰 카메라 프레임 캡처"""
        view = p.computeViewMatrix(
            cameraEyePosition=[1.4, 1.0, 1.1],
            cameraTargetPosition=[0.0, 0.0, 0.1],
            cameraUpVector=[0, 0, 1],
        )
        proj = p.computeProjectionMatrixFOV(CAM_FOV, CAM_W / CAM_H, 0.1, 10.0)
        _, _, rgba, _, _ = p.getCameraImage(
            CAM_W, CAM_H,
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=p.ER_TINY_RENDERER,
        )
        rgb = np.array(rgba, dtype=np.uint8).reshape(CAM_H, CAM_W, 4)[:, :, :3]
        with self._lock:
            self._frame = rgb

    def _capture_ee(self):
        """EE 카메라 프레임 캡처 (vision.py 로직 재사용)"""
        if self._robot is None:
            return
        try:
            from vision import capture_ee_camera
            rgb, _, _ = capture_ee_camera(self._robot, EE_LINK)
            with self._lock:
                self._ee_frame = rgb
        except Exception:
            pass

    def _step_sim(self):
        """1 시뮬레이션 스텝 실행 + 주기적 프레임 캡처"""
        p.stepSimulation()
        self._step_cnt += 1
        if self._step_cnt % FRAME_INTERVAL == 0:
            self._capture_overview()
        if self._step_cnt % (FRAME_INTERVAL * 4) == 0:
            self._capture_ee()
        time.sleep(1.0 / 240.0)

    # ── 로봇 제어 (원본과 동일하되 _step_sim 사용) ────────────────────────────

    def _solve_ik(self, target_pos, target_orn=None):
        if target_orn is None:
            target_orn = p.getQuaternionFromEuler([math.pi, 0, 0])
        return p.calculateInverseKinematics(
            self._robot, EE_LINK, target_pos, target_orn,
            maxNumIterations=IK_ITER,
            residualThreshold=1e-5,
        )

    def _drive_joints(self, target_joints, steps, pid=None):
        if pid is None:
            pid = PID_PRESETS["normal"]
        for _ in range(steps):
            for j in range(7):
                p.setJointMotorControl2(
                    self._robot, j,
                    p.POSITION_CONTROL,
                    targetPosition=target_joints[j],
                    positionGain=pid["kp"],
                    velocityGain=pid["kd"],
                    maxVelocity=pid["max_vel"],
                    force=MAX_FORCE,
                )
            self._step_sim()

    def _set_gripper(self, width, steps=120):
        for _ in range(steps):
            p.setJointMotorControl2(self._robot, FINGER_JOINT_1, p.POSITION_CONTROL,
                                    targetPosition=width, force=40)
            p.setJointMotorControl2(self._robot, FINGER_JOINT_2, p.POSITION_CONTROL,
                                    targetPosition=width, force=40)
            self._step_sim()

    def _move_to_scan_pose(self):
        for _ in range(600):
            for j in range(NUM_JOINTS):
                p.setJointMotorControl2(self._robot, j, p.POSITION_CONTROL,
                                        targetPosition=SCAN_POSE[j],
                                        positionGain=0.5, velocityGain=0.5,
                                        maxVelocity=0.3, force=MAX_FORCE)
            self._step_sim()

    def _drive_joint0(self, target_rad, steps=SCAN_SIM_STEPS):
        for _ in range(steps):
            p.setJointMotorControl2(self._robot, 0, p.POSITION_CONTROL,
                                    targetPosition=target_rad,
                                    positionGain=0.8, velocityGain=0.5,
                                    maxVelocity=0.5, force=MAX_FORCE)
            self._step_sim()

    def _drive_joint0_until(self, target_rad, tol=0.01, max_steps=1000):
        for _ in range(max_steps):
            current = p.getJointState(self._robot, 0)[0]
            if abs(current - target_rad) < tol:
                break
            p.setJointMotorControl2(self._robot, 0, p.POSITION_CONTROL,
                                    targetPosition=target_rad,
                                    positionGain=0.8, velocityGain=0.5,
                                    maxVelocity=0.8, force=MAX_FORCE)
            self._step_sim()

    def _scan_full_range(self, object_names: list) -> dict:
        from vision import capture_ee_camera, detect_multiple, pixel_to_world, CONF_LOW

        detected = {n: None for n in object_names}
        reported = set()

        phases = [
            ("반시계(CCW)",
             [math.radians(d) for d in range(SCAN_STEP_DEG, JOINT_LIMIT_DEG + 1, SCAN_STEP_DEG)]),
            ("시계(CW)",
             [math.radians(-d) for d in range(SCAN_STEP_DEG, JOINT_LIMIT_DEG + 1, SCAN_STEP_DEG)]),
        ]

        for phase_label, angles in phases:
            self._log(f"스캔 {phase_label} 시작")
            self._set_status(f"스캔 중: {phase_label}")

            for target_rad in angles:
                self._drive_joint0(target_rad)
                for _ in range(2):
                    rgb, depth_m, view_mat = capture_ee_camera(self._robot, EE_LINK)
                    with self._lock:
                        self._ee_frame = rgb.copy()

                    results = detect_multiple(rgb, object_names)
                    for name, det in results.items():
                        if det is None:
                            continue
                        cx, cy, score, _ = det
                        if score < CONF_LOW:
                            continue
                        world_pos = pixel_to_world(cx, cy, depth_m, view_mat)
                        if world_pos is None:
                            continue
                        if detected[name] is None or score > detected[name][0]:
                            detected[name] = (score, world_pos)
                        if name not in reported:
                            reported.add(name)
                            pos = [round(float(v), 3) for v in world_pos]
                            self._log(f"감지: '{name}' conf={score:.2f} 좌표={pos}")

            self._log(f"스캔 {phase_label} 완료, 원위치 복귀")
            self._drive_joint0_until(0.0)

        found = [n for n, v in detected.items() if v is not None]
        self._log(f"스캔 완료 — {len(found)}개 발견: {found}")
        return detected

    def _pick_and_place(self, body_id, pick_pos, half_z, place_dest, pid, label="물체"):
        orn_down = p.getQuaternionFromEuler([math.pi, 0, 0])
        bx, by, bz = pick_pos
        top_z = bz + half_z

        self._log(f"[{label}] Ph1 APPROACH")
        self._set_status(f"[{label}] 접근 중...")
        joints = self._solve_ik([bx, by, top_z + HOVER_HEIGHT], orn_down)
        self._set_gripper(0.04, steps=60)
        self._drive_joints(joints, STEPS_APPROACH, pid=pid)

        w1 = p.getJointState(self._robot, FINGER_JOINT_1)[0]
        w2 = p.getJointState(self._robot, FINGER_JOINT_2)[0]
        if (w1 + w2) < 0.035:
            self._set_gripper(0.04, steps=120)

        self._log(f"[{label}] Ph3 DESCEND [CAREFUL]")
        self._set_status(f"[{label}] 하강 중...")
        joints = self._solve_ik([bx, by, top_z + GRASP_HEIGHT], orn_down)
        self._drive_joints(joints, STEPS_DESCEND, pid=PID_CAREFUL)

        self._log(f"[{label}] Ph4 GRASP")
        self._set_status(f"[{label}] 파지 중...")
        self._set_gripper(0.01, steps=180)
        constraint = p.createConstraint(
            self._robot, EE_LINK, body_id, -1,
            p.JOINT_FIXED, [0, 0, 0], [0, 0, 0.05], [0, 0, 0],
        )
        for _ in range(STEPS_HOLD):
            self._step_sim()

        self._log(f"[{label}] Ph5 LIFT [SLOW]")
        self._set_status(f"[{label}] 들어올리는 중...")
        joints = self._solve_ik([bx, by, LIFT_HEIGHT], orn_down)
        self._drive_joints(joints, STEPS_LIFT, pid=PID_SLOW)

        dx, dy, dz = place_dest
        self._log(f"[{label}] Ph6 TRANSIT → [{dx:.2f}, {dy:.2f}]")
        self._set_status(f"[{label}] 이동 중...")
        joints = self._solve_ik([dx, dy, LIFT_HEIGHT], orn_down)
        self._drive_joints(joints, STEPS_TRANSIT, pid=pid)

        place_ee_z = dz + half_z + GRASP_HEIGHT
        self._log(f"[{label}] Ph7 PLACE [CAREFUL]")
        self._set_status(f"[{label}] 내려놓는 중...")
        joints = self._solve_ik([dx, dy, place_ee_z], orn_down)
        self._drive_joints(joints, STEPS_PLACE_DESC, pid=PID_CAREFUL)

        self._log(f"[{label}] Ph8 RELEASE")
        p.removeConstraint(constraint)
        self._set_gripper(0.04, steps=120)
        joints = self._solve_ik([dx, dy, LIFT_HEIGHT], orn_down)
        self._drive_joints(joints, 800)

        self._log(f"[{label}] 작업 완료!")

    def _find_obstacle(self, dest, target_name):
        dx, dy = dest[0], dest[1]
        for name, info in self._objects.items():
            if name == target_name:
                continue
            ox, oy, _ = p.getBasePositionAndOrientation(info["body_id"])[0]
            dist = math.sqrt((ox - dx) ** 2 + (oy - dy) ** 2)
            if dist < OBSTACLE_THRESH:
                self._log(f"장애물: '{name}' 목적지 근처 (dist={dist:.3f}m)")
                return name, info
        return None

    # ── 명령 실행 ─────────────────────────────────────────────────────────────

    def _execute_nl_command(self, command: str):
        from nl_parser import parse_command
        from vision import capture_ee_camera, detect_multiple, CONF_HIGH

        self._busy = True
        try:
            self._log(f"명령 수신: {command}")
            self._set_status("명령 파싱 중...")

            try:
                parsed = parse_command(command)
            except Exception as e:
                self._log(f"파싱 오류: {e}")
                self._set_status("파싱 실패 — 명령을 다시 입력하세요")
                return

            obj_query = parsed["object"]
            dest      = parsed["destination"]
            speed     = parsed["speed_level"]
            pid_user  = PID_PRESETS.get(speed, PID_PRESETS["normal"])
            self._log(f"파싱 결과: object='{obj_query}'  dest={dest}  speed={speed}")

            # 비전 워밍업
            self._set_status("비전 모델 준비 중...")
            dummy_rgb = capture_ee_camera(self._robot, EE_LINK)[0]
            detect_multiple(dummy_rgb, ["red box"])

            # 스캔
            self._set_status("스캔 자세로 이동 중...")
            self._move_to_scan_pose()
            scan_results = self._scan_full_range(list(self._objects.keys()))

            scan_hit = scan_results.get(obj_query)
            if scan_hit is None:
                self._log(f"오류: '{obj_query}'을(를) 찾지 못했습니다")
                self._set_status(f"'{obj_query}' 미발견")
                return

            score, world_pos = scan_hit
            if score >= CONF_HIGH:
                pick_pos = [float(v) for v in world_pos]
                self._log(f"OWL-ViT 좌표: {[round(v,3) for v in pick_pos]}  (conf={score:.2f})")
            else:
                target_info = self._objects.get(obj_query)
                if target_info:
                    pick_pos = list(p.getBasePositionAndOrientation(target_info["body_id"])[0])
                else:
                    pick_pos = [float(v) for v in world_pos]
                self._log(f"GT 폴백 좌표: {[round(v,3) for v in pick_pos]}")

            target_info = self._objects.get(obj_query, {})
            half_z  = target_info.get("half_z", 0.05)
            body_id = target_info.get("body_id")

            if body_id is None:
                min_dist = float("inf")
                for info in self._objects.values():
                    pos, _ = p.getBasePositionAndOrientation(info["body_id"])
                    d = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos, pick_pos)))
                    if d < min_dist:
                        min_dist, body_id, half_z = d, info["body_id"], info["half_z"]

            # 장애물 회피
            obstacle = self._find_obstacle(dest, obj_query)
            if obstacle:
                obs_name, obs_info = obstacle
                obs_pos  = list(p.getBasePositionAndOrientation(obs_info["body_id"])[0])
                obs_dest = [obs_pos[0] + OBSTACLE_OFFSET, obs_pos[1] + OBSTACLE_OFFSET, obs_pos[2]]
                self._log(f"장애물 '{obs_name}' 먼저 이동")
                self._pick_and_place(obs_info["body_id"], obs_pos, obs_info["half_z"],
                                     obs_dest, PID_PRESETS["normal"], label=obs_name)

            # 메인 pick & place
            self._log(f"'{obj_query}' pick & place 시작")
            self._pick_and_place(body_id, pick_pos, half_z, dest, pid_user, label=obj_query)

            final_pos, _ = p.getBasePositionAndOrientation(body_id)
            self._log(f"완료! 최종 위치: {[round(v,3) for v in final_pos]}")
            self._set_status("작업 완료!")

        except Exception as e:
            self._log(f"실행 오류: {e}")
            self._set_status(f"오류 발생: {e}")
        finally:
            self._busy = False

    # ── 메인 루프 ─────────────────────────────────────────────────────────────

    def _main_loop(self):
        try:
            p.connect(p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)
            p.setRealTimeSimulation(0)

            p.loadURDF("plane.urdf")

            robot = p.loadURDF(PANDA_URDF, basePosition=[0, 0, 0], useFixedBase=True)
            self._robot = robot
            home = [0, -math.pi / 4, 0, -3 * math.pi / 4, 0, math.pi / 2, math.pi / 4,
                    0.04, 0.04, 0, 0, 0]
            for i in range(NUM_JOINTS):
                p.resetJointState(robot, i, home[i])

            # 물체 로드
            for obj in OBJECTS_DEF:
                name = obj["name"]
                pos  = obj["pos"]
                if obj["shape"] == "box":
                    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=obj["half"])
                    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=obj["half"],
                                              rgbaColor=obj["color"])
                elif obj["shape"] == "cylinder":
                    col = p.createCollisionShape(p.GEOM_CYLINDER,
                                                 radius=obj["radius"], height=obj["h_ext"] * 2)
                    vis = p.createVisualShape(p.GEOM_CYLINDER,
                                              radius=obj["radius"], length=obj["h_ext"] * 2,
                                              rgbaColor=obj["color"])
                elif obj["shape"] == "sphere":
                    col = p.createCollisionShape(p.GEOM_SPHERE, radius=obj["radius"])
                    vis = p.createVisualShape(p.GEOM_SPHERE, radius=obj["radius"],
                                              rgbaColor=obj["color"])
                else:
                    continue

                body_id = p.createMultiBody(
                    baseMass=obj["mass"],
                    baseCollisionShapeIndex=col,
                    baseVisualShapeIndex=vis,
                    basePosition=pos,
                )
                self._objects[name] = {
                    "body_id": body_id,
                    "half_z":  obj["half_z"],
                    "pos":     list(pos),
                }
                self._log(f"로드: '{name}' body_id={body_id}")

            # 안정화
            for _ in range(120):
                p.stepSimulation()

            self._capture_overview()
            self._set_status("대기 중 — 자연어 명령을 입력하세요")
            self._log("시뮬레이션 초기화 완료")

            # 아이들 루프
            while self._running:
                try:
                    cmd = self._cmd_queue.get_nowait()
                    self._execute_nl_command(cmd)
                    self._set_status("대기 중 — 자연어 명령을 입력하세요")
                except queue.Empty:
                    pass

                self._step_sim()

        except Exception as e:
            self._log(f"시뮬레이션 치명적 오류: {e}")
            self._set_status(f"오류: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._running = False
            try:
                p.disconnect()
            except Exception:
                pass
