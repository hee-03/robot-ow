"""
Robot OW Streamlit 시연 앱
PyBullet 시뮬레이션 화면 + 자연어 명령 입력
"""

import time

import streamlit as st

st.set_page_config(
    page_title="Robot OW — 자연어 로봇 제어",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .status-box {
        background: #1e1e2e;
        border-radius: 8px;
        padding: 10px 14px;
        color: #cdd6f4;
        font-family: monospace;
        font-size: 13px;
        margin-bottom: 6px;
    }
    .status-ok   { border-left: 4px solid #a6e3a1; }
    .status-busy { border-left: 4px solid #f9e2af; }
    .status-err  { border-left: 4px solid #f38ba8; }
    div[data-testid="stMetricValue"] { font-size: 14px; }
</style>
""", unsafe_allow_html=True)


# ── 시뮬레이션 싱글톤 (@st.cache_resource 로 Streamlit 리런 간 유지) ──────────
@st.cache_resource
def get_simulation():
    from robot_sim import RobotSimulation
    sim = RobotSimulation()
    sim.start()
    return sim


# ── 레이아웃 ──────────────────────────────────────────────────────────────────
st.title("Robot OW — 자연어 로봇 제어 시연")
st.caption("Franka Panda 7-DOF  ·  OWL-ViT 비전  ·  LangChain/Ollama")

col_view, col_ctrl = st.columns([3, 1.2], gap="medium")

# ── 제어 패널 ─────────────────────────────────────────────────────────────────
with col_ctrl:
    st.subheader("제어 패널")

    sim = get_simulation()

    # 상태 표시
    status = sim.get_status()
    busy   = sim.is_busy()
    css_cls = "status-busy" if busy else ("status-err" if "오류" in status else "status-ok")
    st.markdown(
        f'<div class="status-box {css_cls}">{status}</div>',
        unsafe_allow_html=True,
    )

    if busy:
        st.warning("작업 실행 중입니다...")

    st.divider()

    # ── 자연어 명령 입력 ──────────────────────────────────────────────────────
    st.subheader("자연어 명령")

    command = st.text_area(
        "명령 입력",
        height=90,
        placeholder="예: 빨간 네모를 0.3 0.2 0.025로 빠르게 옮겨줘",
        disabled=busy,
        key="cmd_input",
    )

    if st.button("명령 실행", type="primary", disabled=busy or not command.strip()):
        sim.send_command(command.strip())
        st.toast(f"명령 전송: {command.strip()[:40]}...", icon="🤖")
        st.rerun()

    st.divider()

    # ── 예시 명령 버튼 ────────────────────────────────────────────────────────
    st.caption("빠른 예시")

    EXAMPLES = [
        ("빨간 네모 빠르게 이동",    "빨간 네모를 0.3 0.2 0.025로 빠르게 옮겨줘"),
        ("파란 원기둥 천천히 이동",  "파란 원기둥을 0.2 0.3 0.05로 천천히 이동시켜줘"),
        ("초록 공 보통 속도 이동",   "초록 공을 -0.2 0.4 0.04로 옮겨줘"),
        ("빨간 박스 신중하게 이동",  "빨간 박스를 -0.3 -0.2 0.025로 조심해서 옮겨줘"),
    ]

    for label, cmd in EXAMPLES:
        if st.button(label, key=f"ex_{label}", disabled=busy, use_container_width=True):
            sim.send_command(cmd)
            st.toast(f"명령 전송!", icon="🤖")
            st.rerun()

    st.divider()

    # ── 물체 현재 위치 ────────────────────────────────────────────────────────
    st.subheader("물체 위치")
    obj_positions = sim.get_object_positions()
    if obj_positions:
        for name, pos in obj_positions.items():
            color_map = {
                "red box":       "#f38ba8",
                "blue cylinder": "#89b4fa",
                "green sphere":  "#a6e3a1",
            }
            color = color_map.get(name, "#cdd6f4")
            st.markdown(
                f'<span style="color:{color}">■</span> **{name}**  '
                f'`[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]`',
                unsafe_allow_html=True,
            )
    else:
        st.caption("로딩 중...")

# ── 시뮬레이션 화면 ───────────────────────────────────────────────────────────
with col_view:
    tab_overview, tab_ee, tab_log = st.tabs(["오버뷰 카메라", "EE 카메라", "실행 로그"])

    with tab_overview:
        frame_ph = st.empty()
        frame = sim.get_frame()
        if frame is not None:
            frame_ph.image(frame, channels="RGB", use_container_width=True,
                           caption="오버뷰 — 고정 카메라 (20 fps)")
        else:
            frame_ph.info("시뮬레이션 초기화 중... 잠시 기다려주세요.")

    with tab_ee:
        ee_ph = st.empty()
        ee_frame = sim.get_ee_frame()
        if ee_frame is not None:
            ee_ph.image(ee_frame, channels="RGB", use_container_width=True,
                        caption="EE 카메라 — 엔드이펙터 시점 (~5 fps)")
        else:
            ee_ph.info("EE 카메라 준비 중...")

    with tab_log:
        logs = sim.get_logs()
        log_text = "\n".join(logs[-30:]) if logs else "로그 없음"
        st.text_area("실행 로그", log_text, height=400, key="log_area")
        if st.button("로그 새로고침"):
            st.rerun()

# ── 자동 새로고침 (활성 시에만) ──────────────────────────────────────────────
if sim.is_running():
    time.sleep(0.05)   # ~20 fps
    st.rerun()
