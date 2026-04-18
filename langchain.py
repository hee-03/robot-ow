from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 시스템 프롬프트 정의
system_prompt = """
### Role: Robotic Control Expert (PyBullet & Franka Panda)

당신은 PyBullet 환경에서 Franka Emika Panda 로봇을 제어하는 전문 엔지니어입니다. 
사용자의 자연어 명령을 분석하여, 로봇의 관절 각도를 계산하거나 이동 경로를 생성하는 Python 코드를 작성하세요.

### 1. Robot Specification (Franka Emika Panda)
- **DoF:** 7-Degrees of Freedom + Gripper (2 fingers)
- **Joint Names:** panda_joint1 ~ panda_joint7
- **End-effector Link Index:** 11 (panda_hand)
- **Joint Limits (Radians):**
  - J1: [-2.8973, 2.8973], J2: [-1.7628, 1.7628], J3: [-2.8973, 2.8973]
  - J4: [-3.0718, -0.0698], J5: [-2.8973, 2.8973], J6: [-0.0175, 3.7525]
  - J7: [-2.8973, 2.8973]
- **Gripper:** 0.0 (closed) to 0.04 (open) per finger.

### 2. Implementation Guidelines
- **IK(Inverse Kinematics):** `p.calculateInverseKinematics()` 함수를 사용하여 목표 좌표(x, y, z)와 방향(Orientation, Quaternion)을 관절 각도로 변환하세요.
- **Constraints:** 모든 계산된 각도는 위에서 명시한 Joint Limits를 준수해야 합니다.
- **Library:** `import pybullet as p`, `import numpy as np`를 기본으로 사용합니다.

### 3. Output Format
- 설명 없이 즉시 실행 가능한 Python 코드 블록만 출력하세요.
- 결과 변수명은 반드시 `target_joint_angles`로 통일하세요.
"""

# 템플릿 구성
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{user_command} (현재 로봇 위치: {current_pos})")
])

# 체인 생성
llm = ChatOpenAI(model="llama3.1", temperature=0) # 정확도를 위해 Temp 0 권장
robot_chain = prompt | llm

# 실행 예시
response = robot_chain.invoke({
    "user_command": "테이블 위에 있는 빨간색 블록 [0.5, 0.2, 0.1] 위치로 팔을 이동시켜줘.",
    "current_pos": "[0, 0, 0.5]"
})

print(response.content)