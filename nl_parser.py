"""
자연어 명령 → 구조화된 JSON 파싱 모듈
LangChain + Ollama (llama3.1)
"""

import json
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

SYSTEM_PROMPT = """
당신은 로봇 제어 명령 파서입니다.
사용자의 자연어 명령을 분석하여 반드시 아래 JSON 형식으로만 응답하세요.
설명, 마크다운 코드블록, 주석 없이 JSON 객체 하나만 출력하세요.

출력 형식:
{{
  "object": "<인식할 객체 이름 (영문 또는 한글)>",
  "destination": [x, y, z],
  "speed_level": "fast" | "normal" | "slow" | "careful"
}}

파싱 규칙:
- 좌표는 명령에서 숫자 3개를 [x, y, z] 순서로 추출
- "빠르게" / "빨리" / "quickly" / "fast"   → speed_level = "fast"
- "조심해서" / "천천히" / "carefully"       → speed_level = "careful"
- "느리게" / "slowly"                      → speed_level = "slow"
- 속도 언급 없음                            → speed_level = "normal"
- object는 인식 가능한 물체 이름으로 영문 변환 권장 (예: 빨간 네모 → red box)
"""

_chain = None

def _get_chain():
    global _chain
    if _chain is None:
        llm = ChatOllama(model="llama3.1", temperature=0)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{command}"),
        ])
        _chain = prompt | llm
    return _chain


def parse_command(command: str) -> dict:
    """
    자연어 명령을 파싱하여 dict 반환.
    {object, destination, speed_level}
    """
    print(f"[Parser] 명령 파싱 중: '{command}'")
    response = _get_chain().invoke({"command": command})
    raw = response.content.strip()

    # 코드블록 마크다운 제거
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

    # JSON 추출 (응답에 여분 텍스트가 섞여도 처리)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"LLM 응답에서 JSON을 찾을 수 없습니다:\n{raw}")

    parsed = json.loads(match.group())

    # 필수 키 검증
    for key in ("object", "destination", "speed_level"):
        if key not in parsed:
            raise ValueError(f"파싱 결과에 '{key}' 키가 없습니다: {parsed}")

    dest = parsed["destination"]
    if not (isinstance(dest, list) and len(dest) == 3):
        raise ValueError(f"destination이 [x,y,z] 형식이 아닙니다: {dest}")

    parsed["destination"] = [float(v) for v in dest]
    print(f"[Parser] 결과: object='{parsed['object']}'  "
          f"destination={parsed['destination']}  speed='{parsed['speed_level']}'")
    return parsed
