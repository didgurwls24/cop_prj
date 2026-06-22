"""
회의록 요약 + 할 일 추출 웹 앱
Google Gemini API를 사용해 회의록을 분석하고 결과를 표시합니다.
"""

import json
import os
import time
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from groq import Groq

# .env 파일에서 API 키 로드 (로컬 실행 시)
load_dotenv()


def _get_secret(key: str) -> str:
    """로컬은 .env, Streamlit Cloud는 st.secrets에서 API 키를 읽습니다."""
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, "")

# ─────────────────────────────────────────
# 상수 정의
# ─────────────────────────────────────────
MODEL_NAME = "gemini-2.0-flash"

# AI 제공자 설정
PROVIDER_GEMINI = "Gemini"
PROVIDER_GROQ   = "Groq"

GEMINI_MODEL = "gemini-2.0-flash"
GROQ_MODEL   = "llama-3.3-70b-versatile"

# 청크당 최대 글자 수 (25,000자 기준으로 분할)
CHUNK_SIZE = 25_000

# 429 오류 발생 시 재시도 횟수와 각 대기 시간(초)
MAX_RETRIES = 3
RETRY_DELAYS = [15, 30, 60]

# 청크 간 자동 대기 시간(초) — Gemini 무료 티어 RPM 한도 초과 방지
INTER_CHUNK_DELAY = 5

PRIORITY_ORDER = {"높음": 0, "보통": 1, "낮음": 2}

# Gemini API에 요청할 JSON 스키마
RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "required": ["summary", "action_items"],
    "properties": {
        "summary": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "회의 핵심 요약 (3~6개 불릿)",
        },
        "action_items": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "required": ["task", "owner", "due", "priority"],
                "properties": {
                    "task":     {"type": "STRING", "description": "할 일 내용"},
                    "owner":    {"type": "STRING", "description": "담당자 (없으면 '미정')"},
                    "due":      {"type": "STRING", "description": "기한 (없으면 '미정')"},
                    "priority": {
                        "type": "STRING",
                        "enum": ["높음", "보통", "낮음"],
                        "description": "우선순위",
                    },
                },
            },
        },
    },
}

SYSTEM_PROMPT = """
당신은 회의록을 분석하는 전문 어시스턴트입니다.
주어진 회의록 텍스트를 읽고 다음 JSON 형식으로만 응답하세요.

규칙:
- summary: 핵심 내용 불릿 3~6개 (간결하고 명확하게)
- action_items: 회의에서 도출된 모든 할 일 항목
  - task: 할 일 내용
  - owner: 담당자 이름 (회의록에 없으면 "미정")
  - due: 기한 (회의록에 없으면 "미정")
  - priority: "높음" / "보통" / "낮음" 중 하나
    - 높음: 기한이 임박하거나 "급함", "반드시", "즉시", "긴급" 등의 표현이 있는 경우
    - 낮음: 장기 과제이거나 "여유롭게", "나중에" 등의 표현이 있는 경우
    - 보통: 그 외의 일반적인 할 일
"""

# Groq는 스키마 강제가 없으므로 예시를 포함한 상세 프롬프트 사용
GROQ_SYSTEM_PROMPT = SYSTEM_PROMPT + """
반드시 아래 형식의 JSON만 출력하세요. 다른 텍스트는 절대 포함하지 마세요.

{
  "summary": ["핵심 내용 1", "핵심 내용 2", "..."],
  "action_items": [
    {"task": "할 일 내용", "owner": "담당자 또는 미정", "due": "기한 또는 미정", "priority": "높음"},
    {"task": "할 일 내용", "owner": "미정", "due": "미정", "priority": "보통"}
  ]
}
"""


# ─────────────────────────────────────────
# 커스텀 예외
# ─────────────────────────────────────────
class RateLimitError(Exception):
    """Gemini 무료 사용량 한도 초과 시 발생하는 예외."""
    pass


# ─────────────────────────────────────────
# 입력 처리 함수
# ─────────────────────────────────────────
def load_text_from_file(uploaded_file) -> str:
    """업로드된 .txt 파일의 내용을 문자열로 반환합니다."""
    try:
        return uploaded_file.read().decode("utf-8")
    except UnicodeDecodeError:
        # UTF-8 실패 시 EUC-KR(한국어 인코딩) 시도
        uploaded_file.seek(0)
        return uploaded_file.read().decode("euc-kr", errors="replace")


# ─────────────────────────────────────────
# AI 호출 함수
# ─────────────────────────────────────────
def _is_rate_limit_error(e: Exception) -> bool:
    """예외가 429 한도 초과 오류인지 확인합니다."""
    msg = str(e)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower()


def _split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """
    긴 텍스트를 문단 경계를 살려 chunk_size 글자 이하의 청크로 분할합니다.
    짧으면 리스트 1개 그대로 반환합니다.
    """
    if len(text) <= chunk_size:
        return [text]

    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = (current + "\n\n" + para) if current else para
        if len(candidate) > chunk_size and current:
            chunks.append(current.strip())
            current = para
        else:
            current = candidate

    if current.strip():
        chunks.append(current.strip())

    return chunks


def _call_single_chunk(client, prompt: str) -> dict:
    """
    재시도(지수 백오프) 로직을 포함한 단일 Gemini API 호출.
    429 오류가 MAX_RETRIES 번 모두 실패하면 RateLimitError를 발생시킵니다.
    """
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=RESPONSE_SCHEMA,
                ),
            )
            return json.loads(response.text)

        except Exception as e:
            last_error = e
            if _is_rate_limit_error(e):
                if attempt < MAX_RETRIES - 1:
                    wait = RETRY_DELAYS[attempt]
                    time.sleep(wait)
                    continue
                # 마지막 시도까지 실패
                raise RateLimitError(str(e)) from e
            # 429가 아닌 다른 오류는 즉시 재발생
            raise

    raise last_error  # 방어 코드 (도달 불가)


def _call_groq_chunk(groq_client, meeting_text: str, chunk_info: str = "") -> dict:
    """
    Groq API를 호출해 회의록 청크를 분석합니다.
    429 오류 발생 시 재시도합니다.
    """
    user_content = f"{chunk_info}\n\n---회의록 시작---\n{meeting_text}\n---회의록 끝---"

    for attempt in range(MAX_RETRIES):
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": GROQ_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            raw = response.choices[0].message.content
            return json.loads(raw)
        except Exception as e:
            if _is_rate_limit_error(e):
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAYS[attempt])
                    continue
                raise RateLimitError(str(e)) from e
            raise

    raise RuntimeError("Groq API 호출 실패")


def _merge_summaries(client, partial_summaries: list[str]) -> list[str]:
    """
    여러 청크에서 나온 부분 요약들을 하나의 최종 요약(3~6개 불릿)으로 합칩니다.
    client가 None이면 Groq 대신 간단히 앞 6개를 반환합니다.
    """
    if len(partial_summaries) <= 6:
        return partial_summaries

    # client가 None이면 Groq 경로: 추가 API 호출 없이 앞 6개 사용
    if client is None:
        return partial_summaries[:6]

    bullets = "\n".join(f"- {s}" for s in partial_summaries)
    merge_prompt = (
        "다음은 긴 회의록을 여러 부분으로 나누어 분석한 부분 요약들입니다.\n"
        "이 내용을 바탕으로 전체 회의의 핵심을 3~6개 불릿으로 간결하게 통합 요약해주세요.\n\n"
        f"부분 요약:\n{bullets}"
    )
    merge_schema = {
        "type": "OBJECT",
        "required": ["summary"],
        "properties": {
            "summary": {"type": "ARRAY", "items": {"type": "STRING"}},
        },
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=merge_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=merge_schema,
                ),
            )
            merged = json.loads(response.text)
            return merged.get("summary", partial_summaries[:6])
        except Exception as e:
            if _is_rate_limit_error(e) and attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAYS[attempt])
                continue
            return partial_summaries[:6]

    return partial_summaries[:6]


def call_api(meeting_text: str, provider: str = PROVIDER_GEMINI, status_placeholder=None) -> dict:
    """
    회의록을 분석해 {"summary": [...], "action_items": [...]} 를 반환합니다.

    - provider: PROVIDER_GEMINI 또는 PROVIDER_GROQ
    - 텍스트가 CHUNK_SIZE를 초과하면 자동으로 청크 분할 후 병합합니다.
    - 429 오류 발생 시 자동 재시도(백오프)합니다.
    """
    def _update_status(msg: str):
        if status_placeholder is not None:
            status_placeholder.info(msg)

    # ── 제공자별 클라이언트 초기화 ──────────
    if provider == PROVIDER_GROQ:
        api_key = _get_secret("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY가 설정되지 않았습니다.")
        groq_client = Groq(api_key=api_key)
    else:
        api_key = _get_secret("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY가 설정되지 않았습니다.")
        gemini_client = genai.Client(api_key=api_key)

    chunks = _split_into_chunks(meeting_text)
    total  = len(chunks)

    # ── 단일 청크 호출 헬퍼 ─────────────────
    def _call_chunk(chunk: str, chunk_info: str = "") -> dict:
        if provider == PROVIDER_GROQ:
            return _call_groq_chunk(groq_client, chunk, chunk_info)
        else:
            prompt = f"{SYSTEM_PROMPT}\n\n{chunk_info}\n\n---회의록 시작---\n{chunk}\n---회의록 끝---"
            return _call_single_chunk(gemini_client, prompt)

    # ── 단일 청크 ───────────────────────────
    if total == 1:
        _update_status("🔍 회의록을 분석 중입니다...")
        result = _call_chunk(chunks[0])

    # ── 복수 청크 ───────────────────────────
    else:
        _update_status(f"📄 회의록이 길어 {total}개 구간으로 나누어 분석합니다.")
        all_summaries:    list[str]  = []
        all_action_items: list[dict] = []

        for i, chunk in enumerate(chunks):
            # Gemini 무료 티어만 청크 간 대기 (Groq는 한도가 넉넉해 불필요)
            if i > 0 and provider == PROVIDER_GEMINI:
                for remaining in range(INTER_CHUNK_DELAY, 0, -1):
                    _update_status(f"⏳ API 한도 초과 방지를 위해 {remaining}초 대기 중...")
                    time.sleep(1)

            _update_status(f"🔍 구간 {i + 1} / {total} 분석 중...")
            part = _call_chunk(chunk, f"[전체 회의록의 {i + 1}/{total} 구간]")
            all_summaries.extend(part.get("summary", []))
            all_action_items.extend(part.get("action_items", []))

        # 요약 합성 (Gemini만 청크 간 대기)
        if provider == PROVIDER_GEMINI:
            for remaining in range(INTER_CHUNK_DELAY, 0, -1):
                _update_status(f"⏳ API 한도 초과 방지를 위해 {remaining}초 대기 중...")
                time.sleep(1)

        _update_status("✂️ 부분 요약을 통합 중입니다...")
        final_summary = _merge_summaries(gemini_client if provider == PROVIDER_GEMINI else None, all_summaries)
        result = {"summary": final_summary, "action_items": all_action_items}

    result["action_items"].sort(
        key=lambda item: PRIORITY_ORDER.get(item.get("priority", "보통"), 1)
    )
    return result


# ─────────────────────────────────────────
# 결과 표시 함수
# ─────────────────────────────────────────
def display_summary(summary: list[str]) -> None:
    """회의 요약 불릿을 화면에 표시합니다."""
    st.subheader("📋 회의 요약")
    for bullet in summary:
        st.markdown(f"- {bullet}")


def display_action_items(action_items: list[dict]) -> None:
    """할 일 목록을 우선순위 강조와 함께 표로 표시합니다."""
    st.subheader("✅ 할 일 목록")

    if not action_items:
        st.info("추출된 할 일 항목이 없습니다.")
        return

    # 다크/라이트 모드 양쪽에서 잘 보이는 rgba 기반 뱃지
    priority_badge = {
        "높음": '<span style="background:#ef4444;color:#fff;padding:4px 12px;border-radius:20px;font-size:0.78rem;font-weight:700;letter-spacing:0.02em;white-space:nowrap;">높음</span>',
        "보통": '<span style="background:#d97706;color:#fff;padding:4px 12px;border-radius:20px;font-size:0.78rem;font-weight:700;letter-spacing:0.02em;white-space:nowrap;">보통</span>',
        "낮음": '<span style="background:#475569;color:#fff;padding:4px 12px;border-radius:20px;font-size:0.78rem;font-weight:700;letter-spacing:0.02em;white-space:nowrap;">낮음</span>',
    }

    # 우선순위별 행 스타일 — rgba로 다크/라이트 모드 모두 대응
    row_styles = {
        "높음": "background:rgba(239,68,68,0.15);border-left:4px solid #ef4444;",
        "보통": "background:rgba(217,119,6,0.08);border-left:4px solid transparent;",
        "낮음": "background:rgba(100,116,139,0.06);border-left:4px solid transparent;",
    }

    cell  = "padding:12px 16px;border-bottom:1px solid rgba(148,163,184,0.15);font-size:0.88rem;line-height:1.5;"
    c_mid = cell + "text-align:center;white-space:nowrap;"

    table = f"""
    <table style="width:100%;border-collapse:collapse;margin-top:4px;border-radius:10px;overflow:hidden;">
      <thead>
        <tr style="background:#1e3a5f;">
          <th style="padding:12px 16px;text-align:left;color:#e2e8f0;font-weight:600;font-size:0.85rem;letter-spacing:0.04em;">할 일 내용</th>
          <th style="padding:12px 16px;text-align:center;color:#e2e8f0;font-weight:600;font-size:0.85rem;letter-spacing:0.04em;white-space:nowrap;">담당자</th>
          <th style="padding:12px 16px;text-align:center;color:#e2e8f0;font-weight:600;font-size:0.85rem;letter-spacing:0.04em;white-space:nowrap;">기한</th>
          <th style="padding:12px 16px;text-align:center;color:#e2e8f0;font-weight:600;font-size:0.85rem;letter-spacing:0.04em;white-space:nowrap;">우선순위</th>
        </tr>
      </thead>
      <tbody>
    """

    for item in action_items:
        priority = item.get("priority", "보통")
        row_style = row_styles.get(priority, row_styles["보통"])
        table += f"""
        <tr style="{row_style}">
          <td style="{cell}">{item.get('task', '')}</td>
          <td style="{c_mid}">{item.get('owner', '미정')}</td>
          <td style="{c_mid}">{item.get('due', '미정')}</td>
          <td style="{c_mid}">{priority_badge.get(priority, priority)}</td>
        </tr>
        """

    table += "</tbody></table>"
    st.html(table)


# ─────────────────────────────────────────
# 테스트 모드 샘플 데이터
# ─────────────────────────────────────────
def get_sample_result() -> dict:
    """API 호출 없이 UI를 테스트하기 위한 샘플 결과를 반환합니다."""
    return {
        "summary": [
            "2분기 마케팅 캠페인 예산이 전년 대비 15% 증가하여 총 3억 원으로 확정됨",
            "SNS 광고(인스타그램·유튜브) 집중 운영으로 MZ세대 타깃 강화 방향 합의",
            "신규 브랜드 슬로건 최종 후보 3개를 다음 주까지 디자인팀에서 시안 제출하기로 결정",
            "7월 론칭 이벤트 장소를 성수동 팝업 스토어로 확정, 운영 기간은 2주",
            "외부 인플루언서 협업 건은 법무팀 계약 검토 후 진행 예정 (기한: 6월 말)",
        ],
        "action_items": [
            {
                "task": "SNS 광고 소재 3종 제작 및 A/B 테스트 설계",
                "owner": "김지수",
                "due": "2026-06-27",
                "priority": "높음",
            },
            {
                "task": "인플루언서 계약서 법무팀 검토 요청",
                "owner": "박민준",
                "due": "2026-06-30",
                "priority": "높음",
            },
            {
                "task": "브랜드 슬로건 시안 3개 제출 (디자인팀)",
                "owner": "이수연",
                "due": "2026-06-28",
                "priority": "높음",
            },
            {
                "task": "성수동 팝업 스토어 운영 인력 배치표 작성",
                "owner": "최현우",
                "due": "2026-07-05",
                "priority": "보통",
            },
            {
                "task": "2분기 예산 집행 계획서 재무팀 공유",
                "owner": "김지수",
                "due": "2026-07-03",
                "priority": "보통",
            },
            {
                "task": "유튜브 채널 운영 가이드라인 문서 업데이트",
                "owner": "미정",
                "due": "미정",
                "priority": "낮음",
            },
        ],
    }


# ─────────────────────────────────────────
# 마크다운 생성 함수
# ─────────────────────────────────────────
def generate_markdown(summary: list[str], action_items: list[dict]) -> str:
    """분석 결과를 마크다운 형식의 문자열로 변환합니다."""
    now = datetime.now().strftime("%Y년 %m월 %d일 %H:%M")
    lines = [
        f"# 회의록 분석 결과",
        f"",
        f"생성 시각: {now}",
        f"",
        f"---",
        f"",
        f"## 📋 회의 요약",
        f"",
    ]
    for bullet in summary:
        lines.append(f"- {bullet}")

    lines += [
        f"",
        f"---",
        f"",
        f"## ✅ 할 일 목록",
        f"",
        f"| 할 일 내용 | 담당자 | 기한 | 우선순위 |",
        f"|---|---|---|---|",
    ]
    for item in action_items:
        task     = item.get("task", "")
        owner    = item.get("owner", "미정")
        due      = item.get("due", "미정")
        priority = item.get("priority", "보통")
        # 마크다운 파이프 문자 이스케이프
        task = task.replace("|", "\\|")
        lines.append(f"| {task} | {owner} | {due} | {priority} |")

    return "\n".join(lines)


# ─────────────────────────────────────────
# 메인 앱
# ─────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="회의록 분석기",
        page_icon="📝",
        layout="wide",
    )

    # 헤더
    st.title("📝 회의록 요약 & 할 일 추출기")
    st.caption("클로바노트 등에서 변환한 회의록 텍스트를 붙여넣거나 .txt 파일을 업로드하세요.")

    # ── AI 제공자 선택 ──────────────────────
    provider = st.radio(
        "AI 제공자 선택",
        options=[PROVIDER_GROQ, PROVIDER_GEMINI],
        index=0,
        horizontal=True,
        captions=["무료 · 한도 넉넉 (권장)", "무료 티어 · 분당 한도 있음"],
    )

    st.divider()

    # ── 입력 영역 ──────────────────────────
    col_left, col_right = st.columns([3, 1])

    with col_right:
        st.markdown("**📂 파일 업로드**")
        uploaded_file = st.file_uploader(
            "label_hidden",
            type=["txt"],
            label_visibility="collapsed",
            help=".txt 파일을 업로드하면 내용이 자동으로 입력창에 채워집니다.",
        )

    # 파일이 업로드되면 텍스트 추출
    file_content = ""
    if uploaded_file is not None:
        file_content = load_text_from_file(uploaded_file)
        st.toast(f"✅ '{uploaded_file.name}' 파일을 불러왔습니다.")

    with col_left:
        meeting_text = st.text_area(
            "**📄 회의록 입력**",
            value=file_content,
            height=300,
            placeholder="여기에 회의록 텍스트를 직접 붙여넣거나, 오른쪽에서 .txt 파일을 업로드하세요...",
        )

    st.divider()

    # ── 분석 버튼 + 테스트 모드 ────────────
    btn_col, toggle_col = st.columns([2, 3])
    with btn_col:
        analyze_btn = st.button("🔍 분석하기", type="primary", use_container_width=False)
    with toggle_col:
        test_mode = st.toggle(
            "🧪 테스트 모드 (API 호출 없이 샘플 결과 표시)",
            value=False,
            help="API 한도 걱정 없이 UI를 확인하고 싶을 때 사용하세요.",
        )

    if analyze_btn:
        # ── 테스트 모드: API 미호출 ─────────
        if test_mode:
            st.info("🧪 테스트 모드: 샘플 데이터로 결과를 표시합니다. (API 미호출)")
            result = get_sample_result()
            result["action_items"].sort(
                key=lambda item: PRIORITY_ORDER.get(item.get("priority", "보통"), 1)
            )

        # ── 실제 분석 모드 ──────────────────
        else:
            if not meeting_text.strip():
                st.warning("⚠️ 회의록 내용을 입력하거나 파일을 업로드해주세요.")
                return

        if not test_mode:
            spinner_msg = f"{'Groq' if provider == PROVIDER_GROQ else 'Gemini'} AI가 회의록을 분석 중입니다... 잠시 기다려주세요."
            status_placeholder = st.empty()
            with st.spinner(spinner_msg):
                try:
                    result = call_api(meeting_text, provider=provider, status_placeholder=status_placeholder)
                except RateLimitError:
                    status_placeholder.empty()
                    st.error("⏳ 무료 사용량 한도에 걸렸어요.")
                    st.warning(
                        "잠시 후 다시 시도하거나, 회의록을 짧게 나눠서 입력해 주세요.\n\n"
                        "Gemini 무료 티어는 분당 요청 횟수에 제한이 있습니다. "
                        "1~2분 뒤 재시도하면 대부분 해결됩니다."
                    )
                    return
                except ValueError as e:
                    status_placeholder.empty()
                    st.error(f"🔑 API 키 오류: {e}")
                    st.info("프로젝트 루트에 `.env` 파일을 만들고 `GEMINI_API_KEY=your_key` 를 추가하세요.")
                    return
                except json.JSONDecodeError as e:
                    status_placeholder.empty()
                    st.error("❌ AI 응답을 JSON으로 파싱하는 데 실패했습니다.")
                    st.warning(
                        "Gemini 모델이 예상치 못한 형식으로 응답했습니다. "
                        "잠시 후 다시 시도해보거나, 회의록 내용이 충분한지 확인해주세요."
                    )
                    with st.expander("기술적 오류 상세 보기"):
                        st.code(str(e))
                    return
                except Exception as e:
                    status_placeholder.empty()
                    st.error("❌ 분석 중 오류가 발생했습니다.")
                    with st.expander("기술적 오류 상세 보기"):
                        st.code(str(e))
                    return
            status_placeholder.empty()

        # ── 결과 표시 ──────────────────────
        st.success("✅ 분석이 완료되었습니다!" if not test_mode else "🧪 테스트 모드 결과입니다.")
        st.divider()

        summary      = result.get("summary", [])
        action_items = result.get("action_items", [])

        display_summary(summary)
        st.markdown("<br>", unsafe_allow_html=True)
        display_action_items(action_items)

        # ── 다운로드 버튼 ──────────────────
        st.divider()
        md_content = generate_markdown(summary, action_items)
        filename   = f"회의록_분석결과_{datetime.now().strftime('%Y%m%d_%H%M')}.md"

        st.download_button(
            label="⬇️ 마크다운 파일로 다운로드",
            data=md_content.encode("utf-8"),
            file_name=filename,
            mime="text/markdown",
        )


if __name__ == "__main__":
    main()
