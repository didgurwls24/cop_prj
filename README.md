# 📝 회의록 요약 & 할 일 추출기

클로바노트 등에서 변환한 회의록 텍스트를 입력하면 Google Gemini AI가 자동으로 **회의 요약**과 **할 일 목록(담당자·기한·우선순위 포함)** 을 추출해 주는 웹 앱입니다.

---

## 주요 기능

- 텍스트 직접 붙여넣기 또는 `.txt` 파일 업로드 지원
- Gemini Flash 모델을 사용한 회의 요약 (핵심 불릿 3~6개)
- 할 일 목록 자동 추출 (담당자·기한·우선순위 포함)
- 우선순위 높음 항목 색상 강조 + 자동 정렬 (높음 → 보통 → 낮음)
- 결과를 마크다운 파일로 다운로드

---

## 실행 방법

### 1단계: 저장소 클론 또는 파일 준비

프로젝트 폴더로 이동합니다.

```bash
cd meeting-summarizer
```

### 2단계: Python 가상환경 생성 및 활성화 (권장)

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
# venv\Scripts\activate       # Windows
```

### 3단계: 패키지 설치

```bash
pip install -r requirements.txt
```

### 4단계: API 키 설정

`.env.example` 파일을 복사해 `.env` 파일을 만들고, 발급받은 Gemini API 키를 입력합니다.

```bash
cp .env.example .env
```

`.env` 파일을 열어 아래와 같이 수정합니다:

```
GEMINI_API_KEY=AIza...실제키입력...
```

> **Gemini API 키 발급 방법**: [Google AI Studio](https://aistudio.google.com/app/apikey) 에서 무료로 발급받을 수 있습니다.

### 5단계: 앱 실행

```bash
streamlit run app.py
```

브라우저에서 자동으로 `http://localhost:8501` 이 열립니다.

---

## 사용 방법

1. 회의록 텍스트를 입력창에 붙여넣거나, `.txt` 파일을 업로드합니다.
2. **"🔍 분석하기"** 버튼을 클릭합니다.
3. 잠시 후 회의 요약과 할 일 목록이 화면에 표시됩니다.
4. **"⬇️ 마크다운 파일로 다운로드"** 버튼으로 결과를 저장할 수 있습니다.

---

## 파일 구조

```
meeting-summarizer/
├── app.py              # 메인 Streamlit 앱
├── requirements.txt    # 필요 패키지 목록
├── .env                # API 키 (직접 생성, git에 포함하지 말 것)
├── .env.example        # .env 작성 예시
└── README.md           # 이 파일
```

---

## 주의사항

- `.env` 파일은 절대 git에 커밋하지 마세요. `.gitignore`에 `.env`를 추가해두는 것을 권장합니다.
- Gemini 무료 티어는 분당 요청 횟수에 제한이 있습니다. 오류 발생 시 잠시 후 재시도하세요.
