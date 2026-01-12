# 🔍 AI Image Filter Pipeline

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**AI 생성 이미지를 필터링하는 3-Layer 검증 파이프라인**

> 생성형 AI의 발전으로 학습 데이터 오염(Data Contamination) 문제가 발생하고 있습니다. 
> 이 서비스는 현재 오픈소스 모델 + 해시 + 메타데이터를 사용하여 이미지 데이터의 오염을 예방하는 것이 가능한지 테스트해보기 위해 진행하였습니다.
> [Provenance Detection for AI-Generated Images: Combining Perceptual Hashing, Homomorphic Encryption, and AI Detection Models](https://arxiv.org/html/2503.11195v1)을 참고했습니다.
---

## 📋 목차

- [주요 기능](#-주요-기능)
- [아키텍처](#-아키텍처)
- [빠른 시작](#-빠른-시작)
- [API 문서](#-api-문서)
- [배포](#-배포)
- [기술 스택](#-기술-스택)

---

## ✨ 주요 기능

### 3-Layer 검증 시스템

| Layer | 기능 | 설명 |
|-------|------|------|
| **Layer 1** | Hash Check | MD5/SHA256 해시 계산 및 Perceptual Hash 기반 이미지 분석 |
| **Layer 2** | Metadata Analysis | C2PA Content Credentials 검증, EXIF 분석, AI 도구 시그니처 탐지 |
| **Layer 3** | AI Detection | HuggingFace 모델 기반 AI 생성 이미지 판별 |

### 주요 특징

- 🚀 **빠른 분석**: 단일 이미지 2-5초 내 분석 완료
- 📦 **배치 처리**: 최대 50개 이미지 동시 분석
- 📊 **상세 리포트**: 각 Layer별 분석 결과 및 판정 근거 제공
- 🔌 **REST API**: FastAPI 기반 확장 가능한 API
- 🎨 **웹 UI**: Streamlit 기반 직관적인 인터페이스

---

## 🏗 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI                         │
│                   (streamlit_app.py)                    │
└─────────────────────┬───────────────────────────────────┘
                      │ HTTP
                      ▼
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Backend                      │
│                     (app/main.py)                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   Layer 1   │  │   Layer 2   │  │   Layer 3   │    │
│  │ Hash Check  │  │  Metadata   │  │ AI Detect   │    │
│  │             │  │  Analysis   │  │             │    │
│  │ - MD5       │  │ - C2PA      │  │ - HF Model  │    │
│  │ - SHA256    │  │ - EXIF      │  │ - Inference │    │
│  │ - pHash     │  │ - Signature │  │             │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                  Pipeline Service                       │
│              (종합 판정 + 가중치 계산)                    │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 빠른 시작

### 1. 저장소 클론

```bash
git clone https://github.com/world970511/ai-image-filter.git
cd ai-image-filter
```

### 2. 가상환경 설정

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 환경변수 설정

```bash
cp .env.example .env
# .env 파일을 편집하여 필요한 설정 입력
```

### 4. 서버 실행

```bash
# FastAPI 서버 (터미널 1)
uvicorn app.main:app --reload --port 8000

# Streamlit UI (터미널 2)
streamlit run streamlit_app.py
```

### 5. 접속

- **API 문서**: http://localhost:8000/docs
- **웹 UI**: http://localhost:8501

---

## 📡 API 문서

### 단일 이미지 분석

```bash
POST /api/v1/analyze
Content-Type: multipart/form-data

curl -X POST "http://localhost:8000/api/v1/analyze" \
  -F "file=@image.jpg"
```

**응답 예시:**
```json
{
  "id": "uuid",
  "filename": "image.jpg",
  "final_verdict": "ai_generated",
  "confidence_score": 0.87,
  "reasoning": "🤖 AI 탐지 모델 판정: AI 생성 (확신도: 87.0%)",
  "hash_result": { "md5": "...", "sha256": "..." },
  "metadata_result": { "has_c2pa": false, "ai_tool_signatures": [] },
  "detection_result": { "is_ai_generated": true, "confidence": 0.87 }
}
```

### 기타 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/api/v1/analyze` | 단일 이미지 분석 |
| POST | `/api/v1/analyze/batch` | 배치 분석 (최대 50개) |

---

## ☁️ 배포

### Hugging Face Spaces (권장)

1. [Hugging Face](https://huggingface.co)에서 새 Space 생성
2. SDK로 **Docker** 선택
3. 이 저장소 파일들 업로드
4. Secrets 설정 불필요 (Stateless 모드)

### Docker

```bash
# 빌드 및 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f
```

---

## 🛠 기술 스택

| 분류 | 기술 |
|------|------|
| **Backend** | FastAPI, Pydantic, Uvicorn |
| **Frontend** | Streamlit |
| **AI/ML** | HuggingFace Transformers, PyTorch |
| **Image Processing** | Pillow, imagehash |

| **Deployment** | Docker, HuggingFace Spaces |

---

## 📊 google의 SynthID Detector와 비교

---

##  라이선스

MIT License - 자유롭게 사용, 수정, 배포 가능합니다.

---

