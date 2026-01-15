# 🔍 AI Image Filter

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**AI 생성 이미지를 필터링하는 3-Layer 검증 서비스**

- 생성형 AI의 발전으로 학습 데이터 오염(Data Contamination) 문제가 발생하고 있습니다.  
- 이 서비스는 ai로 생성된 이미지의 경우 일반 디지털 사진에 있는 카메라 모델, 렌즈 유형, 셔터 속도, GPS 위치 정보 등 EXIF 데이터가 존재하지 않는다는 부분과 [Provenance Detection for AI-Generated Images: Combining Perceptual Hashing, Homomorphic Encryption, and AI Detection Models](https://arxiv.org/html/2503.11195v1)에 제시된 내용을 바탕으로 현재 해시 - 메타데이터 - 오픈소스 탐지 모델 이렇게 3Layer를 사용하여 이미지 데이터의 오염을 예방하는 것이 가능한지 테스트해보기 위해 진행하였습니다.  
- DinoHash (DinoV2 기반 지각적 해싱) 사용 시 필요한 데이터는 [ai-vs-human-generated-dataset](https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset/data)를 다운받아 ai 생성 이미지 39975장을 DinoV2로 벡터화하여 npy 파일로 저장하여 활용하였습니다. 유사도 임계값은 DINOv2 논문과 의료 영상 연구를 참고하여 0.85로 설정하였으며, 70-85% 구간은 불확실 영역으로 점진적 점수를 부여합니다.  
- 메타데이터 검사의 경우 EXIF 진위성 점수 계산을 핵심으로 하며, C2PA Content Credentials 검증과 AI 도구 시그니처 탐지를 보조적으로 활용합니다. EXIF 분석에서는 카메라 정보, 촬영 설정, GPS 등을 종합하여 진위성 점수 (0.0 ~ 1.0)를 계산하고 비정상 패턴을 탐지합니다.  
- AI 모델은 허깅스페이스의 [ai_vs_human_generated_image_detection](https://huggingface.co/dima806/ai_vs_human_generated_image_detection)을 사용하였습니다.
- 최종적으로는 해시/메타데이터/오픈소스 탐지 결과 각각에 0.3/0.4/0.3의 가중치를 각각 부여하여 종합적으로 판정하였습니다.  
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
| **Layer 1** | Hash Check | DinoHash (DinoV2 벡터 유사도, 임계값 0.85) 기반 AI 이미지 DB 매칭, 점진적 점수 계산 |
| **Layer 2** | Metadata Analysis | EXIF 진위성 점수 계산 (핵심) + C2PA 검증/AI 시그니처 탐지 (보조) |
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
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │   Layer 1   │  │   Layer 2   │  │   Layer 3   │      │
│  │ Hash Check  │  │  Metadata   │  │ AI Detect   │      │
│  │             │  │  Analysis   │  │             │      │
│  │ - DinoV2    │  │ - EXIF      │  │ - HF Model  │      │
│  │   Vector    │  │   Score     │  │ - Inference │      │
│  │ Similarity  │  │ - C2PA/Sign │  │             │      │
│  └─────────────┘  └─────────────┘  └─────────────┘      │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                  Pipeline Service                       │
│              (종합 판정 + 가중치 계산)                     │
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
  "hash_result": { "DinoHash": "..."},
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
> 테스트 데이터 이미지에 대한 설명은 /testIMG 폴더 내의 data-readme.md를 참고해주세요. 관련 내용은 프로젝트 회고와 함께 블로그 포스트에 올릴 예정입니다.



---

##  라이선스

MIT License - 자유롭게 사용, 수정, 배포 가능합니다.

---

