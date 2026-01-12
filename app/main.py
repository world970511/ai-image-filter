"""
AI Image Filter Pipeline - FastAPI Backend
ML 학습 데이터셋에서 AI 생성 이미지를 필터링하는 파이프라인
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api import routes



@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 실행되는 로직"""
    # Startup
    print("✅ Service initialized (Stateless)")
    yield
    # Shutdown
    print("👋 Shutting down...")


app = FastAPI(
    title="AI Image Filter Pipeline",
    description="""
    ## ML 학습 데이터 품질 검증 파이프라인
    
    AI 생성 이미지를 탐지하여 학습 데이터셋의 품질을 보장합니다.
    
    ### 3-Layer 검증 시스템
    - **Layer 1**: Hash Check - 이미지 해시 계산 (MD5, SHA256, Perceptual Hash)
    - **Layer 2**: Metadata Analysis - C2PA/EXIF 분석 및 AI 도구 시그니처 탐지
    - **Layer 3**: AI Detection - ML 모델 기반 AI 생성 이미지 탐지
    
    *Stateless 서비스 - 데이터베이스 미사용*
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(routes.router, prefix="/api/v1", tags=["Image Analysis"])


@app.get("/", tags=["Health"])
async def root():
    return {
        "message": "AI Image Filter Pipeline API",
        "docs": "/docs",
        "health": "ok"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy"}
