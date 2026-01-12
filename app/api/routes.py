"""
API Routes - 이미지 분석 엔드포인트
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Optional
import uuid
from datetime import datetime

from app.services.hash_service import HashService
from app.services.metadata_service import MetadataService
from app.services.detection_service import DetectionService
from app.services.pipeline_service import PipelineService
from app.models.schemas import (
    AnalysisResult,
    BatchAnalysisResult,
    PipelineConfig
)

router = APIRouter()

# 서비스 인스턴스
hash_service = HashService()
metadata_service = MetadataService()
detection_service = DetectionService()
pipeline_service = PipelineService()


@router.post("/analyze", response_model=AnalysisResult)
async def analyze_single_image(
    file: UploadFile = File(...),
    skip_ai_detection: bool = False
):
    """
    단일 이미지 분석
    
    3개 Layer를 순차적으로 실행하여 종합 판정 결과를 반환합니다.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")
    
    try:
        contents = await file.read()
        result = await pipeline_service.analyze_image(
            image_bytes=contents,
            filename=file.filename,
            skip_ai_detection=skip_ai_detection
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/batch", response_model=BatchAnalysisResult)
async def analyze_batch_images(
    files: List[UploadFile] = File(...),
    skip_ai_detection: bool = False
):
    """
    배치 이미지 분석 (최대 50개)
    """
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="최대 50개 파일까지 업로드 가능합니다.")
    
    results = []
    for file in files:
        if file.content_type and file.content_type.startswith("image/"):
            try:
                contents = await file.read()
                result = await pipeline_service.analyze_image(
                    image_bytes=contents,
                    filename=file.filename,
                    skip_ai_detection=skip_ai_detection
                )
                results.append(result)
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "error": str(e),
                    "status": "failed"
                })
    
    # 통계 계산
    total = len(results)
    ai_detected = sum(1 for r in results if isinstance(r, dict) and r.get("final_verdict") == "ai_generated")
    real_detected = sum(1 for r in results if isinstance(r, dict) and r.get("final_verdict") == "likely_real")
    
    return BatchAnalysisResult(
        total_processed=total,
        ai_generated_count=ai_detected,
        likely_real_count=real_detected,
        uncertain_count=total - ai_detected - real_detected,
        results=results
    )



