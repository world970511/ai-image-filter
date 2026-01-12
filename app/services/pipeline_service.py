"""
Pipeline Service
3개 Layer를 통합하여 종합 판정 수행
"""

import uuid
import time
from datetime import datetime
from typing import Dict, Any, Optional

from app.services.hash_service import HashService
from app.services.metadata_service import MetadataService
from app.services.detection_service import DetectionService
from app.models.schemas import (
    AnalysisResult, 
    HashResult, 
    MetadataResult, 
    DetectionResult,
    VerdictType
)



class PipelineService:
    """3-Layer 분석 파이프라인 서비스"""
    
    def __init__(self):
        self.hash_service = HashService()
        self.metadata_service = MetadataService()
        self.detection_service = DetectionService()
        
        # 판정 임계값
        self.CONFIDENCE_THRESHOLD = 0.7
        self.AI_DETECTION_WEIGHT = 0.6
        self.METADATA_WEIGHT = 0.3
        self.HASH_WEIGHT = 0.1
    
    async def analyze_image(
        self, 
        image_bytes: bytes, 
        filename: str,
        skip_ai_detection: bool = False
    ) -> AnalysisResult:
        """
        이미지 종합 분석 실행
        
        Args:
            image_bytes: 이미지 바이너리 데이터
            filename: 파일명
            skip_ai_detection: AI 탐지 스킵 여부 (빠른 분석용)
        """
        start_time = time.time()
        analysis_id = str(uuid.uuid4())
        layers_executed = []
        
        # ========== Layer 1: Hash Check ==========
        layer1_start = time.time()
        hash_data = self.hash_service.compute_hash(image_bytes)
        is_duplicate = await self.hash_service.check_duplicate(hash_data["md5"])
        
        hash_result = HashResult(
            md5=hash_data["md5"],
            sha256=hash_data["sha256"],
            perceptual_hash=hash_data.get("perceptual_hash"),
            is_duplicate=is_duplicate
        )
        layers_executed.append("hash_check")
        layer1_time = (time.time() - layer1_start) * 1000
        
        # ========== Layer 2: Metadata Analysis ==========
        layer2_start = time.time()
        metadata_data = self.metadata_service.analyze(image_bytes, filename)
        
        metadata_result = MetadataResult(
            has_c2pa=metadata_data.get("has_c2pa", False),
            c2pa_info=metadata_data.get("c2pa_info"),
            exif_data=metadata_data.get("exif_data"),
            ai_tool_signatures=metadata_data.get("ai_tool_signatures", []),
            software_used=metadata_data.get("software_used"),
            creation_date=metadata_data.get("creation_date")
        )
        layers_executed.append("metadata_analysis")
        layer2_time = (time.time() - layer2_start) * 1000
        
        # ========== Layer 3: AI Detection ==========
        detection_result = None
        layer3_time = 0
        
        if not skip_ai_detection:
            layer3_start = time.time()
            detection_data = await self.detection_service.detect(image_bytes)
            
            if "error" not in detection_data:
                detection_result = DetectionResult(
                    model_name=detection_data["model_name"],
                    is_ai_generated=detection_data["is_ai_generated"],
                    confidence=detection_data["confidence"],
                    raw_scores=detection_data.get("raw_scores")
                )
            layers_executed.append("ai_detection")
            layer3_time = (time.time() - layer3_start) * 1000
        
        # ========== 종합 판정 ==========
        verdict, confidence, reasoning = self._compute_verdict(
            hash_result=hash_result,
            metadata_result=metadata_result,
            detection_result=detection_result
        )
        
        total_time = (time.time() - start_time) * 1000
        
        # 결과 생성
        result = AnalysisResult(
            id=analysis_id,
            filename=filename,
            analyzed_at=datetime.utcnow(),
            hash_result=hash_result,
            metadata_result=metadata_result,
            detection_result=detection_result,
            final_verdict=verdict,
            confidence_score=confidence,
            reasoning=reasoning,
            total_execution_time_ms=round(total_time, 2),
            layers_executed=layers_executed
        )
        

        
        return result
    
    def _compute_verdict(
        self,
        hash_result: HashResult,
        metadata_result: MetadataResult,
        detection_result: Optional[DetectionResult]
    ) -> tuple[VerdictType, float, str]:
        """
        종합 판정 계산
        
        가중치 기반 판정:
        - AI Detection: 60%
        - Metadata: 30%
        - Hash: 10%
        """
        scores = {
            "ai": 0.0,
            "real": 0.0
        }
        reasons = []
        
        # 1. Hash 기반 판정 (중복이면 이전 판정 참조 가능)
        if hash_result.is_duplicate:
            reasons.append("⚠️ 중복 이미지 발견")
        
        # 2. Metadata 기반 판정
        if metadata_result.ai_tool_signatures:
            tools = ", ".join(metadata_result.ai_tool_signatures)
            scores["ai"] += self.METADATA_WEIGHT
            reasons.append(f"🔍 AI 도구 시그니처 발견: {tools}")
        
        if metadata_result.has_c2pa:
            reasons.append("📜 C2PA Content Credentials 발견")
            # C2PA가 있으면 추가 분석 (AI 관련 assertion 확인)
            c2pa_info = metadata_result.c2pa_info or {}
            if c2pa_info.get("ai_related_assertions"):
                scores["ai"] += self.METADATA_WEIGHT * 0.5
                reasons.append("🤖 C2PA에 AI 생성 관련 정보 포함")
        
        # EXIF에 특정 패턴이 없으면 의심
        if not metadata_result.exif_data or len(metadata_result.exif_data) < 3:
            scores["ai"] += self.METADATA_WEIGHT * 0.3
            reasons.append("📷 EXIF 메타데이터 부족/없음 (AI 이미지 특성)")
        else:
            scores["real"] += self.METADATA_WEIGHT * 0.3
            reasons.append("📷 EXIF 메타데이터 존재")
        
        # 3. AI Detection 기반 판정
        if detection_result:
            if detection_result.is_ai_generated:
                scores["ai"] += self.AI_DETECTION_WEIGHT * detection_result.confidence
                reasons.append(
                    f"🤖 AI 탐지 모델 판정: AI 생성 "
                    f"(확신도: {detection_result.confidence:.1%})"
                )
            else:
                scores["real"] += self.AI_DETECTION_WEIGHT * detection_result.confidence
                reasons.append(
                    f"✅ AI 탐지 모델 판정: 실제 이미지 가능성 "
                    f"(확신도: {detection_result.confidence:.1%})"
                )
        else:
            reasons.append("⏭️ AI 탐지 스킵됨")
        
        # 최종 판정
        total_score = scores["ai"] + scores["real"]
        if total_score == 0:
            verdict = VerdictType.UNCERTAIN
            confidence = 0.5
        else:
            ai_ratio = scores["ai"] / total_score if total_score > 0 else 0.5
            
            if ai_ratio >= self.CONFIDENCE_THRESHOLD:
                verdict = VerdictType.AI_GENERATED
                confidence = ai_ratio
            elif ai_ratio <= (1 - self.CONFIDENCE_THRESHOLD):
                verdict = VerdictType.LIKELY_REAL
                confidence = 1 - ai_ratio
            else:
                verdict = VerdictType.UNCERTAIN
                confidence = 0.5 + abs(ai_ratio - 0.5)
        
        reasoning = " | ".join(reasons)
        
        return verdict, round(confidence, 4), reasoning
    

