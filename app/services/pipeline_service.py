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

    def __init__(
        self,
        db_vectors_path: str = './data/ai_dinohashes.npy',
        metadata_path: str = './data/ai_metadata.csv',
        similarity_threshold: float = 0.85
    ):
        """
        PipelineService 초기화

        Args:
            db_vectors_path: AI 이미지 벡터 파일 경로
            metadata_path: AI 이미지 메타데이터 파일 경로
            similarity_threshold: DinoV2 유사도 임계값
        """
        self.hash_service = HashService(
            db_vectors_path=db_vectors_path,
            metadata_path=metadata_path,
            threshold=similarity_threshold if similarity_threshold else 0.85
        )
        self.metadata_service = MetadataService()
        self.detection_service = DetectionService()

        # 판정 임계값
        self.CONFIDENCE_THRESHOLD = 0.7
        self.AI_DETECTION_WEIGHT = 0.3
        self.METADATA_WEIGHT = 0.4
        self.HASH_WEIGHT = 0.3
    
    async def analyze_image(
        self, 
        image_bytes: bytes, 
        filename: str,
    ) -> AnalysisResult:
        """
        이미지 종합 분석 실행
        
        Args:
            image_bytes: 이미지 바이너리 데이터
            filename: 파일명
        """
        start_time = time.time()
        analysis_id = str(uuid.uuid4())
        layers_executed = []
        
        # ========== Layer 1: Hash Check ==========
        layer1_start = time.time()
        hash_data = self.hash_service.compute_hash(image_bytes)
        
        hash_result = HashResult(
            is_ai=hash_data["is_ai"],
            similarity=hash_data["similarity"],
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
            creation_date=metadata_data.get("creation_date"),
            exif_authenticity_score=metadata_data.get("exif_authenticity_score", 0.0),
            exif_inconsistencies=metadata_data.get("exif_inconsistencies", [])
        )
        layers_executed.append("metadata_analysis")
        layer2_time = (time.time() - layer2_start) * 1000
        
        # ========== Layer 3: AI Detection ==========
        detection_result = None
        layer3_time = 0
        
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
        - Hash: 30% (DinoV2 유사도, 점진적 계산)
        - Metadata: 40% (EXIF 진위성 + C2PA/시그니처)
        - AI Detection: 30% (HuggingFace 모델)

        Hash 점진적 계산:
        - 85% 이상: AI 점수 (강도에 비례)
        - 70-85%: 불확실 영역 (양쪽 점수 분배)
        - 70% 미만: Real 점수
        """
        scores = {
            "ai": 0.0,
            "real": 0.0
        }
        reasons = []
        
        # 1. Hash 기반 판정 (DinoV2 벡터 유사도) - 점진적 점수 계산
        similarity = hash_result.similarity

        if similarity >= 0.85:
            # 85% 이상: 확실한 AI 이미지 (임계값 이상)
            ai_score = self.HASH_WEIGHT * min((similarity - 0.85) / 0.15 + 0.5, 1.0)
            scores["ai"] += ai_score
            reasons.append(
                f"⚠️ AI 이미지 DB와 {'매칭됨' if hash_result.is_ai else '높은 유사도'} "
                f"(유사도: {similarity:.1%})"
            )
        elif similarity >= 0.70:
            # 70-85%: 불확실한 영역 (유사하지만 확신 부족)
            # 유사도에 비례하여 점수 분배
            uncertainty = (0.85 - similarity) / 0.15
            ai_portion = self.HASH_WEIGHT * 0.5 * (1 - uncertainty)
            real_portion = self.HASH_WEIGHT * 0.5 * uncertainty
            scores["ai"] += ai_portion
            scores["real"] += real_portion
            reasons.append(
                f"⚠️ AI 이미지 DB와 중간 유사도 "
                f"(유사도: {similarity:.1%}, 불확실)"
            )
        else:
            # 70% 미만: 실제 이미지 가능성
            real_score = self.HASH_WEIGHT * 0.5
            scores["real"] += real_score
            reasons.append(
                f"✓ AI 이미지 DB와 낮은 유사도 "
                f"(최대 유사도: {similarity:.1%})"
            )
        
        # 2. Metadata 기반 판정
        # 2-1. AI 도구 시그니처 (강력한 AI 증거)
        if metadata_result.ai_tool_signatures:
            tools = ", ".join(metadata_result.ai_tool_signatures)
            scores["ai"] += self.METADATA_WEIGHT * 0.4
            reasons.append(f"🔍 AI 도구 시그니처 발견: {tools}")

        # 2-2. C2PA 분석
        if metadata_result.has_c2pa:
            c2pa_info = metadata_result.c2pa_info or {}
            if c2pa_info.get("ai_related_assertions"):
                scores["ai"] += self.METADATA_WEIGHT * 0.2
                reasons.append("🤖 C2PA에 AI 생성 관련 정보 포함")
            else:
                # C2PA가 있지만 AI 관련 정보가 없으면 실제 이미지 가능성
                scores["real"] += self.METADATA_WEIGHT * 0.15
                reasons.append("📜 C2PA Content Credentials 존재 (AI 관련 정보 없음)")

        # 2-3. EXIF 진위성 점수 활용 (새로 추가된 핵심 기능)
        exif_score = metadata_result.exif_authenticity_score

        if exif_score >= 0.7:
            # 높은 EXIF 진위성 = 실제 카메라로 촬영
            scores["real"] += self.METADATA_WEIGHT * 0.35 * exif_score
            reasons.append(f"📷 EXIF 진위성 높음 (점수: {exif_score:.2f}) - 실제 카메라 촬영 가능성")
        elif exif_score >= 0.3:
            # 중간 수준
            scores["real"] += self.METADATA_WEIGHT * 0.15 * exif_score
            reasons.append(f"📷 EXIF 데이터 존재 (진위성: {exif_score:.2f})")
        else:
            # 낮은 EXIF 진위성 = AI 생성 의심
            scores["ai"] += self.METADATA_WEIGHT * 0.25
            reasons.append(f"⚠️ EXIF 진위성 낮음 (점수: {exif_score:.2f}) - AI 생성 의심")

        # 2-4. EXIF 비정상 패턴 탐지
        if metadata_result.exif_inconsistencies:
            inconsistency_weight = min(len(metadata_result.exif_inconsistencies) * 0.05, 0.15)
            scores["ai"] += self.METADATA_WEIGHT * inconsistency_weight
            inconsistency_msgs = {
                "editing_software_without_camera": "편집 SW만 존재",
                "perfect_square_ai_resolution": "AI 생성 해상도",
                "unrealistic_aperture": "비현실적 촬영값",
                "missing_datetime_original": "원본 시간 누락"
            }
            detected = [inconsistency_msgs.get(inc, inc) for inc in metadata_result.exif_inconsistencies]
            reasons.append(f"⚠️ EXIF 비정상 패턴: {', '.join(detected)}")
        
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
    

