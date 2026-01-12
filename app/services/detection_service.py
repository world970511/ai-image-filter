"""
Layer 3: AI Detection Service
HuggingFace 모델 기반 AI 이미지 탐지
"""

import io
import asyncio
from typing import Dict, Any, Optional
from PIL import Image
import torch
from transformers import pipeline, AutoModelForImageClassification, AutoFeatureExtractor
from functools import lru_cache


class DetectionService:
    """AI 생성 이미지 탐지 서비스"""
    
    # 사용 가능한 모델 목록
    AVAILABLE_MODELS = {
        "umm-maybe/AI-image-detector": {
            "description": "AI vs Real image classifier",
            "labels": {"artificial": "ai", "human": "real"}
        },
        "Organika/sdxl-detector": {
            "description": "SDXL generated image detector",
            "labels": {"artificial": "ai", "real": "real"}
        }
    }
    
    DEFAULT_MODEL = "umm-maybe/AI-image-detector"
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self._classifier = None
        self._model_loaded = False
    
    @property
    def classifier(self):
        """Lazy loading of the classifier"""
        if self._classifier is None:
            self._load_model()
        return self._classifier
    
    def _load_model(self):
        """모델 로드 (최초 호출 시)"""
        try:
            print(f"🔄 Loading model: {self.model_name}")
            
            # GPU 사용 가능 여부 확인
            device = 0 if torch.cuda.is_available() else -1
            
            self._classifier = pipeline(
                "image-classification",
                model=self.model_name,
                device=device
            )
            
            self._model_loaded = True
            print(f"✅ Model loaded successfully on {'GPU' if device == 0 else 'CPU'}")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    async def detect(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        이미지의 AI 생성 여부 탐지
        
        Returns:
            - is_ai_generated: AI 생성 여부
            - confidence: 확신도 (0.0 ~ 1.0)
            - raw_scores: 원본 점수
        """
        try:
            # 이미지 로드
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # 추론 실행 (비동기로 실행)
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, 
                lambda: self.classifier(img)
            )
            
            # 결과 파싱
            return self._parse_results(results)
            
        except Exception as e:
            return {
                "model_name": self.model_name,
                "is_ai_generated": False,
                "confidence": 0.0,
                "error": str(e),
                "raw_scores": None
            }
    
    def _parse_results(self, results: list) -> Dict[str, Any]:
        """모델 결과 파싱"""
        raw_scores = {r["label"]: r["score"] for r in results}
        
        # 모델별 라벨 매핑
        model_config = self.AVAILABLE_MODELS.get(self.model_name, {})
        label_map = model_config.get("labels", {"artificial": "ai", "human": "real"})
        
        # AI 관련 라벨 점수 합산
        ai_score = 0.0
        real_score = 0.0
        
        for label, score in raw_scores.items():
            label_lower = label.lower()
            
            # AI 관련 라벨
            if any(ai_key in label_lower for ai_key in ["artificial", "ai", "fake", "generated", "synthetic"]):
                ai_score += score
            # Real 관련 라벨
            elif any(real_key in label_lower for real_key in ["human", "real", "authentic", "natural"]):
                real_score += score
        
        # 판정
        is_ai = ai_score > real_score
        confidence = ai_score if is_ai else real_score
        
        return {
            "model_name": self.model_name,
            "is_ai_generated": is_ai,
            "confidence": round(confidence, 4),
            "raw_scores": raw_scores
        }
    
    async def detect_with_multiple_models(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        여러 모델로 앙상블 탐지 (더 정확한 결과)
        """
        results = {}
        
        for model_name in self.AVAILABLE_MODELS.keys():
            try:
                temp_detector = DetectionService(model_name)
                result = await temp_detector.detect(image_bytes)
                results[model_name] = result
            except Exception as e:
                results[model_name] = {"error": str(e)}
        
        # 앙상블 결과 계산
        ai_votes = 0
        total_confidence = 0.0
        valid_count = 0
        
        for model_name, result in results.items():
            if "error" not in result:
                valid_count += 1
                total_confidence += result["confidence"]
                if result["is_ai_generated"]:
                    ai_votes += 1
        
        ensemble_result = {
            "individual_results": results,
            "ensemble_verdict": ai_votes > valid_count / 2 if valid_count > 0 else False,
            "ensemble_confidence": total_confidence / valid_count if valid_count > 0 else 0.0,
            "models_used": valid_count
        }
        
        return ensemble_result
    
    def get_model_info(self) -> Dict[str, Any]:
        """현재 모델 정보 반환"""
        return {
            "model_name": self.model_name,
            "model_loaded": self._model_loaded,
            "available_models": list(self.AVAILABLE_MODELS.keys()),
            "device": "GPU" if torch.cuda.is_available() else "CPU"
        }
