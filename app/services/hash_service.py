"""
Layer 1: Hash Service
이미지 해시 계산 및 중복 검사
"""

import io
from typing import Dict, Optional, Tuple
from PIL import Image
import numpy as np
import pandas as pd
import torch
from transformers import AutoImageProcessor, AutoModel

class HashService:
    """이미지 해시 계산 및 중복 검사 서비스"""

    def __init__(self, db_vectors_path: str = './data/ai_dinohashes.npy',
                 metadata_path: str = './data/ai_metadata.csv',
                 threshold: float = 0.85):
        """
        HashService 

        Args:
            db_vectors_path: AI 이미지 벡터 파일 경로
            metadata_path: AI 이미지 메타데이터 파일 경로
            threshold: 유사도 임계값 (0~1)
        """
        # DinoV2 모델 로드
        self.model_name = "facebook/dinov2-small"
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()

        # GPU 사용 가능 시 GPU로 이동
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # DB 벡터 및 메타데이터 로드
        self.db_vectors = np.load(db_vectors_path)
        self.metadata = pd.read_csv(metadata_path)
        self.threshold = threshold

    def _extract_features(self, image: Image.Image) -> np.ndarray:
        """
        DinoV2 모델을 사용하여 이미지에서 특징 벡터 추출

        Args:
            image: PIL Image 객체

        Returns:
            특징 벡터 (numpy array)
        """
        # 이미지 전처리
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 특징 추출 (no gradient 계산)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # CLS 토큰의 출력 사용
            features = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return features.flatten()

    def _compute_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        두 벡터 간의 코사인 유사도 계산

        Args:
            vec1: 벡터 1
            vec2: 벡터 2

        Returns:
            코사인 유사도 (0~1)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2)

    def find_similar_image(self, image_vector: np.ndarray) -> Tuple[Optional[int], float]:
        """
        DB에서 가장 유사한 이미지 찾기

        Args:
            image_vector: 검색할 이미지의 벡터

        Returns:
            (인덱스, 유사도) 튜플. 유사한 이미지가 없으면 (None, 0.0)
        """
        max_similarity = 0.0
        max_idx = None

        for idx, db_vector in enumerate(self.db_vectors):
            similarity = self._compute_cosine_similarity(image_vector, db_vector)
            if similarity > max_similarity:
                max_similarity = similarity
                max_idx = idx

        if max_similarity >= self.threshold:
            return max_idx, max_similarity
        return None, max_similarity

    def compute_hash(self, image_bytes: bytes) -> Dict[str, any]:
        """
        이미지의 dinohash 계산 및 AI 이미지 여부 판단

        Args:
            image_bytes: 이미지 바이트 데이터

        Returns:
            - is_ai: AI 이미지 여부
            - similarity: 최대 유사도 점수
        """
        # 이미지 로드
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # DinoV2로 특징 벡터 추출
        image_vector = self._extract_features(image)

        # DB에서 유사한 이미지 찾기
        matched_idx, similarity = self.find_similar_image(image_vector)

        result = {
            "is_ai": matched_idx is not None,
            "similarity": float(similarity),
        }

        return result
    