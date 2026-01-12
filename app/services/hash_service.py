"""
Layer 1: Hash Service
이미지 해시 계산 및 중복 검사
"""

import hashlib
import io
from typing import Dict, Optional
from PIL import Image
import imagehash


class HashService:
    """이미지 해시 계산 및 중복 검사 서비스"""
    
    def compute_hash(self, image_bytes: bytes) -> Dict[str, str]:
        """
        이미지의 여러 해시값 계산
        
        Returns:
            - md5: 정확한 중복 검사용
            - sha256: 무결성 검증용
            - perceptual_hash: 유사 이미지 검사용 (리사이즈/압축에 강함)
        """
        # MD5 해시
        md5_hash = hashlib.md5(image_bytes).hexdigest()
        
        # SHA256 해시
        sha256_hash = hashlib.sha256(image_bytes).hexdigest()
        
        # Perceptual Hash (이미지 유사도 검사)
        perceptual = None
        try:
            img = Image.open(io.BytesIO(image_bytes))
            perceptual = str(imagehash.phash(img))
        except Exception:
            pass
        
        return {
            "md5": md5_hash,
            "sha256": sha256_hash,
            "perceptual_hash": perceptual
        }
    
    async def check_duplicate(self, md5_hash: str) -> bool:
        """
        중복 체크 비활성화 (DB 없음)
        항상 False 반환
        """
        return False
    
    async def check_similar(self, perceptual_hash: str, threshold: int = 10) -> Optional[Dict]:
        """
        유사 이미지 체크 비활성화 (DB 없음)
        항상 None 반환
        """
        return None
