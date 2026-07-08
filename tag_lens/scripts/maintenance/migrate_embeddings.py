#!/usr/bin/env python3
"""기존 사진들에 CNN 특징 벡터(embedding) 생성 및 저장"""

import json
import sqlite3
from pathlib import Path

import numpy as np

from ai.genre_predict import extract_features
from opencv.preprocess import load_image

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "tag_lens.db"
UPLOAD_DIR = BASE_DIR / "uploads"


def migrate_embeddings():
    """기존 모든 사진에 embedding 추가"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    # embedding이 없는 사진들 조회
    photos = conn.execute(
        "SELECT id, filename FROM photo WHERE embedding IS NULL ORDER BY id"
    ).fetchall()
    
    total = len(photos)
    if total == 0:
        print("✅ 모든 사진에 embedding이 있습니다!")
        conn.close()
        return
    
    print(f"📊 처리할 사진: {total}장\n")
    
    processed = 0
    for idx, photo in enumerate(photos, 1):
        photo_id = photo["id"]
        filename = photo["filename"]
        image_path = UPLOAD_DIR / filename
        
        if not image_path.exists():
            print(f"[{idx}/{total}] ⚠️  {filename} - 파일 없음 (건너뜀)")
            continue
        
        try:
            # 이미지 로드 및 특징 추출
            image = load_image(str(image_path))
            embedding = extract_features(image, str(BASE_DIR / "models" / "genre_model.h5"))
            embedding_json = json.dumps(embedding.tolist())
            
            # 데이터베이스 업데이트
            conn.execute(
                "UPDATE photo SET embedding = ? WHERE id = ?",
                (embedding_json, photo_id)
            )
            conn.commit()
            
            print(f"[{idx}/{total}] ✅ {filename} (ID: {photo_id})")
            processed += 1
        except Exception as e:
            print(f"[{idx}/{total}] ❌ {filename} - {e}")
    
    conn.close()
    print(f"\n✅ 완료: {processed}/{total}장 처리됨")


if __name__ == "__main__":
    print("🔄 Embedding 마이그레이션 시작...\n")
    migrate_embeddings()
