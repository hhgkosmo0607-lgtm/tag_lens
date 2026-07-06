#!/usr/bin/env python3
"""재학습된 모델로 기존 사진들을 다시 분류"""

import sqlite3
from pathlib import Path

from ai.genre_predict import predict_genre
from opencv.preprocess import load_image

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "tag_lens.db"
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_PATH = BASE_DIR / "models" / "genre_model.h5"


def reclassify_all_photos():
    """모든 사진을 새 모델로 재분류"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    photos = conn.execute(
        "SELECT id, filename, genre FROM photo ORDER BY id"
    ).fetchall()
    
    total = len(photos)
    if total == 0:
        print("✅ 사진이 없습니다!")
        conn.close()
        return
    
    print(f"🔄 {total}장 재분류 시작...\n")
    
    updated = 0
    changed = 0
    for idx, photo in enumerate(photos, 1):
        photo_id = photo["id"]
        filename = photo["filename"]
        old_genre = photo["genre"]
        image_path = UPLOAD_DIR / filename
        
        if not image_path.exists():
            print(f"[{idx}/{total}] ⚠️  {filename} - 파일 없음")
            continue
        
        try:
            # 이미지 로드 및 새 모델로 예측
            image = load_image(str(image_path))
            new_genre, probs = predict_genre(image, str(MODEL_PATH))
            
            # DB 업데이트
            conn.execute(
                "UPDATE photo SET genre = ? WHERE id = ?",
                (new_genre, photo_id)
            )
            conn.commit()
            
            if new_genre == old_genre:
                status = "✅"
            else:
                status = f"🔄 ({old_genre} → {new_genre})"
                changed += 1
            
            prob_str = ", ".join([f"{g}: {p:.1%}" for g, p in probs.items()])
            print(f"[{idx}/{total}] {status} {filename} | {prob_str}")
            updated += 1
        except Exception as e:
            print(f"[{idx}/{total}] ❌ {filename} - {e}")
    
    conn.close()
    print(f"\n✅ 완료: {updated}/{total}장 처리")
    print(f"📊 변경: {changed}장 ({changed/updated*100:.1f}%)")


if __name__ == "__main__":
    print("=" * 60)
    print("TAG_LENS - 사진 재분류")
    print("=" * 60 + "\n")
    reclassify_all_photos()
