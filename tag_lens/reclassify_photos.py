#!/usr/bin/env python3
"""재학습된 모델로 기존 사진들을 다시 분류 (genre, 태그, face_count 전부 갱신)"""

import sqlite3
from pathlib import Path

from ai.daynight_predict import predict_daynight
from ai.genre_predict import predict_genre
from ai.tagging import generate_tags
from database.database import replace_tags, update_classification
from opencv.face_detect import count_faces
from opencv.preprocess import load_image

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "tag_lens.db"
UPLOAD_DIR = BASE_DIR / "uploads"
GENRE_MODEL_PATH = BASE_DIR / "models" / "genre_model.h5"
DAYNIGHT_MODEL_PATH = BASE_DIR / "models" / "daynight_model.h5"


def reclassify_all_photos():
    """모든 사진을 새 모델로 재분류 (업로드 시점과 동일한 판별 로직 사용)"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    photos = conn.execute(
        "SELECT id, filename, genre FROM photo ORDER BY id"
    ).fetchall()
    conn.close()

    total = len(photos)
    if total == 0:
        print("✅ 사진이 없습니다!")
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
            image = load_image(str(image_path))

            face_count_value = count_faces(image)
            if face_count_value > 0:
                new_genre = "인물"
            else:
                new_genre, _ = predict_genre(image, str(GENRE_MODEL_PATH))

            daynight, _ = predict_daynight(image, str(DAYNIGHT_MODEL_PATH))
            tags = generate_tags(new_genre, daynight)

            update_classification(photo_id, new_genre, face_count_value)
            replace_tags(photo_id, tags)

            if new_genre == old_genre:
                status = "✅"
            else:
                status = f"🔄 ({old_genre} → {new_genre})"
                changed += 1

            print(f"[{idx}/{total}] {status} {filename} | {daynight} | {' '.join(tags)}")
            updated += 1
        except Exception as e:
            print(f"[{idx}/{total}] ❌ {filename} - {e}")

    print(f"\n✅ 완료: {updated}/{total}장 처리")
    if updated:
        print(f"📊 변경: {changed}장 ({changed / updated * 100:.1f}%)")


if __name__ == "__main__":
    print("=" * 60)
    print("TAG_LENS - 사진 재분류")
    print("=" * 60 + "\n")
    reclassify_all_photos()
