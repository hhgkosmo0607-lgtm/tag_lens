#!/usr/bin/env python3
"""재학습된 모델/판별 로직으로 기존 사진들을 다시 분류
(genre, daynight, indoor, animal, is_bw/color_tone/brightness/saturation, 태그, face_count 전부 갱신)"""

import sqlite3
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from ai.animal_predict import predict_animal
from ai.daynight_predict import extract_exif_hour, resolve_daynight
from ai.face_detect import count_faces
from ai.genre_predict import predict_genre, resolve_genre_with_animal
from ai.indoor_predict import INDOOR_CHECK_GENRES, predict_indoor
from ai.tagging import generate_tags
from database.database import replace_tags, update_classification
from opencv.feature_analyzer import analyze_features
from opencv.preprocess import load_image

DB_PATH = BASE_DIR / "tag_lens.db"
UPLOAD_DIR = BASE_DIR / "uploads"
GENRE_MODEL_PATH = BASE_DIR / "models" / "genre_model.h5"
DAYNIGHT_MODEL_PATH = BASE_DIR / "models" / "daynight_model.h5"
INDOOR_MODEL_PATH = BASE_DIR / "models" / "indoor_model.h5"


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
                genre_probs = {"자연": 0.0, "도시": 0.0, "음식": 0.0, "사물": 0.0, "인물": 1.0}
            else:
                new_genre, genre_probs = predict_genre(image, str(GENRE_MODEL_PATH))

            animal = predict_animal(image)
            new_genre = resolve_genre_with_animal(new_genre, animal, face_count_value)

            features = analyze_features(image)
            exif_hour, raw_exif_hour = extract_exif_hour(str(image_path))
            daynight, daynight_probs = resolve_daynight(image, exif_hour, str(DAYNIGHT_MODEL_PATH), raw_exif_hour)
            if new_genre in INDOOR_CHECK_GENRES:
                indoor, _ = predict_indoor(image, str(INDOOR_MODEL_PATH))
            else:
                indoor = None
            tags = generate_tags(new_genre, daynight, features["is_bw"], indoor, features["color_tone"], animal)

            update_classification(
                photo_id,
                new_genre,
                face_count_value,
                genre_probs,
                daynight_probs,
                is_bw=features["is_bw"],
                color_tone=features["color_tone"],
                brightness=features["brightness"],
                saturation=features["saturation"],
            )
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
