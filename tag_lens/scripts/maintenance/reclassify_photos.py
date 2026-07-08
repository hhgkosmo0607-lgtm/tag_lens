#!/usr/bin/env python3
"""재학습된 모델/판별 로직으로 기존 사진들을 다시 분류
(genre, daynight, indoor, animal, is_bw/color_tone/brightness/saturation, 태그, face_count 전부 갱신)"""

import sqlite3
import sys
from pathlib import Path

from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from ai.animal_predict import predict_animal
from ai.daynight_predict import resolve_daynight
from ai.face_detect import count_faces
from ai.genre_predict import predict_genre, resolve_genre_with_animal
from ai.indoor_predict import predict_indoor
from ai.tagging import generate_tags
from database.database import replace_tags, update_classification
from opencv.feature_analyzer import analyze_features
from opencv.preprocess import load_image


def extract_exif_hour(image_path: str) -> int | None:
    """EXIF 메타데이터에서 촬영 시각(0~23시)을 추출한다. 없으면 None. (app.py와 동일 로직)

    DateTimeOriginal(36867)은 최상위 IFD0가 아니라 Exif SubIFD에 있다.
    getexif()만 보면 못 찾고 IFD0의 DateTime(306, 편집 프로그램이 내보내기 시각으로
    덮어썼을 수 있음)으로 잘못 폴백하게 되므로 SubIFD를 먼저 확인해야 한다."""
    try:
        image = Image.open(image_path)
        exif_data = image.getexif()

        try:
            from PIL.ExifTags import IFD

            exif_ifd = exif_data.get_ifd(IFD.Exif)
        except Exception:
            exif_ifd = {}

        date_str = None
        if 36867 in exif_ifd:  # DateTimeOriginal (SubIFD)
            date_str = exif_ifd[36867]
        elif 36868 in exif_ifd:  # DateTimeDigitized (SubIFD)
            date_str = exif_ifd[36868]
        elif 306 in exif_data:  # DateTime (IFD0) — 최후 폴백
            date_str = exif_data[306]

        if date_str:
            time_part = date_str.split(" ")[1]
            return int(time_part.split(":")[0])
    except Exception:
        pass
    return None

DB_PATH = BASE_DIR / "tag_lens.db"
UPLOAD_DIR = BASE_DIR / "uploads"
GENRE_MODEL_PATH = BASE_DIR / "models" / "genre_model.h5"
DAYNIGHT_MODEL_PATH = BASE_DIR / "models" / "daynight_model.h5"
INDOOR_MODEL_PATH = BASE_DIR / "models" / "indoor_model.h5"
INDOOR_CHECK_GENRES = {"자연", "도시", "인물"}


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
            exif_hour = extract_exif_hour(str(image_path))
            daynight, daynight_probs = resolve_daynight(image, exif_hour, str(DAYNIGHT_MODEL_PATH))
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
