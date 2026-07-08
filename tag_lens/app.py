from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from flask import Flask, flash, redirect, render_template, request, send_from_directory, session, url_for
from werkzeug.utils import secure_filename
from PIL import Image

from ai.animal_predict import predict_animal
from ai.daynight_predict import resolve_daynight
from ai.embedding import extract_features
from ai.face_detect import count_faces
from ai.genre_predict import predict_genre, resolve_genre_with_animal
from ai.indoor_predict import predict_indoor
from ai.tagging import generate_tags
from database.database import (
    delete_photo,
    find_duplicate_by_hash,
    find_similar,
    get_photo,
    get_photos,
    get_tag_counts,
    init_db,
    insert_photo,
    link_tags,
    update_rating,
)
from hash.average_hash import create_average_hash
from opencv.feature_analyzer import analyze_features
from opencv.preprocess import load_image

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

# 실내/실외 판별은 배경이 어느 정도 보이는 장르에만 적용한다.
# 사물/음식은 클로즈업 정물 위주라 판단 근거(벽/천장 등 공간 맥락)가 사진에 거의 없음.
INDOOR_CHECK_GENRES = {"자연", "도시", "인물"}

app = Flask(__name__)
app.secret_key = "tag-lens-dev-secret"


def _get_exif_datetime_str(image_path: str) -> str | None:
    """
    EXIF에서 "촬영 시각" 문자열("YYYY:MM:DD HH:MM:SS")을 찾는다.

    DateTimeOriginal(36867)은 최상위 IFD0가 아니라 **Exif SubIFD**(태그 34665로
    가리키는 중첩 구조)에 들어있다. image.getexif()는 IFD0만 반환하므로 거기서
    36867을 찾으면 항상 실패하고, 최상위의 DateTime(306)으로 폴백하게 되는데
    — 이 306은 Lightroom/포토샵 같은 편집 프로그램이 "내보내기한 시각"으로
    덮어써버리는 경우가 많다(실제 촬영 시각이 아님). 그래서 SubIFD를 먼저 봐야 한다.

    우선순위: SubIFD의 DateTimeOriginal → SubIFD의 DateTimeDigitized → IFD0의 DateTime
    """
    try:
        image = Image.open(image_path)
        exif_data = image.getexif()

        try:
            from PIL.ExifTags import IFD

            exif_ifd = exif_data.get_ifd(IFD.Exif)
        except Exception:
            exif_ifd = {}

        if 36867 in exif_ifd:  # DateTimeOriginal (SubIFD)
            return exif_ifd[36867]
        if 36868 in exif_ifd:  # DateTimeDigitized (SubIFD)
            return exif_ifd[36868]
        if 306 in exif_data:  # DateTime (IFD0) — 편집 프로그램이 덮어썼을 수 있는 최후 폴백
            return exif_data[306]
    except Exception:
        pass

    return None


def extract_photo_date(image_path: str) -> str:
    """
    EXIF 메타데이터에서 촬영 날짜를 추출합니다.
    메타데이터가 없으면 현재 날짜를 반환합니다.
    """
    date_str = _get_exif_datetime_str(image_path)
    if date_str:
        try:
            # EXIF 포맷: "YYYY:MM:DD HH:MM:SS" → "YYYY-MM-DD"로 변환
            date_obj = datetime.strptime(date_str.split(" ")[0], "%Y:%m:%d")
            return date_obj.strftime("%Y-%m-%d")
        except Exception:
            pass

    return datetime.now().strftime("%Y-%m-%d")


def extract_exif_hour(image_path: str) -> int | None:
    """
    EXIF 메타데이터에서 촬영 시각(0~23시)을 추출합니다. 없으면 None.
    주간/야간 판별에 씁니다 — 픽셀 밝기를 CNN으로 추정하는 것보다 실제 촬영 시각이
    훨씬 신뢰도 높은 신호이고, 흑백 사진에도 적용 가능합니다(색 정보가 필요 없음).
    """
    date_str = _get_exif_datetime_str(image_path)
    if date_str:
        try:
            time_part = date_str.split(" ")[1]
            return int(time_part.split(":")[0])
        except Exception:
            pass

    return None


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _probs_to_percent(raw_probs_json: str | None) -> dict[str, float]:
    """DB에 저장된 확률 JSON(0~1 값)을 퍼센트 dict로 변환. 저장된 값이 없으면 빈 dict."""
    if not raw_probs_json:
        return {}
    probs = json.loads(raw_probs_json)
    return {k: round(v * 100, 2) for k, v in probs.items()}


@app.before_request
def bootstrap_db() -> None:
    if "db_initialized" not in session:
        init_db()
        session["db_initialized"] = True


@app.route("/")
def home():
    recent_photos = get_photos()[:24]
    return render_template("home.html", photos=recent_photos)


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "GET":
        return render_template("upload.html")

    files = request.files.getlist("photo")
    if not files or all(f.filename == "" for f in files):
        flash("업로드할 이미지를 선택하세요.", "error")
        return redirect(url_for("upload"))

    success_count = 0
    error_count = 0
    first_photo_id = None

    for file in files:
        if file.filename == "":
            continue

        if not allowed_file(file.filename):
            flash(f"❌ {file.filename} - 지원하지 않는 파일 형식입니다. (png/jpg/jpeg/webp)", "error")
            error_count += 1
            continue

        ext = file.filename.rsplit(".", 1)[1].lower()
        saved_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}.{ext}"
        safe_name = secure_filename(saved_name)
        save_path = UPLOAD_DIR / safe_name
        file.save(save_path)

        try:
            image = load_image(str(save_path))

            face_count_value = count_faces(image)
            if face_count_value > 0:
                # 얼굴 감지되면 인물 사진으로 분류 (CNN 무시)
                genre = "인물"
                probabilities = {"자연": 0.0, "도시": 0.0, "음식": 0.0, "사물": 0.0, "인물": 1.0}
            else:
                # 얼굴 없으면 CNN으로 5-class 분류 (자연/도시/음식/사물/인물)
                genre, probabilities = predict_genre(image, str(BASE_DIR / "models" / "genre_model.h5"))

            animal = predict_animal(image)
            # 장르 CNN이 동물 클로즈업(특히 고양이)을 사람 얼굴로 착각하는 사례가
            # 있어(resolve_genre_with_animal 참고), 사람 얼굴이 감지 안 됐는데
            # 동물로 판별됐다면 "인물" 판정을 "기타"로 보정
            genre = resolve_genre_with_animal(genre, animal, face_count_value)

            # 주간/야간 최종 판정: 흑백이면 판별 안 함, EXIF 있으면 EXIF 우선,
            # 없으면 CNN(또는 밝기 휴리스틱) 폴백 (resolve_daynight 참고)
            exif_hour = extract_exif_hour(str(save_path))
            daynight, daynight_probs = resolve_daynight(image, exif_hour, str(BASE_DIR / "models" / "daynight_model.h5"))

            if genre in INDOOR_CHECK_GENRES:
                indoor, _indoor_probs = predict_indoor(image, str(BASE_DIR / "models" / "indoor_model.h5"))
            else:
                indoor = None

            features = analyze_features(image)
            image_hash = create_average_hash(image)

            # 중복 사진 확인
            duplicate = find_duplicate_by_hash(image_hash)
            if duplicate:
                flash(f"⚠️ '{duplicate['filename']}' - 이미 업로드되었습니다. (중복 감지)", "warning")
                if save_path.exists():
                    save_path.unlink(missing_ok=True)
                error_count += 1
                continue

            embedding = extract_features(image)
            tags = generate_tags(genre, daynight, features["is_bw"], indoor, features["color_tone"], animal)

            photo_id = insert_photo(
                {
                    "filename": safe_name,
                    "upload_date": extract_photo_date(str(save_path)),
                    "genre": genre,
                    "brightness": features["brightness"],
                    "color_tone": features["color_tone"],
                    "saturation": features["saturation"],
                    "is_bw": features["is_bw"],
                    "rating": 0,
                    "hash": image_hash,
                    "face_count": face_count_value,
                    "embedding": embedding,
                    "genre_probs": probabilities,
                    "daynight_probs": daynight_probs,
                }
            )
            link_tags(photo_id, tags)

            if first_photo_id is None:
                first_photo_id = photo_id

            success_count += 1
        except Exception as exc:
            print(f"[ERROR] {file.filename}: {type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()
            if save_path.exists():
                save_path.unlink(missing_ok=True)
            flash(f"❌ {file.filename} - 분석 실패: {exc}", "error")
            error_count += 1

    if success_count > 0:
        flash(f"✅ {success_count}장의 사진을 업로드했습니다!", "success")
        return redirect(url_for("result", photo_id=first_photo_id))
    else:
        flash("업로드된 사진이 없습니다.", "error")
        return redirect(url_for("upload"))


@app.route("/result/<int:photo_id>")
def result(photo_id: int):
    photo = get_photo(photo_id)
    if photo is None:
        return render_template("404.html"), 404

    probabilities = _probs_to_percent(photo.get("genre_probs"))

    similar_photos = find_similar(photo.get("embedding"), exclude_photo_id=photo_id, top_k=3)
    return render_template(
        "result.html",
        photo=photo,
        probabilities=probabilities,
        similar_photos=similar_photos,
    )


@app.route("/gallery")
def gallery():
    tags = request.args.getlist("tag")
    rating_min = int(request.args.get("rating_min", 0) or 0)
    date_from = request.args.get("date_from", "")
    date_to = request.args.get("date_to", "")

    photos = get_photos(tags=tags, rating_min=rating_min, date_from=date_from, date_to=date_to)
    tag_counts = get_tag_counts()

    return render_template(
        "gallery.html",
        photos=photos,
        tag_counts=tag_counts,
        selected_tags=tags,
        rating_min=rating_min,
        date_from=date_from,
        date_to=date_to,
    )


@app.route("/tags")
def tag_view():
    tag_counts = get_tag_counts()
    return render_template("tag_view.html", tag_counts=tag_counts)


@app.route("/photo/<int:photo_id>/rate", methods=["POST"])
def rate_photo(photo_id: int):
    rating = int(request.form.get("rating", 0))
    rating = max(0, min(5, rating))
    update_rating(photo_id, rating)

    next_url = request.form.get("next")
    if next_url:
        return redirect(next_url)
    return redirect(url_for("gallery"))


@app.route("/photo/<int:photo_id>/similar")
def similar(photo_id: int):
    photo = get_photo(photo_id)
    if photo is None:
        return render_template("404.html"), 404

    similar_photos = find_similar(photo.get("embedding"), exclude_photo_id=photo_id, top_k=10)
    return render_template("similar.html", photo=photo, similar_photos=similar_photos)


@app.route("/photo/<int:photo_id>/delete", methods=["POST"])
def remove_photo(photo_id: int):
    photo = get_photo(photo_id)
    if photo and (UPLOAD_DIR / photo["filename"]).exists():
        (UPLOAD_DIR / photo["filename"]).unlink(missing_ok=True)
    delete_photo(photo_id)
    flash("사진이 삭제되었습니다.", "ok")
    return redirect(url_for("gallery"))


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/uploads/<path:filename>")
def uploads_proxy(filename: str):
    return send_from_directory(UPLOAD_DIR, filename)


@app.errorhandler(404)
def not_found(_error):
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(_error):
    return render_template("500.html"), 500


if __name__ == "__main__":
    UPLOAD_DIR.mkdir(exist_ok=True)
    init_db()
    app.run(debug=True)
