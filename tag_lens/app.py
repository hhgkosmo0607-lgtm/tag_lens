from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from flask import Flask, flash, redirect, render_template, request, send_from_directory, session, url_for
from werkzeug.utils import secure_filename
from PIL import Image

from ai.genre_predict import predict_genre, extract_features
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
from opencv.face_detect import count_faces
from opencv.feature_analyzer import analyze_features
from opencv.preprocess import load_image

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

app = Flask(__name__)
app.secret_key = "tag-lens-dev-secret"


def extract_photo_date(image_path: str) -> str:
    """
    EXIF 메타데이터에서 촬영 날짜를 추출합니다.
    메타데이터가 없으면 현재 날짜를 반환합니다.
    """
    try:
        image = Image.open(image_path)
        exif_data = image.getexif()
        
        # EXIF 태그: 306=DateTime, 36867=DateTimeOriginal, 36868=DateTimeDigitized
        # DateTimeOriginal(36867)을 우선으로, 없으면 DateTime(306)을 사용
        date_str = None
        if 36867 in exif_data:  # DateTimeOriginal
            date_str = exif_data[36867]
        elif 306 in exif_data:  # DateTime
            date_str = exif_data[306]
        
        if date_str:
            # EXIF 포맷: "YYYY:MM:DD HH:MM:SS" → "YYYY-MM-DD"로 변환
            date_obj = datetime.strptime(date_str.split(" ")[0], "%Y:%m:%d")
            return date_obj.strftime("%Y-%m-%d")
    except Exception:
        pass
    
    # 메타데이터 없거나 오류 발생 시 현재 날짜 반환
    return datetime.now().strftime("%Y-%m-%d")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_tags(genre: str, features: dict[str, object], face_count_value: int) -> list[str]:
    """
    간단하고 일관성 있는 태그 생성
    - 장르 기반 태그만 사용 (일관성)
    - 특징(밝기, 채도, 색상)은 metadata로 표시 (분리)
    """
    tags: set[str] = set()

    # 장르 태그 (기본)
    genre_map = {
        "풍경": ["#풍경"],
        "도시": ["#도시"],
        "음식": ["#음식"],
        "사물": ["#사물"],
        "인물": ["#인물"],
    }
    tags.update(genre_map.get(genre, []))

    # 선택적 추가 태그 (도시 야경만)
    if genre == "도시" and features["brightness_label"] == "어두운":
        tags.add("#야경")

    return sorted(tags)


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
                probabilities = {"풍경": 0.0, "도시": 0.0, "음식": 0.0, "사물": 0.0}
            else:
                # 얼굴 없으면 CNN으로 4-class 분류 (풍경/도시/음식/사물)
                genre, probabilities = predict_genre(image, str(BASE_DIR / "models" / "genre_model.h5"))

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
            
            embedding = extract_features(image, str(BASE_DIR / "models" / "genre_model.h5"))
            tags = generate_tags(genre, features, face_count_value)

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
                }
            )
            link_tags(photo_id, tags)

            if first_photo_id is None:
                first_photo_id = photo_id
                session["last_prediction"] = {
                    "photo_id": photo_id,
                    "probabilities": {k: round(v * 100, 2) for k, v in probabilities.items()},
                }

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

    prediction = session.get("last_prediction", {})
    probabilities = {}
    if prediction.get("photo_id") == photo_id:
        probabilities = prediction.get("probabilities", {})

    # embedding이 있으면 사용, 없으면 해시 사용 (하위 호환성)
    similar_search_key = photo.get("embedding") or photo.get("hash")
    similar_photos = find_similar(similar_search_key, exclude_photo_id=photo_id, top_k=3)
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

    similar_search_key = photo.get("embedding") or photo.get("hash")
    similar_photos = find_similar(similar_search_key, exclude_photo_id=photo_id, top_k=10)
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
