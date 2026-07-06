# PhotoSense AI

AI 기반 사진 장르 분석 및 태그 기반 스마트 갤러리 웹 서비스입니다.

## 주요 기능

- 사진 업로드 및 저장
- 하이브리드 장르 분류
  - 1단계: OpenCV 얼굴 검출 (얼굴 있으면 `인물`)
  - 2단계: 얼굴이 없을 때 CNN 3-class (`풍경/도시/음식`)
- OpenCV 특징 분석 (밝기/색감/채도/흑백)
- 자동 태그 생성 (다중 태그)
- 태그 AND 필터 + 별점 + 날짜 필터
- Average Hash 유사 사진 검색
- 사진 별점(0~5) 및 삭제

## 폴더 구조

```text
PhotoSenseAI/
│ app.py
│ requirements.txt
│ README.md
│ photosense.db (실행 후 생성)
│
├── database/
│      database.py
│
├── templates/
│      base.html
│      home.html
│      upload.html
│      result.html
│      gallery.html
│      tag_view.html
│      similar.html
│      about.html
│      404.html
│      500.html
│
├── static/
│   └── css/
│      style.css
│
├── uploads/
├── models/
│      genre_model.h5   (선택: 있으면 CNN 사용)
│
├── ai/
│      genre_predict.py
│
├── opencv/
│      preprocess.py
│      feature_analyzer.py
│      face_detect.py
│
└── hash/
       average_hash.py
```

## 실행 방법

1. 프로젝트 폴더 이동

```bash
cd PhotoSenseAI
```

2. 패키지 설치

```bash
pip install -r requirements.txt
```

3. 실행

```bash
python app.py
```

4. 접속

- http://127.0.0.1:5000

## CNN 모델 관련

- `models/genre_model.h5` 파일이 있으면 해당 모델로 `풍경/도시/음식` 분류를 수행합니다.
- 모델이 없거나 로드 실패 시, 데모용 휴리스틱 분류로 자동 폴백됩니다.

## 데이터베이스

SQLite(`photosense.db`)를 사용하며, 앱 실행 시 자동으로 테이블이 생성됩니다.

- `photo`
- `tag`
- `photo_tag`
