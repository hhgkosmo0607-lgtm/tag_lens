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

######################################
tag_lens는 이미지 분석을 3단계로 수행한다.

1. Image Attribute Analysis (OpenCV)
   - 흑백/컬러
   - 밝기
   - 야간/주간
   - 실내/실외
   - 채도/대비

2. Object Detection (OpenCV / ML)
   - 얼굴 검출 → 인물/비인물

3. Semantic Classification (CNN)
   - 풍경 / 도시 / 음식 / 자연 / 사물

4. Sub-classification (optional)
   - 음식 → 세부 음식 분류
   - 자연 → 세부 자연 분류

클래스 불균형 문제는 Class Weight를 통해 해결한다.

######################################
## 세션 작업 로그 (2026-07-06) — 다음에 이어갈 것

### 이번 세션에서 완료한 것

- **원인 진단**: "사물" 클래스가 학습 데이터 0장으로 비어있었고(`prepare_dataset.py`의 Caltech-101 수집 실패), CNN에 "인물" 클래스 자체가 없어서 얼굴 검출(Haar Cascade) 실패 시 안전망이 없었음.
- **얼굴 검출기 교체**: `opencv/face_detect.py` — Haar Cascade → mediapipe(BlazeFace) 기반으로 교체. 측면/각도/작은 얼굴도 더 잘 잡음. `requirements.txt`에 `mediapipe` 추가.
- **데이터셋 5-class로 확장** (`prepare_dataset.py`):
  - 사물: Caltech-101 카테고리 재수집 → 1066장 확보
  - 인물: Caltech Faces/Faces_easy(870장, 다양성 낮음) + `ashwingupta3012/human-faces`(7219장) 조합 → 2000장 샘플링
  - 최종: 풍경 2000 / 도시 938 / 음식 2000 / 사물 1066 / 인물 2000, train:val = 8:2
- **주간/야간 이진 분류기 신규 추가**:
  - `prepare_daynight_dataset.py` (Kaggle `ibrahimalobaid/day-and-night-image`, 200장/200장)
  - `train_daynight_model.py` → `models/daynight_model.h5` (**검증 정확도 98.75%**, 학습 완료)
  - `ai/daynight_predict.py`
- **코드 정리**:
  - `ai/tagging.py` 신규 — `generate_tags()`를 공용화해서 `app.py`, `reclassify_photos.py`가 같이 씀
  - `database/database.py` — `replace_tags()`, `update_classification()` 추가
  - `reclassify_photos.py` 재작성 — 얼굴검출→5-class CNN→주간/야간→태그 교체까지 `app.py`와 동일 로직으로 맞춤
  - `app.py` — 주간/야간 예측 연동, 태그에 `#주간`/`#야간`/`#야경` 반영
  - `templates/about.html` 문구를 5-class + mediapipe 기준으로 갱신
- **테스트 데이터 초기화**: `uploads/`의 테스트 사진 7장 + DB(`photo`/`tag`/`photo_tag`) 전부 삭제, 클린 상태로 만듦

### 지금 진행 중인 것 (미완료 — 다음에 이어갈 것)

1. **genre CNN(5-class) 1단계 학습 완료, 의도적으로 여기서 멈춰둔 상태.**
   - 1단계(base 동결, 15 epoch) 완료 — best는 epoch 14, **val_accuracy 99.72%, val_loss 0.0125**로 `models/genre_model.h5`에 저장됨
   - 2단계(fine-tuning) 진입 직후(첫 스텝만 돈 시점) 프로세스를 멈춰서 사실상 손실 없음
   - **다음에 이어서 할 것**: `python train_model_phase2.py` (conda 환경 `firstvenv` 활성화 후) 실행 → `models/genre_model.h5`를 불러와 상위 30레이어 해동 후 fine-tuning(최대 30 epoch) 진행
2. `train_model_phase2.py` 완료 후 남은 할 일:
   - 실제 사진 업로드해서 전체 파이프라인(장르 5-class + 얼굴검출 + 주간/야간 태그) 동작 확인
   - `reclassify_photos.py`는 지금 DB가 비어있어서 당장은 할 게 없음 — 새로 업로드된 사진이 쌓인 뒤 필요시 실행
   - `result.html`에 주간/야간 확률 바 추가할지 여부 결정 (요청은 받았으나 아직 답 안 나옴)

### 나중에 고려할 것 (아직 시작 안 함)

- **실내/실외 이진 분류기** — day/night과 같은 패턴(Kaggle 데이터셋 찾기 → prepare/train/predict 스크립트 → `app.py` 연동)으로 추가 가능. 아직 데이터셋도 안 정함.
- **세션/프로세스 지속성 주의**: 이 세션에서 띄운 백그라운드 학습 프로세스는 VSCode/세션을 완전히 닫으면 같이 종료될 가능성이 높음 (자식 프로세스로 추정). 학습 중엔 세션을 켜두거나, 완전 분리된 프로세스로 재시작해야 안전하게 지속됨.
- 환경: 실제 패키지(tensorflow/opencv/kagglehub/mediapipe 등)는 conda 환경 **`firstvenv`**(Python 3.12)에 설치돼 있음. 프로젝트 루트의 `.venv`는 비어있는 미사용 환경이므로 혼동 주의.

### GPU 문제 — 로컬은 CPU라 학습이 느림

이 컴퓨터엔 GPU가 없거나 TensorFlow가 GPU를 못 잡고 있어서, 학습이 전부 CPU로 돌아 에폭당 200초 이상 걸림 (MobileNetV2 전이학습처럼 가벼운 모델인데도). 데이터가 더 커지면 CPU로는 감당 안 됨 — GPU 필요.

로컬 GPU가 없을 때 현실적인 대안 (나중에 시도, 아직 진행 안 함):

- **Google Colab (추천)**: 이미 만들어둔 `dataset/` 폴더를 zip으로 압축 → Google Drive 업로드 → Colab에서 런타임 유형을 GPU(T4)로 변경 → Drive 마운트 후 압축 해제 → `train_model.py`/`train_model_phase2.py` 코드를 노트북에 올려 실행 → 끝나면 `models/genre_model.h5`를 다시 Drive→로컬로 다운로드. Kaggle 인증 재설정 불필요, 지금 상태를 그대로 이어가기 가장 쉬움.
- **Kaggle Notebook**: kaggle.com/code에서 새 노트북 생성 → "Add Input"으로 intel-image-classification / food41 / caltech-101 / human-faces 데이터셋을 그대로 추가 (업로드 불필요) → Settings에서 Accelerator를 GPU로 설정 → `prepare_dataset.py`/`train_model.py` 로직을 `/kaggle/input/...` 경로에 맞게 고쳐서 실행. 업로드 용량 제한이 없다는 장점.

둘 다 무료 티어로 지금 규모의 전이학습에는 충분히 빠름. 브라우저 작업이 필요해서 사용자가 직접 진행해야 하고, Claude가 대신 클릭할 수는 없음 — 필요한 노트북 코드/스크립트 준비는 요청 시 도와줄 수 있음.